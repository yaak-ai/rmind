"""Host-side decode over the `heads` ONNX/TensorRT engine outputs.

The `heads` export emits the raw code-logits `(b, G, C)` and offset table
`(b, G, C, action_dim)`; this module turns those into an action chunk under any
decode strategy — including stochastic ones (entropy-gated sampling) whose RNG
must not live in a deterministic TensorRT engine. It is pure NumPy/torch, has no
rmind-model dependency, and is meant to be vendored/ported into drivr.

Requires the tokenizer decode table (`decode_table_<tok>.pt`: invert of all
`C**G` code tuples, `(C**G, action_dim)`) and, for chain_greedy, the chain LUTs
(`luts_<tok>.pt`). Both are shipped in the branch and are tokenizer-specific.

    dt = torch.load("decode_table_y74asdtd.pt")["decode_table"]
    luts = torch.load("luts_y74asdtd.pt")["luts"]
    action = decode(code_logits, offsets, dt, strategy="chain_greedy", luts=luts)
"""

from __future__ import annotations

import torch
from torch import Tensor


def _gather_offset(offsets: Tensor, codes: Tensor) -> Tensor:
    idx = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
    return offsets.gather(2, idx).squeeze(2).sum(dim=1)  # (b, action_dim)


def _pack(codes: Tensor, c: int) -> Tensor:
    out = torch.zeros(codes.shape[0], dtype=torch.long, device=codes.device)
    for g in range(codes.shape[1]):
        out = out * c + codes[:, g]
    return out


def _argmax(code_logits: Tensor) -> Tensor:
    return code_logits.argmax(dim=-1)


def _chain_greedy(code_logits: Tensor, luts: list[Tensor], beta: float) -> Tensor:
    _, g, c = code_logits.shape
    logp = code_logits.log_softmax(dim=-1)
    prefix = torch.zeros(
        code_logits.shape[0], dtype=torch.long, device=code_logits.device
    )
    codes: list[Tensor] = []
    for level in range(g):
        prior = luts[level].to(code_logits)[prefix]
        code = (logp[:, level] + beta * prior).argmax(dim=-1)
        codes.append(code)
        prefix = prefix * c + code
    return torch.stack(codes, dim=1)


def _entropy_gated_q0(
    code_logits: Tensor, tau: float, gate_nats: float, generator: torch.Generator | None
) -> Tensor:
    """Sample q0 (temperature tau) only where its entropy exceeds gate_nats;
    argmax elsewhere and for the residual quantizers. Stochastic — seed via
    `generator` for reproducibility. This is the sample-primary-only strategy
    that recovered launches in the closed-loop sampling bench.
    """
    logp0 = (code_logits[:, 0] / tau).log_softmax(dim=-1)
    p0 = code_logits[:, 0].softmax(dim=-1)
    ent = -(p0 * p0.clamp_min(1e-12).log()).sum(dim=-1)
    sampled = torch.multinomial(logp0.exp(), 1, generator=generator).squeeze(-1)
    q0 = torch.where(ent > gate_nats, sampled, code_logits[:, 0].argmax(dim=-1))
    rest = code_logits[:, 1:].argmax(dim=-1)
    return torch.cat([q0[:, None], rest], dim=1)


def decode(  # noqa: PLR0913
    code_logits: Tensor,
    offsets: Tensor,
    decode_table: Tensor,
    *,
    strategy: str = "argmax",
    luts: list[Tensor] | None = None,
    beta: float = 1.0,
    tau: float = 0.7,
    gate_nats: float = 1.56,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Decode heads-engine outputs to an action chunk `(b, action_dim)`.

    strategy: "argmax" | "chain_greedy" (needs `luts`) | "entropy_gated"
    (stochastic; sample the primary quantizer in high-entropy states).

    Raises:
        ValueError: on an unknown strategy or chain_greedy without luts.
    """
    c = code_logits.shape[-1]
    if strategy == "argmax":
        codes = _argmax(code_logits)
    elif strategy == "chain_greedy":
        if luts is None:
            msg = "chain_greedy needs luts"
            raise ValueError(msg)
        codes = _chain_greedy(code_logits, luts, beta)
    elif strategy == "entropy_gated":
        codes = _entropy_gated_q0(code_logits, tau, gate_nats, generator)
    else:
        msg = f"unknown strategy {strategy!r}"
        raise ValueError(msg)
    return decode_table.to(code_logits)[_pack(codes, c)] + _gather_offset(
        offsets, codes
    )
