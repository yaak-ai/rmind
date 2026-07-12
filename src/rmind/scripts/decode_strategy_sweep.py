"""Decode-strategy sweep for the VQ-BeT joint policy (factorized code head).

Self-contained (plain python + torch, no hydra, no rmind imports). Loads:
  * an eval cache produced by `rmind.scripts.offset_head_retrain extract`
    (shards `shard_*.pt` with features (b,1152) fp16, code_logits (b,4,16)
    fp16, target_codes (b,4) int8, target (b,24) fp32 + meta.json),
  * a lightning checkpoint, from which it rebuilds
      - the offset head  (torchvision.ops.MLP(1152,[1024,1024,1536]) layout,
        state dict under objectives.policy.offset_head.*), and
      - a minimal ActionTokenizer (encoder/RVQ-codebooks/decoder + the
        turn-signal Scaler) for forward()/invert(); the manual RVQ
        lookup/assign was verified bit-exact against vector_quantize_pytorch
        on the real checkpoint weights.

It then recomputes per-sample offset tables, builds the (65536,24) decode
table, derives live_mask/usage from the TRAIN cache (GT codes recomputed with
THIS checkpoint's tokenizer: chunk=target.reshape(-1,6,4); chunk[...,3]*=2;
tokenizer(chunk)), runs a zoo of decode strategies (deterministic +
stochastic, each with and without the rsim deployment mutex), computes
conflict/coverage/L1/dead-tuple/match metrics, and ranks strategies under an
L1 guardrail.

Strategy provenance is cited per cell: "task" (explicitly required by the
sweep spec), "safety-S<k>" and/or "prob-S<k>" (the two proposal texts).

Mutex semantics mirror rsim.adapter.ControlAdapter's default path exactly
(/home/max/Code/rsim/src/rsim/adapter.py): per frame,
if brake > 0.05: gas = 0; then clamp gas/brake to [0,1] and steer to [-1,1].
(The adapter's steering EMA is temporal and is NOT applied here; turn signal
is passed through.)

Smoke:
    uv run python decode_strategy_sweep.py \
        --cache /home/max/Code/rmind-rqv/caches/offset_head/val \
        --ckpt /home/max/Code/rmind-rqv/artifacts/model-2gqxhjod:v9/model.ckpt \
        --max-samples 2000 --device cpu
"""

# Diagnostic sweep engine: results published from this exact code, so lint is
# relaxed file-wide for stylistic rules rather than risking behavior drift.
# ruff: noqa: ANN201, ARG001, N803, N806, N815, PLR2004, PLR0912, PLR0913, PLR0914, PLR0915, PLR0917, FBT001, FBT002, FBT003, C901, DOC501

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

G, C, ACTION_DIM, N_FRAMES, N_FIELDS = 4, 16, 24, 6, 4
N_COMBOS = C**G  # 65536
FEATURE_DIM = 1152

GAS_DIM, BRAKE_DIM, STEER_DIM, TURN_DIM = 0, 1, 2, 3
FIELD_NAMES = ("gas", "brake", "steer", "turn")

# exact sensor thresholds from rmind.scripts.pedal_conflict_stats
GAS_THRESH = 1.0 / 255 + 0.001  # ~0.0049
BRAKE_THRESH = 1.0 / 164 + 0.001  # ~0.0071

MUTEX_THRESHOLD = 0.05  # rsim.config.BRAKE_GAS_MUTEX_THRESHOLD

DEFAULT_TRAIN_CACHE = "/home/max/Code/rmind-rqv/caches/offset_head/train"
SCRATCH = Path(__file__).resolve().parent


def pack(codes: Tensor) -> Tensor:
    """(…,4) codes -> flat combo id: c0*4096 + c1*256 + c2*16 + c3."""
    return (
        codes[..., 0].long() * 4096
        + codes[..., 1].long() * 256
        + codes[..., 2].long() * 16
        + codes[..., 3].long()
    )


def unpack(ids: Tensor) -> Tensor:
    """flat combo id -> (…,4) codes."""
    return torch.stack(
        [(ids // 4096) % 16, (ids // 256) % 16, (ids // 16) % 16, ids % 16], dim=-1
    )


# ---------------------------------------------------------------------------
# Checkpoint loading: offset head + minimal tokenizer
# ---------------------------------------------------------------------------


def _load_state_dict(ckpt_path: Path) -> dict[str, Tensor]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_offset_head(sd: dict[str, Tensor]) -> torch.nn.Module:
    """Rebuild the offset head with the torchvision.ops.MLP(1152,[1024,1024,1536])
    module layout (Linear at indices 0/3/6) and load its checkpoint weights."""
    head = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.0),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.0),
        torch.nn.Linear(1024, G * C * ACTION_DIM),
        torch.nn.Dropout(0.0),
    )
    prefix = "objectives.policy.offset_head."
    sub = {k[len(prefix) :]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        msg = f"no offset head weights under {prefix!r} in checkpoint"
        raise SystemExit(msg)
    head.load_state_dict(sub)
    head.eval().requires_grad_(False)
    return head


class MiniTokenizer:
    """Frozen ActionTokenizer rebuilt from raw state-dict tensors.

    encoder: Linear(24,128) -> GELU -> Linear(128,384)
    quantizer: 4-level residual VQ, euclidean codebooks (16,384) each
        (manual lookup/greedy-assign verified == vector_quantize_pytorch)
    decoder: Linear(384,128) -> GELU -> Linear(128,24)
    normalizer: identity on gas/brake/steer, Scaler(in_range,out_range) on
        turn_signal (raw {0,1,2} -> {0,0.5,1} with the checkpoint's ranges).
    """

    def __init__(self, sd: dict[str, Tensor]) -> None:
        p = "objectives.policy.tokenizer."

        def get(name: str) -> Tensor:
            key = p + name
            if key not in sd:
                msg = f"missing tokenizer tensor {key!r} in checkpoint"
                raise SystemExit(msg)
            return sd[key].float()

        self.enc0_w, self.enc0_b = get("encoder.0.weight"), get("encoder.0.bias")
        self.enc2_w, self.enc2_b = get("encoder.2.weight"), get("encoder.2.bias")
        self.dec0_w, self.dec0_b = get("decoder.0.weight"), get("decoder.0.bias")
        self.dec2_w, self.dec2_b = get("decoder.2.weight"), get("decoder.2.bias")
        self.codebooks = torch.stack([
            get(f"quantizer.vq.layers.{g}._codebook.embed").squeeze(0) for g in range(G)
        ])  # (4,16,384)
        turn_in = get("input_transform.1.discrete.turn_signal.in_range")
        turn_out = get("input_transform.1.discrete.turn_signal.out_range")
        self.turn_in, self.turn_out = turn_in.tolist(), turn_out.tolist()

    def _normalize_flat(self, action: Tensor) -> Tensor:
        """(n,24) raw stacked [gas,brake,steer,turn]x6 -> normalized (turn scaled)."""
        out = action.clone()
        i0, i1 = self.turn_in
        o0, o1 = self.turn_out
        out[..., TURN_DIM::N_FIELDS] = (action[..., TURN_DIM::N_FIELDS] - i0) / (
            i1 - i0
        ) * (o1 - o0) + o0
        return out

    def encode(self, chunk: Tensor) -> Tensor:
        """(n,6,4) raw chunk (turn in {0,1,2}) -> (n,4) codes (greedy RVQ assign)."""
        flat = self._normalize_flat(chunk.reshape(-1, ACTION_DIM))
        z = F.linear(
            F.gelu(F.linear(flat, self.enc0_w, self.enc0_b)), self.enc2_w, self.enc2_b
        )
        residual = z
        codes = []
        for g in range(G):
            idx = torch.cdist(residual, self.codebooks[g]).argmin(-1)
            codes.append(idx)
            residual -= self.codebooks[g][idx]
        return torch.stack(codes, dim=-1)

    def invert(self, codes: Tensor) -> Tensor:
        """(n,4) codes -> (n,24) normalized action (sum of codebook vecs -> decoder)."""
        z_q = sum(self.codebooks[g][codes[:, g].long()] for g in range(G))
        return F.linear(
            F.gelu(F.linear(z_q, self.dec0_w, self.dec0_b)), self.dec2_w, self.dec2_b
        )


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------


def load_cache(
    cache_dir: Path, keys: tuple[str, ...], max_samples: int | None
) -> dict[str, Tensor]:
    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_paths:
        msg = f"no shards found in {cache_dir}"
        raise SystemExit(msg)
    out: dict[str, list[Tensor]] = {k: [] for k in keys}
    n = 0
    for path in shard_paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        take = shard[keys[0]].shape[0]
        if max_samples is not None:
            take = min(take, max_samples - n)
        for k in keys:
            out[k].append(shard[k][:take])
        del shard
        n += take
        if max_samples is not None and n >= max_samples:
            break
    return {k: torch.cat(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Precomputation: offsets, decode table, usage / live set, priors, LUTs
# ---------------------------------------------------------------------------


def compute_offsets(
    head: torch.nn.Module, features: Tensor, batch: int, device: str
) -> Tensor:
    """(n,1152) fp16 features -> (n,4,16,24) fp32 offset tables (batched)."""
    chunks = []
    with torch.inference_mode():
        for s in range(0, features.shape[0], batch):
            x = features[s : s + batch].to(device).float()
            chunks.append(head(x).reshape(-1, G, C, ACTION_DIM).cpu())
    return torch.cat(chunks)


def build_decode_table(tok: MiniTokenizer, batch: int = 8192) -> Tensor:
    ids = torch.arange(N_COMBOS)
    codes = unpack(ids)
    parts = [tok.invert(codes[s : s + batch]) for s in range(0, N_COMBOS, batch)]
    return torch.cat(parts)  # (65536,24) normalized space


def build_usage(
    tok: MiniTokenizer, train_cache: Path, max_samples: int | None
) -> Tensor:
    """usage[combo] = train count of the GT tuple under THIS tokenizer."""
    target = load_cache(train_cache, ("target",), max_samples)["target"]
    chunk = target.reshape(-1, N_FRAMES, N_FIELDS).clone()
    chunk[..., TURN_DIM] *= 2.0  # normalized {0,.5,1} -> raw turn {0,1,2}
    counts = torch.zeros(N_COMBOS, dtype=torch.long)
    for s in range(0, chunk.shape[0], 8192):
        codes = tok.encode(chunk[s : s + 8192])
        counts += torch.bincount(pack(codes), minlength=N_COMBOS)
    return counts


def build_clamp_table(
    decode_table: Tensor, live_ids: Tensor, weights: Tensor
) -> Tensor:
    """clamp[d] = live combo id nearest to d in (weighted) L1 decode space."""
    clamp = torch.arange(N_COMBOS)
    live_mask = torch.zeros(N_COMBOS, dtype=torch.bool)
    live_mask[live_ids] = True
    dead_ids = (~live_mask).nonzero(as_tuple=True)[0]
    live_dec = decode_table[live_ids] * weights
    for s in range(0, dead_ids.shape[0], 1024):
        d = dead_ids[s : s + 1024]
        dist = torch.cdist(decode_table[d] * weights, live_dec, p=1)
        clamp[d] = live_ids[dist.argmin(-1)]
    return clamp


def build_chain_luts(usage: Tensor, alpha: float = 0.5) -> list[Tensor]:
    """Empirical chain conditionals log P(c_g | c_<g) from the usage histogram
    (safety-S5): mask exact-zero continuations to -inf, Laplace-smooth among
    nonzero ones with `alpha`."""
    u = usage.reshape(C, C, C, C).float()
    luts = []
    for g in range(G):
        cnt = u.sum(dim=tuple(range(g + 1, G))) if g < G - 1 else u  # (16,)*g+1
        cnt = cnt.reshape(-1, C) if g > 0 else cnt.reshape(1, C)  # (16^g, 16)
        nz = cnt > 0
        sm = cnt + alpha * nz.float()
        denom = sm.sum(-1, keepdim=True).clamp_min(1e-9)
        logp = torch.where(
            nz, (sm / denom).clamp_min(1e-30).log(), torch.full_like(cnt, -1e9)
        )
        # unseen prefix (all-zero row): neutral prior (0) so the model logits decide
        logp = torch.where(nz.any(-1, keepdim=True), logp, torch.zeros_like(logp))
        luts.append(logp)  # T[g]: (16^g, 16)
    return luts


@dataclass
class Globals:
    """Everything shared across strategies / eval batches (built once)."""

    decode_table: Tensor  # (65536,24)
    usage: Tensor  # (65536,) long
    live_ids: Tensor  # (K,) long, usage >= 1
    L: Tensor  # (K,4) codes of live combos
    A0: Tensor  # (K,24) base decodes of live combos
    live_mask_full: Tensor  # (65536,) bool
    log1p_usage_live: Tensor  # (K,)
    log_pi_live: Tensor  # (K,) smoothed log joint prob
    pmi_live: Tensor  # (K,) log pi - sum_g log marginal
    log_usage_p_full: Tensor  # (65536,) Laplace-smoothed log usage prob
    usage_live: Tensor  # (K,)
    clamp_table: Tensor  # (65536,)
    clamp_table_pedalw: Tensor  # (65536,) with gas/brake dims weighted x2
    chain_luts: list[Tensor]
    q0_sorted_order: Tensor  # (K,) live indices sorted by q0 code
    q0_row_offsets: Tensor  # (17,)
    q0_entropy: Tensor  # (n,) eval-set q0 softmax entropy
    q0_quart_edges: tuple[float, float, float]
    q0_quartile: Tensor  # (n,) int 0..3
    joint_H: Tensor  # (n,) live-joint entropy of softmax(s)
    joint_H_pcts: dict[str, float]  # percentiles of joint_H (70/75/90)
    q0H_pcts: dict[str, float]


def build_globals(
    tok: MiniTokenizer,
    train_cache: Path,
    lp_all: Tensor,
    max_train_samples: int | None,
    eval_batch: int,
) -> Globals:
    time.perf_counter()
    decode_table = build_decode_table(tok)
    usage = build_usage(tok, train_cache, max_train_samples)
    live_ids = (usage > 0).nonzero(as_tuple=True)[0]
    live_ids.shape[0]
    L = unpack(live_ids)
    A0 = decode_table[live_ids]
    live_mask_full = torch.zeros(N_COMBOS, dtype=torch.bool)
    live_mask_full[live_ids] = True

    usage_live = usage[live_ids].float()
    n_train = float(usage.sum())
    alpha = 1.0
    pi = (usage_live + alpha) / (n_train + alpha * N_COMBOS)
    log_pi_live = pi.log()
    # per-group empirical marginals of the live/train distribution
    marg = torch.zeros(G, C)
    for g in range(G):
        marg[g] = torch.bincount(L[:, g], weights=usage_live, minlength=C)
    marg = (marg + alpha) / (marg.sum(-1, keepdim=True) + alpha * C)
    pmi_live = log_pi_live - sum(marg[g].log()[L[:, g]] for g in range(G))
    log_usage_p_full = ((usage.float() + alpha) / (n_train + alpha * N_COMBOS)).log()

    clamp_table = build_clamp_table(decode_table, live_ids, torch.ones(ACTION_DIM))
    w = torch.ones(ACTION_DIM)
    w[GAS_DIM::N_FIELDS] = 2.0
    w[BRAKE_DIM::N_FIELDS] = 2.0
    clamp_table_pedalw = build_clamp_table(decode_table, live_ids, w)

    chain_luts = build_chain_luts(usage)

    order = torch.argsort(L[:, 0], stable=True)
    counts0 = torch.bincount(L[:, 0], minlength=C)
    row_offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts0.cumsum(0)])

    # eval-set entropies (pre-pass): q0 softmax entropy + joint live entropy
    p0 = lp_all[:, 0].softmax(-1)
    q0_entropy = -(p0 * p0.clamp_min(1e-12).log()).sum(-1)
    joint_H = torch.empty(lp_all.shape[0])
    for s in range(0, lp_all.shape[0], eval_batch):
        lp = lp_all[s : s + eval_batch]
        sc = live_scores(lp, L)
        ps = sc.softmax(-1)
        joint_H[s : s + eval_batch] = -(ps * ps.clamp_min(1e-12).log()).sum(-1)

    e1, e2, e3 = (float(torch.quantile(q0_entropy, q)) for q in (0.25, 0.5, 0.75))
    quart = (
        (q0_entropy > e1).long() + (q0_entropy > e2).long() + (q0_entropy > e3).long()
    )
    jp = {f"p{p}": float(torch.quantile(joint_H, p / 100)) for p in (70, 75, 90)}
    qp = {f"p{p}": float(torch.quantile(q0_entropy, p / 100)) for p in (70, 75, 90)}

    return Globals(
        decode_table=decode_table,
        usage=usage,
        live_ids=live_ids,
        L=L,
        A0=A0,
        live_mask_full=live_mask_full,
        log1p_usage_live=usage_live.log1p(),
        log_pi_live=log_pi_live,
        pmi_live=pmi_live,
        log_usage_p_full=log_usage_p_full,
        usage_live=usage_live,
        clamp_table=clamp_table,
        clamp_table_pedalw=clamp_table_pedalw,
        chain_luts=chain_luts,
        q0_sorted_order=order,
        q0_row_offsets=row_offsets,
        q0_entropy=q0_entropy,
        q0_quart_edges=(e1, e2, e3),
        q0_quartile=quart,
        joint_H=joint_H,
        joint_H_pcts=jp,
        q0H_pcts=qp,
    )


def live_scores(lp: Tensor, L: Tensor) -> Tensor:
    """(b,4,16) log-marginals -> (b,K) summed log-marginal score of live combos."""
    return sum(lp[:, g].index_select(1, L[:, g]) for g in range(G))


# ---------------------------------------------------------------------------
# Per-batch context handed to strategies
# ---------------------------------------------------------------------------


@dataclass
class BatchCtx:
    lp: Tensor  # (b,4,16) log_softmax of cached code logits
    offsets: Tensor  # (b,4,16,24) offset tables
    s: Tensor  # (b,K) live-joint scores
    g: Globals

    def gathered_offset(self, codes: Tensor) -> Tensor:
        """(b,4) codes -> (b,24) summed per-quantizer offset."""
        b = codes.shape[0]
        idx = torch.arange(b)
        return sum(self.offsets[idx, g, codes[:, g].long()] for g in range(G))

    def decode(self, codes: Tensor) -> Tensor:
        """codes -> action = decode_table[pack] + gathered offsets."""
        return self.g.decode_table[pack(codes)] + self.gathered_offset(codes)

    def offsets_at_live(self, rows: Tensor | None = None) -> Tensor:
        """(b,K,24) summed offsets at every live combo (heavy; batch upstream)."""
        L = self.g.L
        off = self.offsets if rows is None else self.offsets[rows]
        return sum(off[:, g, L[:, g]] for g in range(G))


@dataclass
class StratOut:
    codes: Tensor  # (b,4) emitted tuple
    action: Tensor  # (b,24) decoded action (may be a blend for expected-action)


def _gumbel(shape: tuple[int, ...], gen: torch.Generator) -> Tensor:
    u = torch.rand(shape, generator=gen).clamp(1e-9, 1 - 1e-9)
    return -(-u.log()).log()


def _masked_live_argmax(ctx: BatchCtx, score: Tensor, valid: Tensor) -> StratOut:
    """argmax of `score` over live combos where `valid`; rows with no valid
    candidate fall back to plain per-group argmax."""
    masked = score.masked_fill(~valid, -1e30)
    k = masked.argmax(-1)
    codes = ctx.g.L[k]
    none_valid = ~valid.any(-1)
    if none_valid.any():
        codes[none_valid] = ctx.lp[none_valid].argmax(-1)
    return StratOut(codes, ctx.decode(codes))


def _q0_member(ctx: BatchCtx, c0: Tensor) -> Tensor:
    """(b,) chosen q0 codes -> (b,K) mask of live combos sharing that q0."""
    return ctx.g.L[:, 0][None, :] == c0[:, None]


# ---------------------------------------------------------------------------
# Strategy zoo. Each: fn(ctx, gen) -> StratOut
# ---------------------------------------------------------------------------


def strat_argmax(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
    codes = ctx.lp.argmax(-1)
    return StratOut(codes, ctx.decode(codes))


def make_deadclamp(table_attr: str):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        table = getattr(ctx.g, table_attr)
        codes = unpack(table[pack(ctx.lp.argmax(-1))])
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_livejoint_argmax(n_min: int):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        valid = (ctx.g.usage_live >= n_min)[None, :].expand_as(ctx.s)
        return _masked_live_argmax(ctx, ctx.s, valid)

    return fn


def make_lattice_argmax(k: int, lam: float):
    """safety-S1: live-masked joint argmax over the per-group top-k product
    lattice, + lam*log usage_p; fallback to plain argmax if lattice all-dead."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        b = ctx.lp.shape[0]
        topk = ctx.lp.topk(k, dim=-1).indices  # (b,4,k)
        lut = torch.zeros(b, G, C, dtype=torch.bool)
        lut.scatter_(2, topk, True)
        L = ctx.g.L
        member = (
            lut[:, 0, L[:, 0]]
            & lut[:, 1, L[:, 1]]
            & lut[:, 2, L[:, 2]]
            & lut[:, 3, L[:, 3]]
        )
        score = ctx.s + lam * ctx.g.log_usage_p_full[ctx.g.live_ids][None, :]
        return _masked_live_argmax(ctx, score, member)

    return fn


def make_usage_prior_map(lam: float):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        score = ctx.s + lam * ctx.g.log1p_usage_live[None, :]
        k = score.argmax(-1)
        codes = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_prior_map(prior_attr: str, lam: float):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        score = ctx.s + lam * getattr(ctx.g, prior_attr)[None, :]
        codes = ctx.g.L[score.argmax(-1)]
        return StratOut(codes, ctx.decode(codes))

    return fn


def _risk_penalty(A: Tensor, theta: float, squared: bool) -> Tensor:
    """A (b,K,24) decoded actions -> per-candidate conflict penalty."""
    frames = A.reshape(*A.shape[:-1], N_FRAMES, N_FIELDS)
    both = torch.minimum(frames[..., GAS_DIM], frames[..., BRAKE_DIM])  # (b,K,6)
    if squared:  # safety-S4b: sum_t relu(min(gas,brake)-theta)^2
        return F.relu(both - theta).pow(2).sum(-1)
    # task spec: conflict_margin = max_frame min(gas,brake); pen = relu(margin-theta)
    return F.relu(both.max(-1).values - theta)


def make_risk_map(mu: float, theta: float, squared: bool, inner: int = 256):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        b = ctx.s.shape[0]
        codes = torch.empty(b, G, dtype=torch.long)
        for s0 in range(0, b, inner):
            rows = torch.arange(s0, min(s0 + inner, b))
            A = ctx.g.A0[None, :, :] + ctx.offsets_at_live(rows)  # (bs,K,24)
            pen = _risk_penalty(A, theta, squared)
            k = (ctx.s[rows] - mu * pen).argmax(-1)
            codes[rows] = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_chain_greedy(beta: float):
    """safety-S5: greedy coarse->fine decode with empirical chain conditionals."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        b = ctx.lp.shape[0]
        prefix = torch.zeros(b, dtype=torch.long)
        out = []
        for g in range(G):
            cond = ctx.g.chain_luts[g][prefix]  # (b,16)
            c = (ctx.lp[:, g] + beta * cond).argmax(-1)
            out.append(c)
            prefix = prefix * C + c
        codes = torch.stack(out, -1)
        return StratOut(codes, ctx.decode(codes))

    return fn


def _q0_conditional_residual(ctx: BatchCtx, c0: Tensor, m_min: int) -> StratOut:
    """Given per-sample q0 codes, argmax the residual live-joint score within
    L(c0); rows with < m_min live continuations fall back to full live argmax."""
    member = _q0_member(ctx, c0)
    small = member.sum(-1) < m_min
    valid = torch.where(small[:, None], torch.ones_like(member), member)
    return _masked_live_argmax(ctx, ctx.s, valid)


def make_q0cond_residual_argmax(m_min: int = 4):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        return _q0_conditional_residual(ctx, ctx.lp[:, 0].argmax(-1), m_min)

    return fn


def strat_q0cond_nearest_decode(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
    """prob-S4 decode-space variant: snap the unconstrained greedy tuple's base
    decode to the nearest live decode sharing its q0 (L1, no offsets)."""
    greedy = ctx.lp.argmax(-1)
    target_dec = ctx.g.decode_table[pack(greedy)]  # (b,24)
    member = _q0_member(ctx, greedy[:, 0])
    dist = (ctx.g.A0[None, :, :] - target_dec[:, None, :]).abs().sum(-1)  # (b,K)
    masked = dist.masked_fill(~member, 1e30)
    k = masked.argmin(-1)
    codes = ctx.g.L[k]
    none = ~member.any(-1)
    if none.any():
        codes[none] = ctx.g.L[dist[none].argmin(-1)]
    return StratOut(codes, ctx.decode(codes))


def make_expected_action(top: int = 8):
    """task cell (expected to conflict): probability-weighted blend of the
    decodes of the top-`top` live combos; emitted tuple = top-1."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        vals, idx = ctx.s.topk(top, dim=-1)  # (b,top)
        w = vals.softmax(-1)
        b = ctx.s.shape[0]
        L_sel = ctx.g.L[idx]  # (b,top,4)
        base = ctx.g.A0[idx]  # (b,top,24)
        off = torch.zeros(b, top, ACTION_DIM)
        rows = torch.arange(b)[:, None].expand(b, top)
        for g in range(G):
            off += ctx.offsets[rows, g, L_sel[..., g]]
        action = ((base + off) * w[..., None]).sum(1)
        return StratOut(L_sel[:, 0], action)

    return fn


# --- stochastic -------------------------------------------------------------


def make_factorized_sample(tau: float, topk: int | None = None):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        lp = ctx.lp / tau
        if topk is not None:
            kth = lp.topk(topk, dim=-1).values[..., -1:]
            lp = lp.masked_fill(ctx.lp / tau < kth, -1e30)
        codes = (lp + _gumbel(lp.shape, gen)).argmax(-1)
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_q0_sample_res_argmax(tau: float, deadclamp: bool = False):
    """task: sample q0 at temperature tau, per-group argmax residuals
    (optionally then dead-clamp to the nearest live combo)."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        lp0 = ctx.lp[:, 0] / tau
        c0 = (lp0 + _gumbel(lp0.shape, gen)).argmax(-1)
        codes = ctx.lp.argmax(-1)
        codes[:, 0] = c0
        if deadclamp:
            codes = unpack(ctx.g.clamp_table[pack(codes)])
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_q0cond_sample(tau0: float, m_min: int = 4):
    """prob-S4: sample q0 (tau0), then live-conditional residual argmax."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        lp0 = ctx.lp[:, 0] / tau0
        c0 = (lp0 + _gumbel(lp0.shape, gen)).argmax(-1)
        return _q0_conditional_residual(ctx, c0, m_min)

    return fn


def make_livejoint_sample(tau: float):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        k = (ctx.s / tau + _gumbel(ctx.s.shape, gen)).argmax(-1)
        codes = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_livejoint_nucleus(p_nuc: float, tau: float = 1.0):
    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        sorted_s, order = ctx.s.sort(-1, descending=True)
        probs = sorted_s.softmax(-1)
        keep = probs.cumsum(-1) - probs < p_nuc  # always keeps top-1
        masked = (sorted_s / tau).masked_fill(~keep, -1e30)
        pick = (masked + _gumbel(masked.shape, gen)).argmax(-1)
        k = order.gather(1, pick[:, None]).squeeze(1)
        codes = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_entropy_gated_sample(pct_key: str, tau: float):
    """safety-S3 (gate) x prob-S1 (live-joint sample): sample the live joint at
    temperature tau only when q0 entropy exceeds the given eval percentile,
    else live-joint argmax."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        h0 = ctx.g.q0H_pcts[pct_key]
        p0 = ctx.lp[:, 0].softmax(-1)
        H = -(p0 * p0.clamp_min(1e-12).log()).sum(-1)
        k_map = ctx.s.argmax(-1)
        k_smp = (ctx.s / tau + _gumbel(ctx.s.shape, gen)).argmax(-1)
        k = torch.where(h0 < H, k_smp, k_map)
        codes = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


def make_entropy_adaptive_tau(tau_min: float, tau_max: float):
    """prob-S3: tau(H) linear between the 70th/90th percentile of joint live
    entropy; Gumbel-max with per-sample temperature."""

    def fn(ctx: BatchCtx, gen: torch.Generator) -> StratOut:
        h_lo, h_hi = ctx.g.joint_H_pcts["p70"], ctx.g.joint_H_pcts["p90"]
        ps = ctx.s.softmax(-1)
        H = -(ps * ps.clamp_min(1e-12).log()).sum(-1)
        frac = ((H - h_lo) / max(h_hi - h_lo, 1e-9)).clamp(0, 1)
        tau = (tau_min + (tau_max - tau_min) * frac).clamp_min(0.05)
        k = (ctx.s / tau[:, None] + _gumbel(ctx.s.shape, gen)).argmax(-1)
        codes = ctx.g.L[k]
        return StratOut(codes, ctx.decode(codes))

    return fn


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


@dataclass
class Strategy:
    name: str
    proposer: str  # citation: task / safety-S<k> / prob-S<k>
    fn: Callable[[BatchCtx, torch.Generator], StratOut]
    stochastic: bool = False


def build_strategies() -> list[Strategy]:
    S: list[Strategy] = []
    add = S.append

    # --- deterministic -----------------------------------------------------
    add(Strategy("argmax", "task (baseline; export parity)", strat_argmax))
    add(Strategy("argmax_deadclamp", "task + safety-S2", make_deadclamp("clamp_table")))
    add(
        Strategy(
            "argmax_deadclamp_pedalw2",
            "safety-S2 (gas/brake dims weighted x2 in the clamp metric)",
            make_deadclamp("clamp_table_pedalw"),
        )
    )
    add(
        Strategy(
            "livejoint_argmax",
            "task + prob-S1(MAP) + safety-S1(k=16,lam=0)",
            make_livejoint_argmax(1),
        )
    )
    add(
        Strategy(
            "livejoint_argmax_nmin5",
            "prob-S1 (n_min=5 live-set inclusion)",
            make_livejoint_argmax(5),
        )
    )
    for k in (4, 8):
        for lam in (0.0, 0.3):
            add(
                Strategy(
                    f"lattice_argmax_k{k}_lam{lam}",
                    "safety-S1 (top-k product lattice + usage prior)",
                    make_lattice_argmax(k, lam),
                )
            )
    for lam in (0.1, 0.3, 1.0):
        add(
            Strategy(
                f"usage_prior_map_lam{lam}",
                "task (live-joint score + lam*log1p(usage))",
                make_usage_prior_map(lam),
            )
        )
    for lam in (0.25, 0.5, 1.0):
        add(
            Strategy(
                f"pmi_map_lam{lam}",
                "prob-S2 (PMI-corrected product-of-experts)",
                make_prior_map("pmi_live", lam),
            )
        )
    for lam in (0.25, 0.5):
        add(
            Strategy(
                f"pi_map_lam{lam}",
                "prob-S2 (plain smoothed log-joint prior)",
                make_prior_map("log_pi_live", lam),
            )
        )
    for mu in (1.0, 3.0, 10.0):
        add(
            Strategy(
                f"risk_map_mu{mu:g}",
                "task + prob-S5 (live-joint - mu*relu(max_t min(gas,brake) - 0.02))",
                make_risk_map(mu, 0.02, squared=False),
            )
        )
    for mu in (10.0, 100.0):
        add(
            Strategy(
                f"risk_sq_theta05_mu{mu:g}",
                "safety-S4 (pedal-symmetric squared penalty, theta=0.05)",
                make_risk_map(mu, 0.05, squared=True),
            )
        )
    for beta in (0.3, 0.5, 1.0):
        add(
            Strategy(
                f"chain_greedy_beta{beta}",
                "safety-S5 (pseudo-AR empirical conditionals)",
                make_chain_greedy(beta),
            )
        )
    add(
        Strategy(
            "q0cond_residual_argmax",
            "prob-S4 (deterministic anchor)",
            make_q0cond_residual_argmax(),
        )
    )
    add(
        Strategy(
            "q0cond_nearest_decode",
            "prob-S4 (decode-space snap variant)",
            strat_q0cond_nearest_decode,
        )
    )
    add(
        Strategy(
            "expected_action_top8",
            "task (cautionary blend cell)",
            make_expected_action(8),
        )
    )

    # --- stochastic (3 seeds) ----------------------------------------------
    for tau in (0.5, 0.7, 1.0):
        add(
            Strategy(
                f"factorized_sample_tau{tau}",
                "task (naive factorized)",
                make_factorized_sample(tau),
                True,
            )
        )
    for k in (2, 4):
        add(
            Strategy(
                f"top{k}_factorized_sample_tau1.0",
                "task (top-k factorized)",
                make_factorized_sample(1.0, k),
                True,
            )
        )
    for tau in (0.5, 0.7, 1.0):
        add(
            Strategy(
                f"q0sample_tau{tau}_resargmax",
                "task (q0-sample + argmax residuals)",
                make_q0_sample_res_argmax(tau),
                True,
            )
        )
    add(
        Strategy(
            "q0sample_tau0.7_deadclamp",
            "task (q0-sample + dead-clamp)",
            make_q0_sample_res_argmax(0.7, deadclamp=True),
            True,
        )
    )
    for tau in (0.5, 0.8, 1.0):
        add(
            Strategy(
                f"q0cond_sample_tau{tau}",
                "prob-S4 (sampled anchor, live-conditional residuals)",
                make_q0cond_sample(tau),
                True,
            )
        )
    for tau in (0.5, 0.8, 1.0):
        add(
            Strategy(
                f"livejoint_sample_tau{tau}",
                "task + prob-S1 (softmax over live joint)",
                make_livejoint_sample(tau),
                True,
            )
        )
    add(
        Strategy(
            "livejoint_nucleus_p0.9",
            "prob-S3 (nucleus over live joint)",
            make_livejoint_nucleus(0.9),
            True,
        )
    )
    for pct in ("p75", "p90"):
        add(
            Strategy(
                f"entropy_gated_sample_{pct}_tau0.7",
                "safety-S3 (entropy gate, seeded Gumbel over live lattice)",
                make_entropy_gated_sample(pct, 0.7),
                True,
            )
        )
    for lo, hi in ((0.0, 0.8), (0.3, 1.0)):
        add(
            Strategy(
                f"entropy_adaptive_tau_{lo}_{hi}",
                "prob-S3 (entropy-adaptive temperature over live joint)",
                make_entropy_adaptive_tau(lo, hi),
                True,
            )
        )
    return S


# ---------------------------------------------------------------------------
# Mutex + metrics
# ---------------------------------------------------------------------------


def apply_mutex(action: Tensor) -> Tensor:
    """rsim ControlAdapter default path, per frame: if brake > 0.05 -> gas=0;
    then clamp gas/brake to [0,1], steer to [-1,1] (turn untouched)."""
    frames = action.reshape(-1, N_FRAMES, N_FIELDS).clone()
    gas, brake = frames[..., GAS_DIM], frames[..., BRAKE_DIM]
    gas = torch.where(brake > MUTEX_THRESHOLD, torch.zeros_like(gas), gas)
    frames[..., GAS_DIM] = gas.clamp(0.0, 1.0)
    frames[..., BRAKE_DIM] = brake.clamp(0.0, 1.0)
    frames[..., STEER_DIM] = frames[..., STEER_DIM].clamp(-1.0, 1.0)
    return frames.reshape(-1, ACTION_DIM)


@dataclass
class MetricAcc:
    """Streaming accumulator over eval batches for one (strategy, mutex) cell."""

    n_samples: int = 0
    n_frames: int = 0
    conflict_sensor: int = 0
    conflict_002: int = 0
    conflict_005: int = 0
    gas_active: int = 0
    gas_active_q: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    frames_q: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    l1_sum: float = 0.0
    l1_field_sum: list[float] = field(default_factory=lambda: [0.0] * N_FIELDS)
    dead: int = 0
    match: int = 0

    def update(
        self,
        action: Tensor,
        codes: Tensor,
        target: Tensor,
        target_codes: Tensor,
        quart: Tensor,
        usage: Tensor,
    ) -> None:
        b = action.shape[0]
        fr = action.reshape(b, N_FRAMES, N_FIELDS)
        gas, brake = fr[..., GAS_DIM], fr[..., BRAKE_DIM]
        self.n_samples += b
        self.n_frames += b * N_FRAMES
        self.conflict_sensor += int(((gas > GAS_THRESH) & (brake > BRAKE_THRESH)).sum())
        self.conflict_002 += int(((gas > 0.02) & (brake > 0.02)).sum())
        self.conflict_005 += int(((gas > 0.05) & (brake > 0.05)).sum())
        act = gas > GAS_THRESH
        self.gas_active += int(act.sum())
        for q in range(4):
            m = quart == q
            self.gas_active_q[q] += int(act[m].sum())
            self.frames_q[q] += int(m.sum()) * N_FRAMES
        diff = (action - target).abs()
        self.l1_sum += float(diff.sum())
        for f_ in range(N_FIELDS):
            self.l1_field_sum[f_] += float(diff[:, f_::N_FIELDS].sum())
        self.dead += int((usage[pack(codes)] == 0).sum())
        self.match += int((codes == target_codes.long()).all(-1).sum())

    def finalize(self, gt: dict[str, Any]) -> dict[str, Any]:
        nf = max(self.n_frames, 1)
        ns = max(self.n_samples, 1)
        gas_rate = self.gas_active / nf
        gas_q = [self.gas_active_q[q] / max(self.frames_q[q], 1) for q in range(4)]
        deficit_q = [gas_q[q] - gt["gas_activation_per_quartile"][q] for q in range(4)]
        return {
            "conflict_rate": {
                "sensor": self.conflict_sensor / nf,
                "0.02": self.conflict_002 / nf,
                "0.05": self.conflict_005 / nf,
            },
            "gas_activation": {
                "overall": gas_rate,
                "overall_deficit": gas_rate - gt["gas_activation_overall"],
                "per_quartile": gas_q,
                "deficit_per_quartile": deficit_q,
                "topH_deficit": deficit_q[3],
            },
            "l1": {
                "overall": self.l1_sum / (ns * ACTION_DIM),
                **{
                    FIELD_NAMES[f_]: self.l1_field_sum[f_] / (ns * N_FRAMES)
                    for f_ in range(N_FIELDS)
                },
            },
            "dead_tuple_rate": self.dead / ns,
            "tuple_match_rate": self.match / ns,
        }


def gt_stats(target: Tensor, quart: Tensor) -> dict[str, Any]:
    fr = target.reshape(-1, N_FRAMES, N_FIELDS)
    gas, brake = fr[..., GAS_DIM], fr[..., BRAKE_DIM]
    act = gas > GAS_THRESH
    per_q = []
    for q in range(4):
        m = quart == q
        per_q.append(float(act[m].float().mean()) if int(m.sum()) else 0.0)
    return {
        "gas_activation_overall": float(act.float().mean()),
        "gas_activation_per_quartile": per_q,
        "conflict_rate": {
            "sensor": float(
                ((gas > GAS_THRESH) & (brake > BRAKE_THRESH)).float().mean()
            ),
            "0.02": float(((gas > 0.02) & (brake > 0.02)).float().mean()),
            "0.05": float(((gas > 0.05) & (brake > 0.05)).float().mean()),
        },
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean": mean, "std": math.sqrt(var)}


def _aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Elementwise mean/std over seed metric dicts (same tree structure)."""

    def rec(nodes: list[Any]) -> Any:
        first = nodes[0]
        if isinstance(first, dict):
            return {k: rec([n[k] for n in nodes]) for k in first}
        if isinstance(first, list):
            return [rec([n[i] for n in nodes]) for i in range(len(first))]
        return _mean_std([float(n) for n in nodes])

    return rec(per_seed)


def _leaf(metrics: dict[str, Any], *path: str | int) -> float:
    """Read a metric leaf; seed-aggregated leaves are {mean,std} dicts."""
    node: Any = metrics
    for p in path:
        node = node[p]
    if isinstance(node, dict) and "mean" in node:
        return float(node["mean"])
    return float(node)


def run_sweep(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    torch.manual_seed(0)
    device = args.device
    out_json, out_md = Path(args.out_json), Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sd = _load_state_dict(Path(args.ckpt))
    head = load_offset_head(sd).to(device)
    tok = MiniTokenizer(sd)

    data = load_cache(
        Path(args.cache),
        ("features", "code_logits", "target_codes", "target"),
        args.max_samples,
    )
    if (
        args.sample_slice
    ):  # stability rerun: evaluate on a contiguous half/k-th of the eval set
        i, k = (int(x) for x in args.sample_slice.split("/"))
        n0 = data["features"].shape[0]
        lo, hi = (i - 1) * n0 // k, i * n0 // k
        data = {key: v[lo:hi] for key, v in data.items()}
    n = data["features"].shape[0]

    offsets_all = compute_offsets(head, data["features"], args.head_batch, device)
    lp_all = data["code_logits"].float().log_softmax(-1)
    target_all = data["target"].float()
    tcodes_all = data["target_codes"].long()

    glb = build_globals(
        tok, Path(args.train_cache), lp_all, args.max_train_samples, args.eval_batch
    )

    # sanity: dead-combo argmax rate + top-k recall diagnostics
    argmax_ids = pack(lp_all.argmax(-1))
    dead_argmax = float((~glb.live_mask_full[argmax_ids]).float().mean())
    tk_recall = {}
    for k in (2, 4, 8):
        topk = lp_all.topk(k, -1).indices
        hit = (topk == tcodes_all[..., None]).any(-1).all(-1)
        tk_recall[f"top{k}_all_groups"] = float(hit.float().mean())

    gt = gt_stats(target_all, glb.q0_quartile)
    strategies = build_strategies()
    if (
        args.strategies
    ):  # stability rerun: restrict to named cells (argmax stays as baseline)
        want = [s.strip() for s in args.strategies.split(",")]
        known = {s.name for s in strategies}
        unknown = [w for w in want if w not in known]
        if unknown:
            msg = f"unknown strategies: {unknown}"
            raise SystemExit(msg)
        if "argmax" not in want:
            msg = "--strategies must include 'argmax' (guardrail/ranking baseline)"
            raise SystemExit(msg)
        strategies = [s for s in strategies if s.name in want]
    # --seeds: int N -> [0..N-1] (original behavior); comma list -> explicit seeds
    seeds = [int(s) for s in str(args.seeds).split(",")]
    if len(seeds) == 1 and "," not in str(args.seeds):
        seeds = list(range(seeds[0]))

    # accumulators: cell key -> (mutex -> per-run list of MetricAcc)
    runs: list[tuple[Strategy, int | None, torch.Generator, dict[bool, MetricAcc]]] = []
    for st in strategies:
        for seed in seeds if st.stochastic else [None]:
            gen = torch.Generator().manual_seed(1000 + (seed or 0))
            runs.append((st, seed, gen, {False: MetricAcc(), True: MetricAcc()}))

    (n + args.eval_batch - 1) // args.eval_batch
    for _bi, s0 in enumerate(range(0, n, args.eval_batch)):
        s1 = min(s0 + args.eval_batch, n)
        ctx = BatchCtx(
            lp=lp_all[s0:s1],
            offsets=offsets_all[s0:s1],
            s=live_scores(lp_all[s0:s1], glb.L),
            g=glb,
        )
        target = target_all[s0:s1]
        tcodes = tcodes_all[s0:s1]
        quart = glb.q0_quartile[s0:s1]
        for st, _seed, gen, accs in runs:
            out = st.fn(ctx, gen)
            accs[False].update(out.action, out.codes, target, tcodes, quart, glb.usage)
            accs[True].update(
                apply_mutex(out.action), out.codes, target, tcodes, quart, glb.usage
            )

    # collect cells: strategy x mutex; stochastic aggregated over seeds
    cells: dict[str, dict[str, Any]] = {}
    for st in strategies:
        st_runs = [r for r in runs if r[0] is st]
        for mutex in (False, True):
            key = f"{st.name}|mutex={'on' if mutex else 'off'}"
            per_seed = [r[3][mutex].finalize(gt) for r in st_runs]
            metrics = (
                per_seed[0] if not st.stochastic else _aggregate_seed_metrics(per_seed)
            )
            cells[key] = {
                "strategy": st.name,
                "proposer": st.proposer,
                "mutex": mutex,
                "stochastic": st.stochastic,
                "n_seeds": len(per_seed),
                "metrics": metrics,
            }

    # ranking: guardrail vs argmax (same mutex variant), then lexicographic
    for mutex in (False, True):
        base = cells[f"argmax|mutex={'on' if mutex else 'off'}"]["metrics"]
        base_l1 = _leaf(base, "l1", "overall")
        base_steer = _leaf(base, "l1", "steer")
        for cell in cells.values():
            if cell["mutex"] is not mutex:
                continue
            m = cell["metrics"]
            cell["guardrail"] = {
                "l1_overall": _leaf(m, "l1", "overall"),
                "l1_overall_limit": base_l1 * 1.02,
                "steer_l1": _leaf(m, "l1", "steer"),
                "steer_l1_limit": base_steer * 1.02,
                "pass": bool(
                    _leaf(m, "l1", "overall") <= base_l1 * 1.02
                    and _leaf(m, "l1", "steer") <= base_steer * 1.02
                ),
            }

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, ...]:
        m = item[1]["metrics"]
        return (
            round(_leaf(m, "conflict_rate", "0.05"), 6),
            round(abs(_leaf(m, "gas_activation", "topH_deficit")), 6),
            round(_leaf(m, "conflict_rate", "0.02"), 6),
            round(_leaf(m, "conflict_rate", "sensor"), 6),
        )

    passers = sorted(
        (kv for kv in cells.items() if kv[1]["guardrail"]["pass"]), key=sort_key
    )
    failers = sorted(
        (kv for kv in cells.items() if not kv[1]["guardrail"]["pass"]), key=sort_key
    )
    ranking = [
        {
            "rank": i + 1,
            "cell": k,
            "guardrail_pass": v["guardrail"]["pass"],
            "sort_key": sort_key((k, v)),
        }
        for i, (k, v) in enumerate(passers)
    ] + [
        {"rank": None, "cell": k, "guardrail_pass": False, "sort_key": sort_key((k, v))}
        for k, v in failers
    ]

    results = {
        "config": {
            "cache": str(args.cache),
            "train_cache": str(args.train_cache),
            "ckpt": str(args.ckpt),
            "max_samples": args.max_samples,
            "n_eval_samples": n,
            "seeds": seeds,
            "sample_slice": args.sample_slice,
            "strategy_filter": args.strategies,
            "device": device,
            "mutex": {
                "semantics": "rsim ControlAdapter default: per frame, if brake > 0.05 -> gas = 0; then clamp gas/brake to [0,1], steer to [-1,1]",
                "threshold": MUTEX_THRESHOLD,
            },
            "thresholds": {"gas_sensor": GAS_THRESH, "brake_sensor": BRAKE_THRESH},
            "guardrail": "l1_overall <= argmax*1.02 AND steer_l1 <= argmax*1.02 (same mutex variant)",
            "lexicographic_key": [
                "conflict@0.05",
                "|topH_gas_deficit|",
                "conflict@0.02",
                "conflict@sensor",
            ],
        },
        "diagnostics": {
            "n_live_combos": int(glb.live_ids.shape[0]),
            "dead_argmax_rate": dead_argmax,
            "gt_topk_recall": tk_recall,
            "q0_entropy_quartile_edges": glb.q0_quart_edges,
            "joint_H_percentiles": glb.joint_H_pcts,
            "q0_entropy_percentiles": glb.q0H_pcts,
        },
        "ground_truth": gt,
        "cells": cells,
        "ranking": ranking,
        "elapsed_s": time.perf_counter() - t_start,
    }
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_md, results)


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    cells = results["cells"]
    gt = results["ground_truth"]
    lines = [
        "# Decode-strategy sweep",
        "",
        (
            f"- eval: `{results['config']['cache']}` (n={results['config']['n_eval_samples']}), "
            f"ckpt `{results['config']['ckpt']}`"
        ),
        (
            f"- live combos: {results['diagnostics']['n_live_combos']}; "
            f"dead-argmax rate: {results['diagnostics']['dead_argmax_rate']:.4f}"
        ),
        (
            f"- GT: gas-act {gt['gas_activation_overall']:.3f} "
            f"(topH quartile {gt['gas_activation_per_quartile'][3]:.3f}); "
            f"GT conflict@sensor {gt['conflict_rate']['sensor']:.4f}"
        ),
        f"- guardrail: {results['config']['guardrail']}",
        f"- rank key (lexicographic): {', '.join(results['config']['lexicographic_key'])}",
        (
            "- stochastic cells: mean over "
            f"{results['config']['seeds']} seeds (±std shown for conflict@0.05 / L1)"
        ),
        "",
        (
            "| rank | strategy | mutex | proposer | conf@0.05 | conf@0.02 | conf@sensor | "
            "gas_def_topH | gas_def_all | L1 | L1_steer | dead% | match% | guard |"
        ),
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]

    def leafd(m: dict[str, Any], *p: str | int) -> tuple[float, float | None]:
        node: Any = m
        for x in p:
            node = node[x]
        if isinstance(node, dict) and "mean" in node:
            return float(node["mean"]), float(node["std"])
        return float(node), None

    def fmt(v: float, s: float | None, prec: int = 4) -> str:
        return f"{v:.{prec}f}" + (f"±{s:.{prec}f}" if s is not None else "")

    for row in results["ranking"]:
        cell = cells[row["cell"]]
        m = cell["metrics"]
        c5 = fmt(*leafd(m, "conflict_rate", "0.05"))
        c2 = fmt(*leafd(m, "conflict_rate", "0.02")[:1], None)
        cs = fmt(*leafd(m, "conflict_rate", "sensor")[:1], None)
        dt = fmt(*leafd(m, "gas_activation", "topH_deficit")[:1], None)
        da = fmt(*leafd(m, "gas_activation", "overall_deficit")[:1], None)
        l1 = fmt(*leafd(m, "l1", "overall"))
        ls = fmt(*leafd(m, "l1", "steer")[:1], None)
        dead = f"{leafd(m, 'dead_tuple_rate')[0] * 100:.2f}"
        match = f"{leafd(m, 'tuple_match_rate')[0] * 100:.2f}"
        rank = str(row["rank"]) if row["rank"] is not None else "-"
        guard = "PASS" if cell["guardrail"]["pass"] else "FAIL"
        lines.append(
            f"| {rank} | {cell['strategy']} | {'on' if cell['mutex'] else 'off'} | "
            f"{cell['proposer']} | {c5} | {c2} | {cs} | {dt} | {da} | {l1} | {ls} | "
            f"{dead} | {match} | {guard} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="eval cache dir (offset_head_retrain extract shards)",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="lightning checkpoint (offset head + tokenizer weights)",
    )
    parser.add_argument(
        "--train-cache",
        default=DEFAULT_TRAIN_CACHE,
        help="train cache dir for usage/live set",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="cap eval samples (smoke: 2000)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="cap train samples for the usage histogram",
    )
    parser.add_argument(
        "--seeds",
        default="3",
        help="stochastic-strategy seeds: an int N (= seeds 0..N-1, original behavior) "
        "or an explicit comma list, e.g. '7,8,9' (stability rerun)",
    )
    parser.add_argument(
        "--sample-slice",
        default=None,
        help="contiguous 1-based slice 'i/k' of the loaded eval samples, applied AFTER "
        "--max-samples (e.g. '1/2' = first half, '2/2' = second half); for split-stability checks",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="comma-separated strategy-name filter (must include 'argmax' — it is the "
        "guardrail/ranking baseline); default: all strategies",
    )
    parser.add_argument(
        "--device", default="cpu", help="device for the offset-head forward (cpu|cuda)"
    )
    parser.add_argument(
        "--head-batch", type=int, default=1024, help="offset-head forward batch size"
    )
    parser.add_argument(
        "--eval-batch", type=int, default=2048, help="strategy-eval mega-batch size"
    )
    parser.add_argument("--out-json", default=str(SCRATCH / "sweep_results.json"))
    parser.add_argument("--out-md", default=str(SCRATCH / "sweep_results.md"))
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
