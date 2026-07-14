"""Calibrate chain-conditional decode LUTs for `chain_greedy` and bake them
into a checkpoint for ONNX/TensorRT export.

The `chain_greedy` decode needs empirical chain conditionals
P(c_g | c_0..c_{g-1}) from the training data's ground-truth codes. This script
computes them (or loads a precomputed `--luts` file), installs them into a
`JointPolicyObjective`'s `chain_log_prior_{g}` buffers, sets
`decode_strategy=chain_greedy` + `decode_beta`, and writes a self-contained
checkpoint the export config can load. Since the LUTs depend only on the frozen
tokenizer + training data (not the policy weights), the committed
`luts_<tokenizer>.pt` can be reused across finetunes of the same tokenizer with
`--luts`, so the export server needs no training data.

Usage (compute from a code cache, then bake):
    uv run python -m rmind.scripts.calibrate_decode_luts \
        --ckpt artifacts/model-<run>:v9/model.ckpt \
        --codes-cache caches/offset_head/train \
        --beta 1.0 --out model.chain_greedy.ckpt --save-luts luts.pt

Usage (bake a precomputed LUT, no training data needed):
    uv run python -m rmind.scripts.calibrate_decode_luts \
        --ckpt <ckpt> --luts luts.pt --beta 1.0 --out model.chain_greedy.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from structlog import get_logger

from rmind.scripts.offset_diag import load_policy

log = get_logger(__name__)


def _load_train_codes(cache: Path) -> torch.Tensor:
    shards = sorted(cache.glob("shard_*.pt"))
    if not shards:
        msg = f"no shard_*.pt in {cache}"
        raise FileNotFoundError(msg)
    return torch.cat([
        torch.load(s, weights_only=False)["target_codes"].long() for s in shards
    ])  # (n, g)


def _chain_luts(
    codes: torch.Tensor, g: int, c: int, alpha: float
) -> list[torch.Tensor]:
    """Laplace-smoothed log P(c_g | prefix) per level; prefix packs c_0..c_{g-1}."""
    luts: list[torch.Tensor] = []
    for level in range(g):
        counts = torch.full((c**level, c), alpha)
        if level == 0:
            prefix = torch.zeros(len(codes), dtype=torch.long)
        else:
            prefix = torch.zeros(len(codes), dtype=torch.long)
            for j in range(level):
                prefix = prefix * c + codes[:, j]
        counts.index_put_(
            (prefix, codes[:, level]), torch.ones(len(codes)), accumulate=True
        )
        luts.append((counts / counts.sum(-1, keepdim=True)).log())
    return luts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--codes-cache", type=Path, default=None)
    p.add_argument("--luts", type=Path, default=None, help="precomputed luts .pt")
    p.add_argument("--save-luts", type=Path, default=None)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = load_policy(str(args.ckpt), args.device)
    policy = model.objectives["policy"]
    g = policy.tokenizer.quantizer.num_quantizers
    c = policy.tokenizer.quantizer.codebook_size

    if args.luts is not None:
        luts = torch.load(args.luts, weights_only=False)["luts"]
        log.info("loaded precomputed luts", path=str(args.luts))
    elif args.codes_cache is not None:
        codes = _load_train_codes(args.codes_cache)
        luts = _chain_luts(codes, g, c, args.alpha)
        log.info("computed chain luts", n_codes=len(codes), g=g, c=c)
        if args.save_luts is not None:
            torch.save({"luts": luts, "g": g, "c": c}, args.save_luts)
    else:
        msg = "provide --codes-cache or --luts"
        raise ValueError(msg)

    for level in range(g):
        buf = f"chain_log_prior_{level}"
        getattr(policy, buf).copy_(luts[level].to(getattr(policy, buf)))
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # bake buffers into state_dict AND decode config into hyper_parameters, so a
    # plain load_from_checkpoint (no jq) reconstructs a self-contained chain_greedy
    # model (the export jq also sets these — belt and suspenders).
    ckpt["state_dict"].update({
        f"objectives.policy.chain_log_prior_{level}": luts[level].float()
        for level in range(g)
    })
    pol_hp = ckpt["hyper_parameters"]["objectives"]["modules"]["policy"]
    pol_hp["decode_strategy"] = "chain_greedy"
    pol_hp["decode_beta"] = args.beta
    torch.save(ckpt, args.out)
    # verify roundtrip: buffers persist AND decode_strategy self-describes
    rt = load_policy(str(args.out), args.device).objectives["policy"]
    ok = all(
        torch.equal(getattr(rt, f"chain_log_prior_{level}").cpu(), luts[level].float())
        for level in range(g)
    )
    strat_ok = rt.decode_strategy == "chain_greedy"
    log.info(
        "wrote calibrated checkpoint",
        out=str(args.out),
        buffers_roundtrip_ok=ok,
        decode_strategy=rt.decode_strategy,
    )
    if not (ok and strat_ok):
        msg = f"roundtrip mismatch (buffers={ok}, strategy={rt.decode_strategy})"
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
