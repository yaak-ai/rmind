"""Offset-cancellation probe for the JointPolicyObjective offset-supervision bug.

Phase-1 diagnostics (brief 1a + 1b) against the finetuned checkpoint, using
export-time decoding (`sample_codes=False`, enforced by `load_policy`):

1a (always): for each sample, enumerate residual-VQ code combinations and
compare the decoded actions with and without the code-conditioned offset.
With G=4 quantizers and C=16 codes the full product (65536) is too large, so
per the brief we enumerate all 16 primary (q=0) codes crossed with each
sample's top-4 most probable secondary (q=1) codes, holding q=2 and q=3 at
their per-sample argmax — 64 combinations per sample. For every combination
`c` we compute `decoded[c] = tokenizer.invert(c)` and
`corrected[c] = decoded[c] + gather_offset(c)`, then the per-action-dimension
cancellation ratio `R = Var_c(corrected) / Var_c(decoded)`. R << 1 means the
offset head collapses distinct codes onto the same output (the predicted bug
signature); R ~ 1 means the offsets preserve cross-code variance. Results are
aggregated per action field (gas, brake, steering, turn_signal: mean over the
6 horizon steps) and per raw dimension, overall and bucketed by quartile of
the primary-code (q=0) softmax entropy.

1b (--audit): per (sample, quantizer) mismatch of the SAMPLED code
(`torch.multinomial`, seeded — the pre-fix training-time path) vs the
ground-truth code, plus the L1 magnitude of the offset target
`|target - tokenizer.invert(sampled_codes)|` split by whether the full code
tuple matched GT. Mismatched targets spanning cross-mode distances (much
larger than matched sub-cell residuals) confirm training-signal contamination.

Usage:
    uv run python -m rmind.scripts.offset_cancellation_probe \
        --split val --max-batches 63 --audit --wandb
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl
import torch

from rmind.scripts.offset_diag import (
    DEFAULT_CKPT,
    REPO_ROOT,
    build_dataloader,
    iter_policy_tensors,
    load_policy,
)

if TYPE_CHECKING:
    from torch import Tensor

    from rmind.components.objectives.joint_policy import JointPolicyObjective
    from rmind.models.action_tokenizer import ActionTokenizer

FIELD_NAMES = ("gas_pedal", "brake_pedal", "steering_angle", "turn_signal")
N_SECONDARY = 4  # top-k of q=1 crossed with all 16 primary codes -> 64 combos
VAR_EPS = 1e-10  # dims with Var_c(decoded) below this are excluded (R undefined)
DEFAULT_OUT = REPO_ROOT / "diag_results" / "offset_cancellation.json"
WANDB_RUN_NAME = "diag-offset-cancellation"
WANDB_TAGS = ("diag/joint_policy_offset_bug",)


def _enumerate_combos(code_logits: Tensor, n_secondary: int = N_SECONDARY) -> Tensor:
    """Code combinations (b, 16 * n_secondary, G) per the brief's scheme.

    All C=16 primary (q=0) codes x each sample's top-`n_secondary` q=1 codes;
    q=2 and q=3 held at their per-sample argmax (full product is too large).
    """
    b, g, c = code_logits.shape
    k = c * n_secondary

    combos = torch.empty(b, k, g, dtype=torch.long, device=code_logits.device)
    primary = torch.arange(c, device=code_logits.device).repeat_interleave(n_secondary)
    combos[..., 0] = primary  # (k,) broadcast over batch
    secondary = code_logits[:, 1].topk(n_secondary, dim=-1).indices  # (b, n_secondary)
    combos[..., 1] = secondary.repeat(1, c)  # tiles to match primary's interleave
    combos[..., 2:] = code_logits[:, 2:].argmax(dim=-1)[:, None, :]
    return combos


def _gather_combo_offsets(offsets: Tensor, combos: Tensor) -> Tensor:
    """Sum each quantizer's offset at `combos`: (b,G,C,A),(b,K,G) -> (b,K,A)."""
    b, k, g = combos.shape
    a = offsets.shape[-1]
    table = offsets.unsqueeze(1).expand(b, k, g, offsets.shape[2], a)
    index = combos[..., None, None].expand(b, k, g, 1, a)
    return table.gather(3, index).squeeze(3).sum(dim=2)


def _cancellation_batch(
    tokenizer: ActionTokenizer, code_logits: Tensor, offsets: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-sample cancellation ratios (b, action_dim) and q=0 entropy (b,)."""
    combos = _enumerate_combos(code_logits)
    decoded = tokenizer.invert(combos)  # (b, K, A), normalized action space
    corrected = decoded + _gather_combo_offsets(offsets, combos)

    var_decoded = decoded.var(dim=1)  # (b, A), across the K combos
    var_corrected = corrected.var(dim=1)
    ratio = torch.where(var_decoded > VAR_EPS, var_corrected / var_decoded, torch.nan)

    probs = code_logits[:, 0].softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)  # nats
    return ratio, entropy


def _sample_codes_seeded(code_logits: Tensor, generator: torch.Generator) -> Tensor:
    """The pre-fix training-time sampling path (`torch.multinomial`), seeded.

    Sampling runs on CPU with a CPU generator so the audit is deterministic
    for a given seed regardless of device; the batch is tiny (b*G rows of C
    probabilities).
    """
    b, g, c = code_logits.shape
    probs = code_logits.softmax(dim=-1).reshape(-1, c).cpu()
    sampled = torch.multinomial(probs, 1, generator=generator).reshape(b, g)
    return sampled.to(code_logits.device)


def _audit_batch(
    tokenizer: ActionTokenizer,
    code_logits: Tensor,
    target_codes: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Sampled-vs-GT code mismatches and offset-target L1 for one batch (on CPU)."""
    sampled = _sample_codes_seeded(code_logits, generator)
    mismatch = sampled != target_codes  # (b, G)
    full_match = ~mismatch.any(dim=-1)  # (b,)
    # what the offset head was asked to produce at the sampled codes
    offset_target_l1 = (target - tokenizer.invert(sampled)).abs()  # (b, A)
    return {
        "mismatch": mismatch.cpu(),
        "full_match": full_match.cpu(),
        "offset_target_l1": offset_target_l1.cpu(),
    }


def _sanitize(value: float) -> float | None:
    return None if math.isnan(value) else value


def _ratio_stats(ratios: Tensor) -> dict[str, Any]:
    """Mean/median R per field (mean over the 6 horizon steps) and per dim."""
    n = ratios.shape[0]
    per_field = ratios.reshape(n, -1, len(FIELD_NAMES)).nanmean(dim=1)  # (n, 4)
    return {
        "n_samples": n,
        "per_field": {
            name: {
                "mean_R": _sanitize(float(per_field[:, i].nanmean())),
                "median_R": _sanitize(float(per_field[:, i].nanmedian())),
            }
            for i, name in enumerate(FIELD_NAMES)
        },
        "per_dim": {
            "mean_R": [_sanitize(float(v)) for v in ratios.nanmean(dim=0)],
            "median_R": [_sanitize(float(v)) for v in ratios.nanmedian(dim=0).values],
        },
    }


def _aggregate_cancellation(ratios: Tensor, entropies: Tensor) -> dict[str, Any]:
    edges = torch.quantile(entropies, torch.tensor([0.25, 0.5, 0.75]))
    buckets = torch.bucketize(entropies, edges)

    by_quartile = []
    for i in range(4):
        mask = buckets == i
        selected = entropies[mask]
        by_quartile.append({
            "quartile": i,
            "entropy_min": float(selected.min()) if mask.any() else None,
            "entropy_max": float(selected.max()) if mask.any() else None,
            **_ratio_stats(ratios[mask]),
        })

    return {
        "combo_scheme": (
            "all 16 primary (q=0) codes x per-sample top-4 q=1 codes; "
            "q=2, q=3 held at per-sample argmax (64 combos; full 16^4 product "
            "too large)"
        ),
        "n_combos": 16 * N_SECONDARY,
        "invalid_dim_fraction": float(ratios.isnan().float().mean()),
        "entropy_quartile_edges_nats": [float(v) for v in edges],
        "overall": _ratio_stats(ratios),
        "by_entropy_quartile": by_quartile,
    }


def _aggregate_audit(
    mismatch: Tensor, full_match: Tensor, offset_target_l1: Tensor
) -> dict[str, Any]:
    n, g = mismatch.shape
    matched_l1 = offset_target_l1[full_match]
    mismatched_l1 = offset_target_l1[~full_match]

    def field_means(l1: Tensor) -> dict[str, float | None]:
        if l1.shape[0] == 0:
            return dict.fromkeys(FIELD_NAMES)
        per_field = l1.reshape(l1.shape[0], -1, len(FIELD_NAMES)).mean(dim=(0, 1))
        return {name: float(per_field[i]) for i, name in enumerate(FIELD_NAMES)}

    return {
        "n_samples": n,
        "code_mismatch_rate_overall": float(mismatch.float().mean()),
        "code_mismatch_rate_per_quantizer": [
            float(mismatch[:, q].float().mean()) for q in range(g)
        ],
        "full_tuple_match_rate": float(full_match.float().mean()),
        "n_full_tuple_matched": int(full_match.sum()),
        "n_full_tuple_mismatched": int((~full_match).sum()),
        "offset_target_l1_matched": (
            float(matched_l1.mean()) if matched_l1.shape[0] else None
        ),
        "offset_target_l1_mismatched": (
            float(mismatched_l1.mean()) if mismatched_l1.shape[0] else None
        ),
        "offset_target_l1_matched_per_field": field_means(matched_l1),
        "offset_target_l1_mismatched_per_field": field_means(mismatched_l1),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    pl.seed_everything(args.seed, workers=True)
    model = load_policy(args.ckpt, args.device)
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    tokenizer = cast("ActionTokenizer", policy.tokenizer)
    loader = build_dataloader(
        args.split, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(  # noqa: T201
        f"[probe] split={args.split}: {len(loader.dataset)} samples, "
        f"{len(loader)} batches of {args.batch_size}; "
        f"max_batches={args.max_batches}, audit={args.audit}"
    )

    generator = torch.Generator()  # CPU: see _sample_codes_seeded
    generator.manual_seed(args.seed)

    ratios: list[Tensor] = []
    entropies: list[Tensor] = []
    audit_parts: dict[str, list[Tensor]] = {}

    for batch_idx, tensors in enumerate(
        iter_policy_tensors(model, loader, args.device, max_batches=args.max_batches)
    ):
        ratio, entropy = _cancellation_batch(
            tokenizer, tensors["code_logits"], tensors["offsets"]
        )
        ratios.append(ratio.cpu())
        entropies.append(entropy.cpu())

        if args.audit:
            part = _audit_batch(
                tokenizer,
                tensors["code_logits"],
                tensors["target_codes"],
                tensors["target"],
                generator,
            )
            for key, value in part.items():
                audit_parts.setdefault(key, []).append(value)

        if (batch_idx + 1) % 10 == 0:
            print(f"[probe] processed {batch_idx + 1} batches")  # noqa: T201

    results: dict[str, Any] = {
        "config": {
            "ckpt": str(args.ckpt),
            "split": args.split,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "seed": args.seed,
            "sample_codes": False,  # export-time argmax decoding
        },
        "n_samples": int(sum(r.shape[0] for r in ratios)),
        "cancellation": _aggregate_cancellation(
            torch.cat(ratios), torch.cat(entropies)
        ),
    }
    if args.audit:
        results["audit"] = _aggregate_audit(
            torch.cat(audit_parts["mismatch"]),
            torch.cat(audit_parts["full_match"]),
            torch.cat(audit_parts["offset_target_l1"]),
        )
    return results


def _log_wandb(results: dict[str, Any]) -> str | None:
    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity="yaak",
        project="rmind",
        name=WANDB_RUN_NAME,
        tags=list(WANDB_TAGS),
        dir=str(REPO_ROOT / "wandb_logs"),
        config=results["config"] | {"n_samples": results["n_samples"]},
    )

    cancellation = results["cancellation"]
    scalars: dict[str, Any] = {}
    for name, stats in cancellation["overall"]["per_field"].items():
        scalars[f"cancellation/R_mean/{name}"] = stats["mean_R"]
        scalars[f"cancellation/R_median/{name}"] = stats["median_R"]

    if (audit := results.get("audit")) is not None:
        scalars["audit/code_mismatch_rate_overall"] = audit[
            "code_mismatch_rate_overall"
        ]
        for q, rate in enumerate(audit["code_mismatch_rate_per_quantizer"]):
            scalars[f"audit/code_mismatch_rate/q{q}"] = rate
        scalars["audit/full_tuple_match_rate"] = audit["full_tuple_match_rate"]
        scalars["audit/offset_target_l1_matched"] = audit["offset_target_l1_matched"]
        scalars["audit/offset_target_l1_mismatched"] = audit[
            "offset_target_l1_mismatched"
        ]

    table = wandb.Table(
        columns=[
            "entropy_quartile",
            "entropy_min",
            "entropy_max",
            "n_samples",
            "field",
            "mean_R",
            "median_R",
        ]
    )
    for bucket in cancellation["by_entropy_quartile"]:
        for name, stats in bucket["per_field"].items():
            table.add_data(
                bucket["quartile"],
                bucket["entropy_min"],
                bucket["entropy_max"],
                bucket["n_samples"],
                name,
                stats["mean_R"],
                stats["median_R"],
            )

    run.log(scalars | {"cancellation/R_by_entropy_quartile": table})
    url = run.url
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "train_debug"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit",
        action="store_true",
        help="also run the 1b sampled-vs-GT code contamination audit",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log scalars and the R-by-entropy-quartile table to wandb",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    results = run_probe(args)

    if args.wandb:
        results["wandb_url"] = _log_wandb(results)
        print(f"[probe] wandb run: {results['wandb_url']}")  # noqa: T201

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] results written to {out}")  # noqa: T201
    print(json.dumps(results["cancellation"]["overall"]["per_field"], indent=2))  # noqa: T201
    if (audit := results.get("audit")) is not None:
        print(  # noqa: T201
            json.dumps(
                {
                    k: audit[k]
                    for k in (
                        "code_mismatch_rate_overall",
                        "code_mismatch_rate_per_quantizer",
                        "full_tuple_match_rate",
                        "offset_target_l1_matched",
                        "offset_target_l1_mismatched",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    import sys

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()

    # torchdata.nodes worker-thread teardown intermittently aborts at
    # interpreter exit ("terminate called without an active exception",
    # SIGABRT) even after `shutdown_dataloader`; every output (wandb, JSON,
    # stdout) is already flushed, so skip interpreter teardown entirely
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
