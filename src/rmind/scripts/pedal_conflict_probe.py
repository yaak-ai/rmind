"""Model-side pedal-conflict probe (brief 1c) for the offset-supervision bug.

Runs the finetuned `ControlTransformer` over the val split with export-time
decoding (`sample_codes=False`, argmax) and measures how often the predicted
action chunks command gas and brake simultaneously, using the exact sensor-side
thresholds from `rmind.scripts.pedal_conflict_stats`. Ground-truth conflict
rates are computed on the same frames in the same pass (the within-dataset
floor). Rates are bucketed by the entropy of the primary-quantizer (q=0) code
logits: the offset-cancellation mechanism predicts conflicts concentrated in
high-entropy states.

Before comparing, the probe verifies that the tokenizer's input transform for
the pedal fields is `Identity`, i.e. predictions live in the same [0, 1]
normalized pedal space as the raw sensor fields the thresholds were derived in.

Usage:
    uv run python -m rmind.scripts.pedal_conflict_probe --split val --max-batches 3 --no-wandb
    uv run python -m rmind.scripts.pedal_conflict_probe --split val --wandb
"""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING, Any, cast

import torch

from rmind.scripts.offset_diag import (
    DEFAULT_CKPT,
    REPO_ROOT,
    build_dataloader,
    iter_policy_tensors,
    load_policy,
)
from rmind.scripts.pedal_conflict_stats import BRAKE_THRESH, GAS_THRESH

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from rmind.components.containers import ModuleDict
    from rmind.components.objectives.joint_policy import JointPolicyObjective
    from rmind.models.action_tokenizer import ActionTokenizer
    from rmind.models.control_transformer import ControlTransformer

GAS_DIM = 0  # action-chunk field order: gas, brake, steering, turn_signal
BRAKE_DIM = 1
N_BUCKETS = 4

DEFAULT_OUT = REPO_ROOT / "diag_results" / "pedal_conflict_model.json"


def check_pedal_normalization(
    tokenizer: ActionTokenizer, device: torch.device
) -> dict[str, Any]:
    """Confirm predictions share the sensor fields' normalized pedal space.

    The thresholds from `pedal_conflict_stats` are defined on the raw
    `*_pedal_normalized` sensor fields. The model reconstructs
    `tokenizer._normalize(chunk)`, so the two are comparable iff the
    tokenizer's per-field input transform is `Identity` on the pedal dims.
    Checked structurally (transform module types) and numerically
    (`_normalize` round-trip on a random chunk).

    Raises:
        TypeError: if a pedal field's transform is not `torch.nn.Identity`.
        ValueError: if `_normalize` changes the pedal dims numerically.
    """
    norm = cast("ModuleDict", tokenizer.input_transform[-1])
    transforms = {
        field: norm.get_deepest(("continuous", field))
        for field in ("gas_pedal", "brake_pedal")
    }
    for field, module in transforms.items():
        if not isinstance(module, torch.nn.Identity):
            msg = (
                f"tokenizer input_transform for {field} is "
                f"{type(module).__name__}, expected Identity: predictions are "
                f"not in raw sensor pedal space, thresholds do not transfer"
            )
            raise TypeError(msg)

    probe = torch.rand(8, 6, 4, device=device)
    normalized = tokenizer._normalize(probe.flatten(-2, -1)).unflatten(  # noqa: SLF001
        -1, (-1, 4)
    )
    max_abs_err = float(
        (normalized[..., [GAS_DIM, BRAKE_DIM]] - probe[..., [GAS_DIM, BRAKE_DIM]])
        .abs()
        .max()
    )
    if max_abs_err > 0.0:
        msg = f"_normalize changed pedal dims (max abs err {max_abs_err})"
        raise ValueError(msg)

    return {
        "gas_transform": type(transforms["gas_pedal"]).__name__,
        "brake_transform": type(transforms["brake_pedal"]).__name__,
        "normalize_pedal_max_abs_err": max_abs_err,
        "pedal_space": "identity: normalized == raw sensor [0, 1]",
    }


def collect(
    model: ControlTransformer,
    dataloader: Any,
    device: str | torch.device,
    max_batches: int | None,
) -> dict[str, Tensor]:
    """Run the model over the split; return per-sample conflict masks and entropy.

    Returns CPU tensors: `pred_conflict` (n, horizon) bool, `gt_conflict`
    (n, horizon) bool, `entropy_q0` (n,) float (nats, over softmax of the
    q=0 code logits).

    Raises:
        RuntimeError: if the policy objective is not in argmax decoding mode.
    """
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    tokenizer = cast("ActionTokenizer", policy.tokenizer)
    if policy.sample_codes:
        msg = "policy.sample_codes must be False (export-time argmax decoding)"
        raise RuntimeError(msg)

    pred_conflicts: list[Tensor] = []
    gt_conflicts: list[Tensor] = []
    entropies: list[Tensor] = []

    t0 = time.perf_counter()
    for batch_idx, tensors in enumerate(
        iter_policy_tensors(model, dataloader, device=device, max_batches=max_batches)
    ):
        code_logits = tensors["code_logits"]  # (b, g, c)
        codes = policy._sample_codes(code_logits)  # noqa: SLF001  # argmax
        offset = policy._gather_offset(tensors["offsets"], codes)  # noqa: SLF001
        pred = (tokenizer.invert(codes) + offset).unflatten(-1, (-1, 4))  # (b, 6, 4)
        gt = tensors["target"].unflatten(-1, (-1, 4))  # (b, 6, 4), normalized GT

        pred_conflicts.append(
            (
                (pred[..., GAS_DIM] > GAS_THRESH)
                & (pred[..., BRAKE_DIM] > BRAKE_THRESH)
            ).cpu()
        )
        gt_conflicts.append(
            (
                (gt[..., GAS_DIM] > GAS_THRESH) & (gt[..., BRAKE_DIM] > BRAKE_THRESH)
            ).cpu()
        )

        probs = code_logits[:, 0, :].softmax(dim=-1)
        entropies.append(-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).cpu())

        if (batch_idx + 1) % 50 == 0:
            rate = (batch_idx + 1) / (time.perf_counter() - t0)
            print(  # noqa: T201
                f"[pedal_conflict_probe] {batch_idx + 1} batches ({rate:.2f}/s)"
            )

    return {
        "pred_conflict": torch.cat(pred_conflicts),
        "gt_conflict": torch.cat(gt_conflicts),
        "entropy_q0": torch.cat(entropies),
    }


def _conflict_stats(conflict: Tensor) -> dict[str, Any]:
    """Frame-level and sample-level (any-frame) conflict rates for a (n, h) mask."""
    return {
        "n_samples": int(conflict.shape[0]),
        "n_frames": int(conflict.numel()),
        "conflict_frames": int(conflict.sum()),
        "frame_conflict_rate": float(conflict.float().mean()),
        "conflict_samples": int(conflict.any(dim=-1).sum()),
        "sample_any_frame_rate": float(conflict.any(dim=-1).float().mean()),
    }


def summarize(collected: dict[str, Tensor]) -> dict[str, Any]:
    """Aggregate collected masks into overall and entropy-quartile-bucketed rates."""
    entropy = collected["entropy_q0"]
    edges = torch.quantile(entropy, torch.tensor([0.25, 0.5, 0.75]))
    bucket_ids = torch.bucketize(entropy, edges)

    def bucketed(conflict: Tensor) -> list[dict[str, Any]]:
        buckets = []
        for b in range(N_BUCKETS):
            mask = bucket_ids == b
            buckets.append({
                "bucket": b,
                "entropy_mean_nats": float(entropy[mask].mean()),
                **_conflict_stats(conflict[mask]),
            })
        return buckets

    pred = _conflict_stats(collected["pred_conflict"])
    gt = _conflict_stats(collected["gt_conflict"])
    gt_floor = gt["frame_conflict_rate"]
    return {
        "entropy_q0_quartile_edges_nats": [float(e) for e in edges],
        "entropy_q0_mean_nats": float(entropy.mean()),
        "predicted": {**pred, "buckets": bucketed(collected["pred_conflict"])},
        "ground_truth": {**gt, "buckets": bucketed(collected["gt_conflict"])},
        "predicted_over_gt_frame_rate": (
            pred["frame_conflict_rate"] / gt_floor if gt_floor > 0 else None
        ),
    }


def _log_wandb(results: dict[str, Any], out_path: Path) -> str:
    import wandb  # noqa: PLC0415

    run = wandb.init(
        entity="yaak",
        project="rmind",
        name="diag-pedal-conflict",
        tags=["diag/joint_policy_offset_bug"],
        dir=str(REPO_ROOT / "wandb_logs"),
        config={
            k: results[k]
            for k in ("ckpt", "split", "decoding", "thresholds", "normalization_check")
        },
    )
    flat: dict[str, float] = {}
    for side in ("predicted", "ground_truth"):
        stats = results[side]
        flat[f"{side}/frame_conflict_rate"] = stats["frame_conflict_rate"]
        flat[f"{side}/sample_any_frame_rate"] = stats["sample_any_frame_rate"]
        for bucket in stats["buckets"]:
            b = bucket["bucket"]
            flat[f"{side}/frame_conflict_rate/entropy_q{b}"] = bucket[
                "frame_conflict_rate"
            ]
            flat[f"{side}/sample_any_frame_rate/entropy_q{b}"] = bucket[
                "sample_any_frame_rate"
            ]
    run.log(flat)
    run.summary["results"] = results
    run.save(str(out_path), base_path=str(out_path.parent), policy="now")
    url = run.url
    run.finish()
    return url


def main(args: argparse.Namespace) -> None:
    model = load_policy(args.ckpt, args.device)
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    tokenizer = cast("ActionTokenizer", policy.tokenizer)

    normalization_check = check_pedal_normalization(
        tokenizer, torch.device(args.device)
    )
    print(f"[pedal_conflict_probe] normalization check: {normalization_check}")  # noqa: T201

    loader = build_dataloader(
        args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(  # noqa: T201
        f"[pedal_conflict_probe] split={args.split}: {len(loader.dataset)} samples, "
        f"{len(loader)} batches of {args.batch_size}"
        + (f" (capped at {args.max_batches})" if args.max_batches else "")
    )

    collected = collect(model, loader, args.device, args.max_batches)
    results: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "decoding": "argmax (sample_codes=False, export parity)",
        "thresholds": {"gas": GAS_THRESH, "brake": BRAKE_THRESH},
        "normalization_check": normalization_check,
        **summarize(collected),
    }

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[pedal_conflict_probe] results written to {out_path}")  # noqa: T201

    if args.wandb:
        url = _log_wandb(results, out_path)
        print(f"[pedal_conflict_probe] wandb run: {url}")  # noqa: T201

    print(json.dumps(results, indent=2))  # noqa: T201


if __name__ == "__main__":
    import multiprocessing as mp
    from pathlib import Path as _Path

    mp.set_forkserver_preload(["rbyte", "polars"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=_Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "train_debug"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=_Path, default=DEFAULT_OUT)
    main(parser.parse_args())
