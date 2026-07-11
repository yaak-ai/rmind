"""Offset-head-only retraining experiment for the offset-supervision bug.

Isolates the training-signal contamination (offset head supervised at SAMPLED
codes, brief section 1) on the bugged finetuned checkpoint
(`artifacts/model-2gqxhjod:v9/model.ckpt`): everything except the offset head
stays frozen at checkpoint weights, and only the offset head is retrained from
cached features under the two supervision variants. Because features and code
logits are fixed, any difference between the variants is attributable to the
offset supervision alone.

Stages (subcommands):

- `extract`: run the frozen model over a split once (via `offset_diag`) and
  cache per-sample tensors to disk as `torch.save` shards + `meta.json`:
  `features` (b, 1152) fp16, `code_logits` (b, 4, 16) fp16, `target_codes`
  (b, 4) int8, `target` (b, 24) fp32. Refuses to overwrite a non-empty cache.

- `train`: deep-copy the checkpoint's `offset_head` (initialization = the
  contaminated weights) and retrain it on the cached tensors with
  `L1(tokenizer.invert(codes) + gather_offset(offsets, codes), target)`,
  where `codes` are either re-sampled per step from the cached (frozen) code
  logits' softmax with a seeded generator (`--variant sampled`, mirroring the
  pre-fix training signal) or the ground-truth codes
  (`--variant teacher_forced`, mirroring the fix). Code logits are inputs,
  never trained; the only trainable parameters are the offset head's.

- `eval`: for the checkpoint's own head (`original`) and any retrained heads,
  on a cached val split: (a) cancellation ratio R per action field with the
  exact combo scheme of `offset_cancellation_probe`, (b) predicted
  pedal-conflict rates overall and per entropy quartile with the thresholds
  of `pedal_conflict_stats` (as `pedal_conflict_probe`), (c) teacher-forced-
  and sampled-style offset L1 (metrics only), (d) mean |gathered offset| at
  argmax codes and over the full table. One JSON, all heads side by side.

Usage:
    uv run python -m rmind.scripts.offset_head_retrain extract \\
        --split val --out caches/offset_head/val
    uv run python -m rmind.scripts.offset_head_retrain train \\
        --cache caches/offset_head/train --variant teacher_forced \\
        --out heads/teacher_forced.pt
    uv run python -m rmind.scripts.offset_head_retrain eval \\
        --cache caches/offset_head/val \\
        --heads original,heads/sampled.pt,heads/teacher_forced.pt
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.scripts.offset_cancellation_probe import (
    FIELD_NAMES,
    _aggregate_cancellation,
    _cancellation_batch,
    _sample_codes_seeded,
)
from rmind.scripts.offset_diag import (
    DEFAULT_CKPT,
    EXPECTED_CODEBOOK_SIZE,
    EXPECTED_NUM_QUANTIZERS,
    REPO_ROOT,
    build_dataloader,
    iter_policy_tensors,
    load_policy,
)
from rmind.scripts.pedal_conflict_probe import BRAKE_DIM, GAS_DIM
from rmind.scripts.pedal_conflict_probe import summarize as summarize_pedal_conflicts
from rmind.scripts.pedal_conflict_stats import BRAKE_THRESH, GAS_THRESH

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from rmind.models.action_tokenizer import ActionTokenizer

FEATURE_DIM = 1152
ACTION_DIM = 24
N_FIELDS = len(FIELD_NAMES)

CACHE_DTYPES: dict[str, torch.dtype] = {
    "features": torch.float16,
    "code_logits": torch.float16,
    "target_codes": torch.int8,
    "target": torch.float32,
}
CACHE_SHAPES: dict[str, tuple[int, ...]] = {
    "features": (FEATURE_DIM,),
    "code_logits": (EXPECTED_NUM_QUANTIZERS, EXPECTED_CODEBOOK_SIZE),
    "target_codes": (EXPECTED_NUM_QUANTIZERS,),
    "target": (ACTION_DIM,),
}
SHARD_SIZE = 8192
DEFAULT_TRAIN_MAX_SAMPLES = 150_000
LOG_EVERY = 200
DEFAULT_EVAL_OUT = REPO_ROOT / "diag_results" / "offset_head_retrain.json"

_gather_offset = JointPolicyObjective._gather_offset  # noqa: SLF001


def _policy_parts(ckpt: str | Path, device: str) -> tuple[ActionTokenizer, Module]:
    """Checkpoint's frozen tokenizer and offset head; the rest of the model is freed."""
    model = load_policy(ckpt, device)
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    tokenizer = cast("ActionTokenizer", policy.tokenizer)
    offset_head = policy.offset_head
    del model, policy  # drop encoder/episode_builder/code_head references
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, offset_head


def _rearrange_offsets(flat: Tensor) -> Tensor:
    """(b, G*C*A) offset-head output -> (b, G, C, A) table."""
    return rearrange(
        flat,
        "b (g c a) -> b g c a",
        g=EXPECTED_NUM_QUANTIZERS,
        c=EXPECTED_CODEBOOK_SIZE,
    )


# --- extract ------------------------------------------------------------------


def _flush_shard(
    out_dir: Path, shard_idx: int, buffers: dict[str, list[Tensor]]
) -> None:
    shard = {key: torch.cat(parts) for key, parts in buffers.items()}
    torch.save(shard, out_dir / f"shard_{shard_idx:05d}.pt")
    for parts in buffers.values():
        parts.clear()


def run_extract(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        msg = (
            f"{out_dir} already exists and is not empty; refusing to overwrite "
            f"(delete it or pick a new --out)"
        )
        raise SystemExit(msg)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples: int | None = args.max_samples
    if max_samples is None and args.split == "train":
        max_samples = DEFAULT_TRAIN_MAX_SAMPLES

    pl.seed_everything(args.seed, workers=True)
    loader = build_dataloader(
        args.split, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(  # noqa: T201
        f"[extract] split={args.split}: {len(loader.dataset)} samples, "
        f"{len(loader)} batches of {args.batch_size}; max_samples={max_samples}"
    )
    model = load_policy(args.ckpt, args.device)

    buffers: dict[str, list[Tensor]] = {key: [] for key in CACHE_DTYPES}
    n_total = n_buffered = n_shards = 0
    t0 = time.perf_counter()
    for batch_idx, tensors in enumerate(
        iter_policy_tensors(model, loader, device=args.device)
    ):
        batch = {
            key: tensors[key].to(dtype).cpu() for key, dtype in CACHE_DTYPES.items()
        }
        if (
            max_samples is not None
            and n_total + batch["features"].shape[0] > max_samples
        ):
            keep = max_samples - n_total
            batch = {key: value[:keep] for key, value in batch.items()}
        for key, value in batch.items():
            buffers[key].append(value)
        n_total += batch["features"].shape[0]
        n_buffered += batch["features"].shape[0]

        if n_buffered >= SHARD_SIZE:
            _flush_shard(out_dir, n_shards, buffers)
            n_shards += 1
            n_buffered = 0

        if (batch_idx + 1) % 50 == 0:
            rate = n_total / (time.perf_counter() - t0)
            print(  # noqa: T201
                f"[extract] {n_total} samples ({batch_idx + 1} batches, "
                f"{rate:.1f} samples/s)"
            )

        if max_samples is not None and n_total >= max_samples:
            break

    if n_buffered > 0:
        _flush_shard(out_dir, n_shards, buffers)
        n_shards += 1

    elapsed = time.perf_counter() - t0
    meta = {
        "split": args.split,
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_samples": max_samples,
        "n_samples": n_total,
        "n_shards": n_shards,
        "fields": {key: str(dtype) for key, dtype in CACHE_DTYPES.items()},
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(  # noqa: T201
        f"[extract] done: {n_total} samples in {n_shards} shards -> {out_dir} "
        f"({elapsed:.1f}s, {n_total / elapsed:.1f} samples/s)"
    )


# --- cache loading ------------------------------------------------------------


def _load_cache(cache_dir: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    shard_paths = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_paths:
        msg = f"no shards found in {cache_dir}"
        raise SystemExit(msg)
    shards = [
        torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths
    ]
    data = {key: torch.cat([shard[key] for shard in shards]) for key in CACHE_DTYPES}

    n = data["features"].shape[0]
    if n != meta["n_samples"]:
        msg = (
            f"cache {cache_dir}: meta says {meta['n_samples']} samples, shards have {n}"
        )
        raise SystemExit(msg)
    for key, trailing in CACHE_SHAPES.items():
        if tuple(data[key].shape) != (n, *trailing):
            msg = f"cache {cache_dir}: {key} has shape {tuple(data[key].shape)}, expected {(n, *trailing)}"
            raise SystemExit(msg)
    return data, meta


# --- train --------------------------------------------------------------------


def _check_only_head_trainable(
    head: Module, tokenizer: Module, data: dict[str, Tensor]
) -> int:
    """The only trainable params are the offset head's; cached inputs carry no grad.

    Raises:
        RuntimeError: if the offset head is not fully trainable, the tokenizer
            is not frozen, or any cached tensor (incl. `code_logits`, which are
            frozen inputs by construction) requires grad.
    """
    head_params = list(head.parameters())
    if not head_params or not all(p.requires_grad for p in head_params):
        msg = "offset head parameters must all be trainable"
        raise RuntimeError(msg)
    if any(p.requires_grad for p in tokenizer.parameters()):
        msg = "tokenizer parameters must stay frozen"
        raise RuntimeError(msg)
    if any(value.requires_grad for value in data.values()):
        msg = "cached tensors (incl. code_logits) must not require grad"
        raise RuntimeError(msg)
    return sum(p.numel() for p in head_params)


def run_train(args: argparse.Namespace) -> None:  # noqa: PLR0914
    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)
    data, meta = _load_cache(Path(args.cache))
    n = data["features"].shape[0]

    tokenizer, head_src = _policy_parts(args.ckpt, args.device)
    # initialization = the checkpoint's contaminated offset-head weights
    head = copy.deepcopy(head_src).to(device).float()
    head.requires_grad_(requires_grad=True)
    head.train()
    n_trainable = _check_only_head_trainable(head, tokenizer, data)
    print(  # noqa: T201
        f"[train] variant={args.variant}: {n} cached samples "
        f"(split={meta['split']}), {n_trainable} trainable offset-head params; "
        f"code logits are frozen cache inputs"
    )

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=args.steps)
        if args.lr_schedule == "cosine"
        else None
    )
    # CPU generator drives both batch indices and (sampled variant) multinomial
    generator = torch.Generator().manual_seed(args.seed)

    loss_history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        index = torch.randint(0, n, (args.batch_size,), generator=generator)
        features = data["features"][index].to(device).float()
        target = data["target"][index].to(device)
        if args.variant == "sampled":
            # re-sample codes from the frozen cached logits each step: the
            # pre-fix training signal (code independent of the GT action)
            codes = _sample_codes_seeded(
                data["code_logits"][index].float(), generator
            ).to(device)
        else:
            codes = data["target_codes"][index].long().to(device)

        with torch.no_grad():
            decoded = tokenizer.invert(codes)

        offsets = _rearrange_offsets(head(features))
        loss = F.l1_loss(decoded + _gather_offset(offsets, codes), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % LOG_EVERY == 0 or step == args.steps:
            lr = optimizer.param_groups[0]["lr"]
            rate = step / (time.perf_counter() - t0)
            loss_value = float(loss.detach())
            loss_history.append({"step": step, "loss": loss_value, "lr": lr})
            print(  # noqa: T201
                f"[train] step {step}/{args.steps} loss={loss_value:.5f} "
                f"lr={lr:.2e} ({rate:.2f} steps/s)"
            )

    config = {
        "variant": args.variant,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "seed": args.seed,
        "cache": str(args.cache),
        "cache_meta": meta,
        "ckpt": str(args.ckpt),
        "n_trainable_params": n_trainable,
        "first_loss": loss_history[0]["loss"],
        "final_loss": loss_history[-1]["loss"],
        "loss_history": loss_history,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    torch.save({"state_dict": state_dict, "config": config}, out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[train] saved retrained offset head -> {out} (config: {sidecar})")  # noqa: T201


# --- eval ---------------------------------------------------------------------


def _build_heads(
    specs: str, template: Module
) -> dict[str, tuple[Module, dict[str, Any] | None]]:
    """Parse --heads: 'original' and/or paths to retrained-head .pt files.

    Raises:
        SystemExit: on duplicate head names or an empty --heads value.
    """
    heads: dict[str, tuple[Module, dict[str, Any] | None]] = {}
    for raw in specs.split(","):
        spec = raw.strip()
        if not spec:
            continue
        head = copy.deepcopy(template)
        if spec == "original":
            name, config = "original", None
        else:
            path = Path(spec)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            head.load_state_dict(payload["state_dict"])
            name, config = path.stem, payload.get("config")
        if name in heads:
            msg = f"duplicate head name {name!r} in --heads"
            raise SystemExit(msg)
        heads[name] = (head, config)
    if not heads:
        msg = "--heads resolved to no heads"
        raise SystemExit(msg)
    return heads


def _eval_head(  # noqa: PLR0913, PLR0914, PLR0917
    head: Module,
    tokenizer: ActionTokenizer,
    data: dict[str, Tensor],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    n = data["features"].shape[0]
    # fresh seeded generator per head -> identical sampled codes across heads
    generator = torch.Generator().manual_seed(seed)

    ratios: list[Tensor] = []
    entropies: list[Tensor] = []
    pred_conflicts: list[Tensor] = []
    gt_conflicts: list[Tensor] = []
    tf_l1: list[Tensor] = []
    sampled_l1: list[Tensor] = []
    abs_argmax_sum = abs_table_sum = 0.0
    abs_argmax_numel = abs_table_numel = 0

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            features = data["features"][start:stop].to(device).float()
            code_logits = data["code_logits"][start:stop].to(device).float()
            target_codes = data["target_codes"][start:stop].long().to(device)
            target = data["target"][start:stop].to(device)

            offsets = _rearrange_offsets(head(features))

            # (a) cancellation ratio, same combo scheme as offset_cancellation_probe
            ratio, entropy = _cancellation_batch(tokenizer, code_logits, offsets)
            ratios.append(ratio.cpu())
            entropies.append(entropy.cpu())

            # (b) pedal conflicts at argmax (export-time) decoding
            argmax_codes = code_logits.argmax(dim=-1)
            offset_argmax = _gather_offset(offsets, argmax_codes)
            pred = (tokenizer.invert(argmax_codes) + offset_argmax).unflatten(
                -1, (-1, N_FIELDS)
            )
            gt = target.unflatten(-1, (-1, N_FIELDS))
            pred_conflicts.append(
                (
                    (pred[..., GAS_DIM] > GAS_THRESH)
                    & (pred[..., BRAKE_DIM] > BRAKE_THRESH)
                ).cpu()
            )
            gt_conflicts.append(
                (
                    (gt[..., GAS_DIM] > GAS_THRESH)
                    & (gt[..., BRAKE_DIM] > BRAKE_THRESH)
                ).cpu()
            )

            # (c) offset L1 under both supervision styles (metrics only)
            tf_pred = tokenizer.invert(target_codes) + _gather_offset(
                offsets, target_codes
            )
            tf_l1.append((tf_pred - target).abs().mean(dim=-1).cpu())
            sampled_codes = _sample_codes_seeded(code_logits, generator)
            sampled_pred = tokenizer.invert(sampled_codes) + _gather_offset(
                offsets, sampled_codes
            )
            sampled_l1.append((sampled_pred - target).abs().mean(dim=-1).cpu())

            # (d) offset magnitudes
            abs_argmax_sum += float(offset_argmax.abs().sum())
            abs_argmax_numel += offset_argmax.numel()
            abs_table_sum += float(offsets.abs().sum())
            abs_table_numel += offsets.numel()

    entropy_q0 = torch.cat(entropies)
    return {
        "cancellation": _aggregate_cancellation(torch.cat(ratios), entropy_q0),
        "pedal_conflict": summarize_pedal_conflicts({
            "pred_conflict": torch.cat(pred_conflicts),
            "gt_conflict": torch.cat(gt_conflicts),
            "entropy_q0": entropy_q0,
        }),
        "offset_l1": {
            "teacher_forced_style": float(torch.cat(tf_l1).mean()),
            "sampled_style": float(torch.cat(sampled_l1).mean()),
        },
        "offset_magnitude": {
            "mean_abs_at_argmax": abs_argmax_sum / abs_argmax_numel,
            "mean_abs_full_table": abs_table_sum / abs_table_numel,
        },
    }


def _fmt(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4f}"


def run_eval(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)
    data, meta = _load_cache(Path(args.cache))
    tokenizer, template = _policy_parts(args.ckpt, args.device)
    heads = _build_heads(args.heads, template)
    print(  # noqa: T201
        f"[eval] {data['features'].shape[0]} cached samples "
        f"(split={meta['split']}), heads: {list(heads)}"
    )

    results: dict[str, Any] = {
        "config": {
            "ckpt": str(args.ckpt),
            "cache": str(args.cache),
            "cache_meta": meta,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "decoding": (
                "argmax (export parity) for cancellation/conflict; sampled-style "
                "L1 uses seeded multinomial over cached code logits"
            ),
            "thresholds": {"gas": GAS_THRESH, "brake": BRAKE_THRESH},
        },
        "heads": {},
    }
    for name, (head, train_config) in heads.items():
        head.to(device).float().eval()
        t0 = time.perf_counter()
        entry = _eval_head(head, tokenizer, data, device, args.batch_size, args.seed)
        if train_config is not None:
            entry["train_config"] = {
                key: value
                for key, value in train_config.items()
                if key != "loss_history"
            }
        results["heads"][name] = entry

        per_field = entry["cancellation"]["overall"]["per_field"]
        print(  # noqa: T201
            f"[eval] {name}: R_median gas={_fmt(per_field['gas_pedal']['median_R'])} "
            f"brake={_fmt(per_field['brake_pedal']['median_R'])} "
            f"steer={_fmt(per_field['steering_angle']['median_R'])} | "
            f"pred_frame_conflict_rate="
            f"{entry['pedal_conflict']['predicted']['frame_conflict_rate']:.5f} | "
            f"L1 tf={entry['offset_l1']['teacher_forced_style']:.5f} "
            f"sampled={entry['offset_l1']['sampled_style']:.5f} "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] results written to {out}")  # noqa: T201


# --- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="offset_head_retrain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=42)

    extract = subparsers.add_parser(
        "extract", parents=[common], help="cache per-sample policy tensors to disk"
    )
    extract.add_argument("--split", required=True, choices=["train", "val"])
    extract.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="cap on cached samples (default: 150000 for train, full split for val)",
    )
    extract.add_argument(
        "--out", required=True, help="cache directory (refused if non-empty)"
    )
    extract.add_argument("--batch-size", type=int, default=48)
    extract.add_argument("--num-workers", type=int, default=3)
    extract.set_defaults(func=run_extract)

    train = subparsers.add_parser(
        "train",
        parents=[common],
        help="retrain a copy of the checkpoint's offset head on cached tensors",
    )
    train.add_argument("--cache", required=True, help="extract-stage cache directory")
    train.add_argument(
        "--variant", required=True, choices=["sampled", "teacher_forced"]
    )
    train.add_argument("--steps", type=int, default=6000)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument(
        "--lr-schedule", default="cosine", choices=["cosine", "constant"]
    )
    train.add_argument("--out", required=True, help="output .pt for the retrained head")
    train.set_defaults(func=run_train)

    evaluate = subparsers.add_parser(
        "eval",
        parents=[common],
        help="compare offset heads side by side on a cached val split",
    )
    evaluate.add_argument(
        "--cache", required=True, help="extract-stage VAL cache directory"
    )
    evaluate.add_argument(
        "--heads",
        required=True,
        help="comma-separated: 'original' and/or retrained-head .pt paths",
    )
    evaluate.add_argument("--batch-size", type=int, default=512)
    evaluate.add_argument("--out", default=str(DEFAULT_EVAL_OUT))
    evaluate.set_defaults(func=run_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    import multiprocessing as mp
    import os
    import sys

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()

    # same rationale as offset_cancellation_probe: torchdata.nodes worker-thread
    # teardown intermittently aborts at interpreter exit; all outputs are
    # flushed by now, so skip interpreter teardown entirely
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
