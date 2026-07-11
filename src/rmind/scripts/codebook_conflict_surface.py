"""Codebook conflict surface + clean-codebook heads retrain (tier 2b).

Two no-dataloader diagnostics for the offset-supervision bug, driven entirely
by the cached shards from `offset_head_retrain extract` and local
`ActionTokenizer` checkpoints (no datamodule, no `.rbyte_cache`):

- `surface`: decode ALL 16^4 = 65536 residual-VQ code combinations of each
  tokenizer (`invert` on the full cartesian product) and measure how much of
  the codebook itself decodes to a gas+brake conflict — before any policy
  head is involved. Reported per tokenizer: unweighted % of combos whose
  decode conflicts (any of the 6 horizon frames; also the per-frame mean),
  under both the sensor-derived thresholds of `pedal_conflict_stats` and a
  flat 0.02/0.02 set. Real-data impact is quantified on BOTH cached splits
  (val and train): USAGE-WEIGHTED % (each combo weighted by its empirical
  ground-truth frequency under that tokenizer; GT codes recomputed from the
  cached normalized target, raw chunk reconstructed by undoing the
  turn_signal Scaler (0,2)->(0,1)), the GT-decode conflict rate (fraction of
  samples whose own GT-code decode conflicts — the codebook floor under
  perfect code prediction), usage concentration (how many distinct combos
  cover 90% of GT usage and how many of those conflict), and — for the OLD
  tokenizer — the policy-side hit rate: the fraction of val samples whose
  cached-code-logits argmax combo decodes to a conflict (offset-free). The
  chunk reconstruction is validated by round-tripping the OLD tokenizer: its
  recomputed codes must match the cached `target_codes` for >= 99% of
  samples.

- `tier2b`: clean-codebook heads retrain on frozen features. Both the
  `code_head` and `offset_head` are initialized from the BUGGED finetuned
  checkpoint and retrained jointly on the cached train features against GT
  codes recomputed under a CLEAN tokenizer (the code head's outputs index a
  new codebook, so it relearns). Loss = per-quantizer FocalLoss on the new GT
  codes + L1(invert(gt_codes) + gather_offset(offsets, gt_codes), target)
  (teacher-forced), plus a control variant whose offset loss uses codes
  SAMPLED from the training head's own logits each step (the pre-fix signal).
  Eval on cached val mirrors `offset_head_retrain eval`: predicted pedal
  conflicts at argmax decoding (with the gathered offset, and offset-free —
  the clean-codebook analogue of the surface policy hit rate) bucketed by
  the NEW code logits' entropy, cancellation ratio R per field,
  per-quantizer code top-1 accuracy, offset magnitudes, and L1 to target.

Usage:
    uv run python -m rmind.scripts.codebook_conflict_surface surface \\
        --val-cache caches/offset_head/val
    uv run python -m rmind.scripts.codebook_conflict_surface tier2b \\
        --train-cache caches/offset_head/train --val-cache caches/offset_head/val
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

from rmind.components.loss import FocalLoss
from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.models.action_tokenizer import ActionTokenizer
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
    load_policy,
)
from rmind.scripts.offset_head_retrain import _load_cache, _rearrange_offsets
from rmind.scripts.pedal_conflict_probe import BRAKE_DIM, GAS_DIM
from rmind.scripts.pedal_conflict_probe import summarize as summarize_pedal_conflicts
from rmind.scripts.pedal_conflict_stats import BRAKE_THRESH, GAS_THRESH

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

G = EXPECTED_NUM_QUANTIZERS
C = EXPECTED_CODEBOOK_SIZE
N_COMBOS = C**G  # 65536
N_FIELDS = len(FIELD_NAMES)
HORIZON = 6
TURN_SIGNAL_DIM = 3  # Scaler (0, 2) -> (0, 1): raw = normalized * 2

# threshold sets: the sensor-derived ones from pedal_conflict_stats plus the
# flat 0.02/0.02 pair behind the "42.2% of combos conflict" observation
THRESHOLD_SETS: dict[str, tuple[float, float]] = {
    "script": (GAS_THRESH, BRAKE_THRESH),
    "flat_0.02": (0.02, 0.02),
}

TOKENIZER_CKPTS: dict[str, Path] = {
    "old_dirty_gkzgn6gk": REPO_ROOT / "artifacts" / "model-gkzgn6gk:v9" / "model.ckpt",
    "clean_baseline_y74asdtd": (
        REPO_ROOT / "artifacts" / "model-y74asdtd:v9" / "model.ckpt"
    ),
    "clean_throttle_s7al9vc8": (
        REPO_ROOT / "artifacts" / "model-s7al9vc8:v9" / "model.ckpt"
    ),
}
OLD_TOKENIZER = "old_dirty_gkzgn6gk"
DEFAULT_CLEAN_TOKENIZER = "clean_baseline_y74asdtd"
MIN_ROUNDTRIP_MATCH_RATE = 0.99

DEFAULT_SURFACE_OUT = REPO_ROOT / "diag_results" / "codebook_conflict_surface.json"
DEFAULT_TIER2B_OUT = REPO_ROOT / "diag_results" / "tier2b_clean_codebook.json"

_gather_offset = JointPolicyObjective._gather_offset  # noqa: SLF001


def load_tokenizer(ckpt: str | Path, device: str) -> ActionTokenizer:
    """Load a frozen ActionTokenizer from a local checkpoint.

    Raises:
        ValueError: if the quantizer geometry is not G=4, C=16.
    """
    tokenizer = ActionTokenizer.load_from_checkpoint(
        Path(ckpt), map_location="cpu", weights_only=False
    )
    tokenizer = tokenizer.to(device).eval().requires_grad_(requires_grad=False)
    quantizer = tokenizer.quantizer
    if (quantizer.num_quantizers, quantizer.codebook_size) != (G, C):
        msg = (
            f"{ckpt}: unexpected quantizer geometry "
            f"G={quantizer.num_quantizers} C={quantizer.codebook_size}"
        )
        raise ValueError(msg)
    return tokenizer


def reconstruct_chunk(target: Tensor) -> Tensor:
    """Cached normalized target (n, 24) -> raw action chunk (n, 6, 4).

    Continuous fields are Identity-normalized; turn_signal was scaled
    (0, 2) -> (0, 1), so the raw chunk is recovered by doubling that column.
    """
    chunk = target.reshape(-1, HORIZON, N_FIELDS).clone()
    chunk[..., TURN_SIGNAL_DIM] *= 2.0
    return chunk


def compute_codes(
    tokenizer: ActionTokenizer, chunk: Tensor, device: str, batch_size: int = 8192
) -> Tensor:
    """GT codes (n, G) int64 on CPU for a raw chunk (n, 6, 4), batched."""
    codes: list[Tensor] = []
    with torch.inference_mode():
        codes.extend(
            tokenizer(chunk[start : start + batch_size].to(device)).cpu()
            for start in range(0, chunk.shape[0], batch_size)
        )
    return torch.cat(codes).long()


def all_combos(device: str) -> Tensor:
    """All C^G code combinations (N_COMBOS, G), row i = base-C digits of i."""
    return torch.cartesian_prod(
        *(torch.arange(C, device=device) for _ in range(G))
    ).long()


def combo_index(codes: Tensor) -> Tensor:
    """(n, G) codes -> flat combo index matching `all_combos` row order."""
    weights = C ** torch.arange(G - 1, -1, -1, dtype=torch.long)
    return (codes.cpu() * weights).sum(dim=-1)


def decode_all_combos(
    tokenizer: ActionTokenizer, device: str, batch_size: int = 8192
) -> Tensor:
    """Decode the full codebook: (N_COMBOS, 6, 4) on CPU, normalized space."""
    combos = all_combos(device)
    frames: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, N_COMBOS, batch_size):
            decoded = tokenizer.invert(combos[start : start + batch_size])
            frames.append(decoded.unflatten(-1, (-1, N_FIELDS)).cpu())
    return torch.cat(frames)


def conflict_frames(frames: Tensor, gas_thresh: float, brake_thresh: float) -> Tensor:
    """(..., 6, 4) decoded frames -> (..., 6) bool gas+brake conflict mask."""
    return (frames[..., GAS_DIM] > gas_thresh) & (frames[..., BRAKE_DIM] > brake_thresh)


def _conflict_summary(frame_mask: Tensor, weights: Tensor | None = None) -> dict:
    """Any-frame and per-frame conflict rates, optionally weighted per row."""
    any_frame = frame_mask.any(dim=-1).float()
    frame_rate = frame_mask.float().mean(dim=-1)
    if weights is None:
        return {
            "any_frame_rate": float(any_frame.mean()),
            "frame_rate": float(frame_rate.mean()),
        }
    total = float(weights.sum())
    return {
        "any_frame_rate": float((any_frame * weights).sum()) / total,
        "frame_rate": float((frame_rate * weights).sum()) / total,
    }


def _top_usage_stats(
    counts: Tensor, combo_any_conflict: Tensor, coverage: float = 0.9
) -> dict[str, Any]:
    """Concentration of GT usage: combos covering `coverage`, conflicts among them.

    A few conflicting-but-frequent combos matter more than thousands of dead
    ones; this reports how many distinct combos cover `coverage` of the GT
    usage, how many of those conflict, and the usage share of the conflicting
    ones within the whole split.
    """
    sorted_counts, order = counts.sort(descending=True)
    total = float(counts.sum())
    cumulative = sorted_counts.cumsum(dim=0)
    k = int((cumulative < coverage * total).sum()) + 1
    top = order[:k]
    top_conflict = combo_any_conflict[top]
    return {
        "coverage": coverage,
        "n_combos": k,
        "n_conflicting": int(top_conflict.sum()),
        "conflicting_usage_share": float(counts[top][top_conflict].sum() / total),
    }


def _split_stats(
    counts: Tensor, indices: Tensor, masks: dict[str, Tensor]
) -> dict[str, Any]:
    """Per-split usage-weighted, GT-decode, and concentration stats."""
    n = indices.shape[0]
    entry: dict[str, Any] = {
        "n_samples": n,
        "n_unique_combos": int((counts > 0).sum()),
        "top_combo_share": float(counts.max() / n),
    }
    for set_name, mask in masks.items():
        entry[set_name] = {
            "usage_weighted": _conflict_summary(mask, counts),
            # fraction of samples whose own GT-code decode conflicts (the
            # codebook floor under perfect code prediction); equals
            # usage_weighted by construction (consistency check)
            "gt_decode": _conflict_summary(mask[indices]),
            "top_usage": _top_usage_stats(counts, mask.any(dim=-1)),
        }
    return entry


def run_surface(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    caches: dict[str, tuple[dict[str, Tensor], dict[str, Any]]] = {
        "val": _load_cache(Path(args.val_cache)),
        "train": _load_cache(Path(args.train_cache)),
    }
    chunks = {
        split: reconstruct_chunk(data["target"]) for split, (data, _) in caches.items()
    }
    for split, (data, meta) in caches.items():
        print(  # noqa: T201
            f"[surface] {split} cache: {data['target'].shape[0]} samples "
            f"(split={meta['split']})"
        )

    results: dict[str, Any] = {
        "config": {
            "val_cache": str(args.val_cache),
            "train_cache": str(args.train_cache),
            "cache_meta": {split: meta for split, (_, meta) in caches.items()},
            "n_combos": N_COMBOS,
            "thresholds": {
                name: {"gas": gas, "brake": brake}
                for name, (gas, brake) in THRESHOLD_SETS.items()
            },
            "tokenizers": {k: str(v) for k, v in TOKENIZER_CKPTS.items()},
        },
        "tokenizers": {},
    }

    for name, ckpt in TOKENIZER_CKPTS.items():
        tokenizer = load_tokenizer(ckpt, args.device)

        # full-codebook decode: is the conflict baked into the codebook?
        combo_frames = decode_all_combos(tokenizer, args.device)  # (65536, 6, 4)
        masks = {
            set_name: conflict_frames(combo_frames, gas, brake)  # (65536, 6)
            for set_name, (gas, brake) in THRESHOLD_SETS.items()
        }

        entry: dict[str, Any] = {"ckpt": str(ckpt)}
        for set_name, mask in masks.items():
            entry[set_name] = {"unweighted": _conflict_summary(mask)}

        # empirical GT code usage on both cached splits under THIS tokenizer
        entry["splits"] = {}
        for split, (data, _) in caches.items():
            gt_codes = compute_codes(tokenizer, chunks[split], args.device)
            if name == OLD_TOKENIZER:
                # round-trip check: the chunk reconstruction is only valid if
                # the OLD tokenizer reproduces the cached target codes
                match = (gt_codes == data["target_codes"].long()).all(dim=-1)
                match_rate = float(match.float().mean())
                results.setdefault("old_tokenizer_code_match_rate", {})[split] = (
                    match_rate
                )
                print(  # noqa: T201
                    f"[surface] old-tokenizer round-trip match rate "
                    f"({split}): {match_rate:.4f}"
                )
                if split == "val" and match_rate < MIN_ROUNDTRIP_MATCH_RATE:
                    msg = (
                        f"chunk reconstruction failed round-trip: old-tokenizer "
                        f"code match rate {match_rate:.4f} < "
                        f"{MIN_ROUNDTRIP_MATCH_RATE}"
                    )
                    raise SystemExit(msg)

            indices = combo_index(gt_codes)
            counts = torch.bincount(indices, minlength=N_COMBOS).float()
            entry["splits"][split] = _split_stats(counts, indices, masks)

        if name == OLD_TOKENIZER:
            # policy-side hit rate: which combos does the deployed (bugged)
            # policy actually visit? cached code logits -> argmax -> decode,
            # offset-free, val split
            argmax_idx = combo_index(caches["val"][0]["code_logits"].argmax(dim=-1))
            entry["policy_argmax_val"] = {
                set_name: _conflict_summary(mask[argmax_idx])
                for set_name, mask in masks.items()
            }

        results["tokenizers"][name] = entry
        print(  # noqa: T201
            f"[surface] {name}: unweighted any-frame "
            f"script={entry['script']['unweighted']['any_frame_rate']:.4f} "
            f"flat={entry['flat_0.02']['unweighted']['any_frame_rate']:.4f} | "
            f"usage-weighted any-frame (script) "
            f"val={entry['splits']['val']['script']['usage_weighted']['any_frame_rate']:.4f} "
            f"train={entry['splits']['train']['script']['usage_weighted']['any_frame_rate']:.4f}"
        )

        del tokenizer, combo_frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["wall_time_s"] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[surface] results written to {out} ({results['wall_time_s']:.1f}s)")  # noqa: T201


# --- tier2b ---------------------------------------------------------------------


def _bugged_heads(ckpt: str | Path) -> tuple[Module, Module]:
    """The bugged checkpoint's code and offset heads (CPU); the rest is freed."""
    model = load_policy(ckpt, "cpu")
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    code_head = copy.deepcopy(policy.code_head)
    offset_head = copy.deepcopy(policy.offset_head)
    del model, policy
    gc.collect()
    return code_head, offset_head


def _train_heads(  # noqa: PLR0913, PLR0914, PLR0917
    variant: str,
    code_head: Module,
    offset_head: Module,
    tokenizer: ActionTokenizer,
    data: dict[str, Tensor],
    gt_codes: Tensor,
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    """Jointly retrain both heads on frozen cached features; returns loss history.

    Raises:
        RuntimeError: if the heads are not trainable or the tokenizer not frozen.
    """
    device = torch.device(args.device)
    n = data["features"].shape[0]
    code_loss_fn = FocalLoss()  # gamma=2.0, matching the finetune config
    params = list(code_head.parameters()) + list(offset_head.parameters())
    if not all(p.requires_grad for p in params) or any(
        p.requires_grad for p in tokenizer.parameters()
    ):
        msg = "heads must be trainable and the tokenizer frozen"
        raise RuntimeError(msg)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # CPU generator drives batch indices and (sampled variant) multinomial
    generator = torch.Generator().manual_seed(args.seed)

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        index = torch.randint(0, n, (args.batch_size,), generator=generator)
        features = data["features"][index].to(device).float()
        target = data["target"][index].to(device)
        target_codes = gt_codes[index].to(device)

        code_logits = rearrange(code_head(features), "b (g c) -> b g c", g=G, c=C)
        offsets = _rearrange_offsets(offset_head(features))

        code_loss = sum(
            code_loss_fn(code_logits[:, q, :], target_codes[:, q]) for q in range(G)
        )

        if variant == "sampled":
            # the pre-fix signal: offsets supervised at codes sampled from the
            # (current) head's own logits, independent of the GT action
            offset_codes = _sample_codes_seeded(code_logits.detach(), generator)
        else:
            offset_codes = target_codes
        with torch.no_grad():
            decoded = tokenizer.invert(offset_codes)
        offset_loss = F.l1_loss(decoded + _gather_offset(offsets, offset_codes), target)

        loss = code_loss + offset_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "code_loss": float(code_loss.detach()),
                "offset_loss": float(offset_loss.detach()),
            }
            history.append(entry)
            rate = step / (time.perf_counter() - t0)
            print(  # noqa: T201
                f"[tier2b/{variant}] step {step}/{args.steps} "
                f"loss={entry['loss']:.5f} code={entry['code_loss']:.5f} "
                f"offset={entry['offset_loss']:.5f} ({rate:.2f} steps/s)"
            )
    return history


def _eval_heads(  # noqa: PLR0913, PLR0914, PLR0917
    code_head: Module,
    offset_head: Module,
    tokenizer: ActionTokenizer,
    data: dict[str, Tensor],
    gt_codes: Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Val-split eval mirroring `offset_head_retrain eval`, with NEW code logits."""
    device = torch.device(args.device)
    n = data["features"].shape[0]

    ratios: list[Tensor] = []
    entropies: list[Tensor] = []
    pred_conflicts: list[Tensor] = []
    pred_conflicts_no_offset: list[Tensor] = []
    gt_conflicts: list[Tensor] = []
    code_correct: list[Tensor] = []
    tf_l1: list[Tensor] = []
    argmax_l1: list[Tensor] = []
    abs_argmax_sum = abs_table_sum = 0.0
    abs_argmax_numel = abs_table_numel = 0

    with torch.inference_mode():
        for start in range(0, n, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, n)
            features = data["features"][start:stop].to(device).float()
            target = data["target"][start:stop].to(device)
            target_codes = gt_codes[start:stop].to(device)

            code_logits = rearrange(code_head(features), "b (g c) -> b g c", g=G, c=C)
            offsets = _rearrange_offsets(offset_head(features))

            # (b) cancellation ratio on the NEW logits + retrained offsets
            ratio, entropy = _cancellation_batch(tokenizer, code_logits, offsets)
            ratios.append(ratio.cpu())
            entropies.append(entropy.cpu())

            # (a) pedal conflicts at argmax (export-time) decoding, with the
            # gathered offset and offset-free (the pure combo-surface hit rate)
            argmax_codes = code_logits.argmax(dim=-1)
            offset_argmax = _gather_offset(offsets, argmax_codes)
            decoded = tokenizer.invert(argmax_codes)
            pred = (decoded + offset_argmax).unflatten(-1, (-1, N_FIELDS))
            gt = target.unflatten(-1, (-1, N_FIELDS))
            pred_conflicts.append(conflict_frames(pred, GAS_THRESH, BRAKE_THRESH).cpu())
            pred_conflicts_no_offset.append(
                conflict_frames(
                    decoded.unflatten(-1, (-1, N_FIELDS)), GAS_THRESH, BRAKE_THRESH
                ).cpu()
            )
            gt_conflicts.append(conflict_frames(gt, GAS_THRESH, BRAKE_THRESH).cpu())

            # (c) per-quantizer top-1 accuracy vs the new GT codes
            code_correct.append((argmax_codes == target_codes).cpu())

            # (e) L1 to target, teacher-forced style and argmax decode
            tf_pred = tokenizer.invert(target_codes) + _gather_offset(
                offsets, target_codes
            )
            tf_l1.append((tf_pred - target).abs().mean(dim=-1).cpu())
            argmax_l1.append((pred.flatten(-2, -1) - target).abs().mean(dim=-1).cpu())

            # (d) offset magnitudes
            abs_argmax_sum += float(offset_argmax.abs().sum())
            abs_argmax_numel += offset_argmax.numel()
            abs_table_sum += float(offsets.abs().sum())
            abs_table_numel += offsets.numel()

    entropy_q0 = torch.cat(entropies)
    correct = torch.cat(code_correct)
    no_offset = torch.cat(pred_conflicts_no_offset)
    return {
        "pedal_conflict": summarize_pedal_conflicts({
            "pred_conflict": torch.cat(pred_conflicts),
            "gt_conflict": torch.cat(gt_conflicts),
            "entropy_q0": entropy_q0,
        }),
        # offset-free argmax decode: the clean-codebook analogue of the
        # surface subcommand's policy-side combo hit rate
        "pedal_conflict_argmax_offset_free": {
            "frame_conflict_rate": float(no_offset.float().mean()),
            "sample_any_frame_rate": float(no_offset.any(dim=-1).float().mean()),
        },
        "cancellation": _aggregate_cancellation(torch.cat(ratios), entropy_q0),
        "code_top1_accuracy": {
            "overall": float(correct.float().mean()),
            "per_quantizer": [float(correct[:, q].float().mean()) for q in range(G)],
        },
        "offset_magnitude": {
            "mean_abs_at_argmax": abs_argmax_sum / abs_argmax_numel,
            "mean_abs_full_table": abs_table_sum / abs_table_numel,
        },
        "l1_to_target": {
            "teacher_forced_style": float(torch.cat(tf_l1).mean()),
            "argmax_decode": float(torch.cat(argmax_l1).mean()),
        },
    }


def run_tier2b(args: argparse.Namespace) -> None:  # noqa: PLR0914
    t0 = time.perf_counter()
    device = torch.device(args.device)
    train_data, train_meta = _load_cache(Path(args.train_cache))
    val_data, val_meta = _load_cache(Path(args.val_cache))

    tokenizer = load_tokenizer(args.tokenizer_ckpt, args.device)
    print(f"[tier2b] clean tokenizer: {args.tokenizer_ckpt}")  # noqa: T201

    # GT codes under the CLEAN tokenizer, from the reconstructed raw chunks
    train_codes = compute_codes(
        tokenizer, reconstruct_chunk(train_data["target"]), args.device
    )
    val_codes = compute_codes(
        tokenizer, reconstruct_chunk(val_data["target"]), args.device
    )
    print(  # noqa: T201
        f"[tier2b] recomputed clean GT codes: train={train_codes.shape[0]} "
        f"(split={train_meta['split']}), val={val_codes.shape[0]} "
        f"(split={val_meta['split']})"
    )

    code_head_src, offset_head_src = _bugged_heads(args.ckpt)

    results: dict[str, Any] = {
        "config": {
            "policy_ckpt": str(args.ckpt),
            "tokenizer_ckpt": str(args.tokenizer_ckpt),
            "train_cache": str(args.train_cache),
            "val_cache": str(args.val_cache),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "code_loss": "FocalLoss(gamma=2.0), per quantizer, summed",
            "offset_loss": "L1(invert(codes) + gather_offset(offsets, codes), target)",
            "init": "both heads initialized from the bugged checkpoint",
            "thresholds": {"gas": GAS_THRESH, "brake": BRAKE_THRESH},
        },
        "heads": {},
    }

    for variant in ("teacher_forced", "sampled"):
        pl.seed_everything(args.seed, workers=True)
        code_head = copy.deepcopy(code_head_src).to(device).float()
        offset_head = copy.deepcopy(offset_head_src).to(device).float()
        code_head.requires_grad_(requires_grad=True).train()
        offset_head.requires_grad_(requires_grad=True).train()

        history = _train_heads(
            variant, code_head, offset_head, tokenizer, train_data, train_codes, args
        )

        code_head.eval()
        offset_head.eval()
        entry = _eval_heads(
            code_head, offset_head, tokenizer, val_data, val_codes, args
        )
        entry["loss_history"] = history
        results["heads"][variant] = entry

        conflict = entry["pedal_conflict"]["predicted"]
        top_bucket = conflict["buckets"][-1]
        per_field = entry["cancellation"]["overall"]["per_field"]
        print(  # noqa: T201
            f"[tier2b/{variant}] conflict frame rate="
            f"{conflict['frame_conflict_rate']:.5f} "
            f"(top entropy quartile {top_bucket['frame_conflict_rate']:.5f}) | "
            f"R_median gas={per_field['gas_pedal']['median_R']:.4f} "
            f"brake={per_field['brake_pedal']['median_R']:.4f} | "
            f"code_acc={entry['code_top1_accuracy']['overall']:.4f} | "
            f"L1 tf={entry['l1_to_target']['teacher_forced_style']:.5f}"
        )

        del code_head, offset_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results["wall_time_s"] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[tier2b] results written to {out} ({results['wall_time_s']:.1f}s)")  # noqa: T201


# --- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="codebook_conflict_surface",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--val-cache", default=str(REPO_ROOT / "caches" / "offset_head" / "val")
    )
    common.add_argument(
        "--train-cache", default=str(REPO_ROOT / "caches" / "offset_head" / "train")
    )

    surface = subparsers.add_parser(
        "surface",
        parents=[common],
        help="decode all 65536 code combos per tokenizer; conflict surface stats",
    )
    surface.add_argument("--out", default=str(DEFAULT_SURFACE_OUT))
    surface.set_defaults(func=run_surface)

    tier2b = subparsers.add_parser(
        "tier2b",
        parents=[common],
        help="retrain both heads jointly on frozen features with a clean codebook",
    )
    tier2b.add_argument(
        "--ckpt",
        default=str(DEFAULT_CKPT),
        help="bugged policy checkpoint (heads init)",
    )
    tier2b.add_argument(
        "--tokenizer-ckpt",
        default=str(TOKENIZER_CKPTS[DEFAULT_CLEAN_TOKENIZER]),
        help="clean ActionTokenizer checkpoint",
    )
    tier2b.add_argument("--steps", type=int, default=6000)
    tier2b.add_argument("--batch-size", type=int, default=4096)
    tier2b.add_argument("--lr", type=float, default=1e-4)
    tier2b.add_argument("--log-every", type=int, default=500)
    tier2b.add_argument("--eval-batch-size", type=int, default=512)
    tier2b.add_argument("--out", default=str(DEFAULT_TIER2B_OUT))
    tier2b.set_defaults(func=run_tier2b)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
