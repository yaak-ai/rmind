#!/usr/bin/env python
"""Validation gates for an LFG label root (`lfg_aux_supervision_task.md` SS7.1/SS7.2).

The Stage 1 checks were done ad hoc and never committed, so they could not be re-run against a new
label root. They are gates, not one-offs -- `v2` needs all of them, and so will any future root.

    # SS7.1 spatial alignment: road-like class must dominate the BOTTOM patch rows, sky the top.
    # A flip/transpose bug here is invisible in the loss, which is why this is the cheapest gate.
    uv run python scripts/lfg_validate_labels.py spatial \\
        --label-root /nasa/drives/yaak/lfg_labels/v2 \\
        --drive Niro096-HQ/2023-01-11--13-47-36

    # SS7.2 datamodule round-trip: what the dataloader delivers must equal the bytes on disk, for
    # the first/middle/last sample of every val drive (the original check only fetched ds[0], which
    # is why it missed the SS12 phase bug on 4 of 5 drives).
    uv run python scripts/lfg_validate_labels.py roundtrip \\
        --label-root /nasa/drives/yaak/lfg_labels/v2

    # Divergence between two roots, e.g. the nearest-linked tree vs exact labels. Expect ~98-99%
    # per-patch agreement and motion MAE ~0.01-0.02 (SS12.4); a much larger gap means something
    # other than the temporal offset changed.
    uv run python scripts/lfg_validate_labels.py agree \\
        --a /nasa/drives/yaak/lfg_labels/v1_nearest \\
        --b /nasa/drives/yaak/lfg_labels/v2

Every subcommand exits 1 on failure so it can gate a launch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rmind.utils.lfg_labels import LFG_LABEL_NBYTES, decode_lfg_label  # noqa: E402

SEG, PURITY, MOTION, CONF = 0, 1, 2, 3
ROWS = 2  # patch rows compared at top vs bottom
PURITY_MIN = 0.6  # matches config .../dinov2_dinowm_causal_lfgaux.yaml aux_purity_min


def read_manifest(label_root: Path, drive_id: str) -> dict:
    return json.loads((label_root / drive_id / "manifest.json").read_text())


def load_label(label_root: Path, drive_id: str, frame_idx: int) -> np.ndarray:
    return decode_lfg_label((label_root / drive_id / f"{frame_idx:09d}.bin").read_bytes())


def cmd_spatial(args: argparse.Namespace) -> int:
    manifest = read_manifest(args.label_root, args.drive)
    frame_indices = manifest["frame_indices"]
    top = np.zeros(7, dtype=np.int64)
    bottom = np.zeros(7, dtype=np.int64)
    for frame_idx in frame_indices:
        seg = load_label(args.label_root, args.drive, frame_idx)[SEG]
        top += np.bincount(seg[:ROWS].ravel(), minlength=7)
        bottom += np.bincount(seg[-ROWS:].ravel(), minlength=7)

    c_bottom, c_top = int(bottom.argmax()), int(top.argmax())
    bottom_frac = bottom[c_bottom] / bottom.sum()
    top_frac = top[c_bottom] / top.sum()
    print(f"{args.drive}: {len(frame_indices)} frames, {manifest.get('frame_selection')}")
    print(f"  dominant class in bottom {ROWS} rows: {c_bottom} ({bottom_frac:.1%} of patches)")
    print(f"  dominant class in top {ROWS} rows:    {c_top} ({top[c_top] / top.sum():.1%})")
    print(f"  class {c_bottom} share: bottom {bottom_frac:.3f} vs top {top_frac:.3f}")

    if c_bottom == c_top or bottom_frac <= top_frac:
        print("SS7.1 FAIL: no vertical structure -- suspect a flip/transpose", file=sys.stderr)
        return 1
    print("SS7.1 PASS")
    return 0


def cmd_agree(args: argparse.Namespace) -> int:
    # drive ids are two levels deep (NiroNNN-HQ/date), so rebuild them from the manifest paths
    drive_ids = sorted(
        str(p.parent.relative_to(args.a)) for p in args.a.glob("*/*/manifest.json")
    )
    if not drive_ids:
        print(f"no manifests under {args.a}", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    drive_ids = rng.sample(drive_ids, min(args.drives, len(drive_ids)))

    n_patch = n_agree = 0
    motion_abs = 0.0
    n_motion = 0
    n_frames = 0
    missing = 0
    per_frame = max(1, args.frames // max(1, len(drive_ids)))
    for drive_id in drive_ids:
        try:
            frame_indices = read_manifest(args.a, drive_id)["frame_indices"]
        except OSError:
            continue
        for frame_idx in rng.sample(frame_indices, min(per_frame, len(frame_indices))):
            try:
                a = load_label(args.a, drive_id, frame_idx)
                b = load_label(args.b, drive_id, frame_idx)
            except (OSError, ValueError):
                missing += 1
                continue
            n_frames += 1
            # the aux loss only supervises confident, spatially pure patches -- score those
            mask = (a[PURITY] >= PURITY_MIN * 255) & (a[CONF] > 0)
            n_patch += int(mask.sum())
            n_agree += int((a[SEG][mask] == b[SEG][mask]).sum())
            motion_abs += float(
                np.abs(a[MOTION].astype(np.int16) - b[MOTION].astype(np.int16)).sum()
            ) / 255.0
            n_motion += a[MOTION].size

    if not n_frames:
        print("no comparable frames found", file=sys.stderr)
        return 1
    print(f"compared {n_frames} frames across {len(drive_ids)} drives ({missing} unreadable)")
    print(f"  seg agreement (purity>={PURITY_MIN}, conf>0): {n_agree / max(1, n_patch):.4f} "
          f"over {n_patch} patches")
    print(f"  motion MAE: {motion_abs / max(1, n_motion):.4f}")
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    import hydra
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment={args.experiment}",
                f"paths.lfg_labels={args.label_root}",
                # never the live run's cache: pipefunc's resume=False wipes a run folder on start
                # and filter-node outputs are keyed by output name only, so pointing this at a
                # training run's cache can wipe or cross-contaminate it
                f"paths.rbyte.cache={args.rbyte_cache}",
            ],
        )
        dataset = hydra.utils.instantiate(cfg.datamodule.val.dataset)

    n_checked = n_bad = 0
    total = len(dataset)
    print(f"val dataset: {total} samples")
    indices = _per_drive_probe_indices(dataset, total)

    for i in indices:
        sample = dataset[i]
        drive_id = _sample_drive_id(sample)
        labels = np.asarray(sample["data"]["lfg_labels"])
        frame_indices = np.asarray(
            sample["meta"]["ImageMetadata.cam_front_left"]["frame_idx"]
        ).ravel()
        if labels.shape[0] != frame_indices.shape[0]:
            print(
                f"sample {i}: {labels.shape[0]} labels vs {frame_indices.shape[0]} frame_idx",
                file=sys.stderr,
            )
            return 1
        for slot, frame_idx in enumerate(frame_indices):
            path = args.label_root / drive_id / f"{int(frame_idx):09d}.bin"
            raw = path.read_bytes()
            if len(raw) != LFG_LABEL_NBYTES:
                print(f"{path}: {len(raw)} bytes", file=sys.stderr)
                return 1
            n_checked += 1
            if not np.array_equal(labels[slot], decode_lfg_label(raw)):
                n_bad += 1
                print(f"MISMATCH sample {i} slot {slot} frame {int(frame_idx)} {path}", file=sys.stderr)

    print(f"SS7.2: {n_checked} (frame, label) pairs checked, {n_bad} mismatched")
    if n_bad:
        return 1
    print("SS7.2 PASS")
    return 0


def _sample_drive_id(sample) -> str:
    value = sample["meta"]["input_id"]
    if isinstance(value, (list, tuple, np.ndarray)):
        value = np.asarray(value).ravel()[0]
    return str(value)


def _per_drive_probe_indices(dataset, total: int) -> list[int]:
    """First, middle and last sample index of every drive in the dataset.

    Scans `input_id` per sample rather than assuming contiguous per-drive blocks.
    """
    spans: dict[str, list[int]] = {}
    for i in range(total):
        drive_id = _sample_drive_id(dataset[i])
        spans.setdefault(drive_id, []).append(i)
    indices: list[int] = []
    for drive_id, idxs in sorted(spans.items()):
        indices += sorted({idxs[0], idxs[len(idxs) // 2], idxs[-1]})
    print(f"probing {len(indices)} samples across {len(spans)} drives")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("spatial", help="SS7.1 road/sky vertical-structure check")
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--drive", required=True)
    p.set_defaults(func=cmd_spatial)

    p = sub.add_parser("agree", help="divergence between two label roots")
    p.add_argument("--a", type=Path, required=True, help="reference root (drives enumerated here)")
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--drives", type=int, default=30)
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_agree)

    p = sub.add_parser("roundtrip", help="SS7.2 dataloader round-trip over every val drive")
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument(
        "--experiment", default="yaak/patch_policy/dinov2_dinowm_causal_lfgaux"
    )
    p.add_argument(
        "--rbyte-cache",
        default=".rbyte_cache_lfg_gate",
        help=(
            "throwaway rbyte cache dir for this check. MUST NOT be a cache a training run is "
            "using (default: .rbyte_cache_lfg_gate)"
        ),
    )
    p.set_defaults(func=cmd_roundtrip)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
