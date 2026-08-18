#!/usr/bin/env python
"""Derive (and verify) the exact set of frames the training pipeline requests per drive.

Why this exists
---------------
`lfg_aux_supervision_task.md` SS1.3 asserts that "only every 10th raw frame is ever loaded" and
therefore that labelling `frame_idx % 10 == 0` is sufficient. **That is wrong**, and the first
full-corpus training attempt died on a missing label file because of it.

The sampler is
`DataFrameGroupByDynamic(index_column=".../frame_idx", every="${episode_stride}i",
period=..., gather_every=${episode_step})`. `every="10i"` places *window boundaries* at multiples
of 10 in frame_idx, but `gather_every: 10` then takes every 10th **surviving row** inside a window
-- and the upstream `filtered` DuckDB query drops rows (`gear == '3'`, speed/pedal ranges,
`COLUMNS(*) IS NOT NULL`). So the first row of a window sits at whatever offset survived filtering.
Empirically the step within a clip is reliably 10, but the *phase* is per-drive and effectively
arbitrary: of 2,136,677 required train frames only 5.6% satisfy `frame_idx % 10 == 0`.

The only safe source of truth is the built samples table itself, which is what this script reads.

Because the answer depends on the dataset config (the `filtered` query, `clip_length`,
`episode_stride`/`episode_step`, and the drive list), the emitted JSON records the run folders it
came from. Re-extract after changing any of that.

Runs in the **rmind** venv (needs pipefunc + polars), unlike `lfg_label_drives.py` which runs in
the LFG venv. The JSON is the interface between the two.

    # 1. derive the required frames from an already-built samples run folder
    nix develop --command uv run python scripts/lfg_required_frames.py extract \\
        --run-folder ~/.rbyte_cache_causal32_lfgaux/yaak/train/clip37/samples \\
        --run-folder ~/.rbyte_cache_causal32_lfgaux/yaak/val/clip37/samples \\
        --output /nasa/drives/yaak/lfg_labels/required_frames_clip37.json

    # 2a. reuse the existing v1 labels by pointing each required frame at the nearest one
    #     (~7 min of hardlinking instead of ~18 h of GPU; see `link` below)
    nix develop --command uv run python scripts/lfg_required_frames.py link \\
        --required /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \\
        --source-root /nasa/drives/yaak/lfg_labels/v1 \\
        --dest-root /nasa/drives/yaak/lfg_labels/v1_nearest

    # 2b. or label the required frames exactly, with scripts/lfg_label_drives.py --frames-from

    # 3. either way, gate the training run on full coverage
    nix develop --command uv run python scripts/lfg_required_frames.py verify \\
        --required /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \\
        --label-root /nasa/drives/yaak/lfg_labels/v1_nearest

Nearest-frame substitution
--------------------------
Because v1's labels sit at `frame_idx % 10 == 0` and the required frames are stride-10 at a
per-drive phase, the nearest existing label is at most 5 frames (167 ms) away; mean 2.96 frames
(99 ms). Measured cost of substituting it, on 900 frames across 6 drives spanning every offset:
per-patch dominant-class agreement 98.1-100% over aux-supervised patches, motion MAE 0.007-0.020.
For scale, one full sample step (10 frames, the natural granularity of this supervision) changes
3.1-5.6% of patches with motion MAE 0.010-0.033 -- i.e. the substitution costs about a third of one
sample step. An offset-0 control reproduced v1 bit-identically on all 150 frames, confirming the
labelling is deterministic and that the residual really is the temporal gap.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# `pipefunc._utils.load` is private, but it is the only reader that understands how pipefunc's
# `file_array` storage encodes a polars DataFrame (parquet body behind a magic prefix, else
# cloudpickle). Reading one per-drive file at a time keeps this script's memory flat, whereas the
# public `load_outputs` would materialize the whole ~14.5 GB corpus table.
from pipefunc._utils import load

FRAME_IDX_COLUMN = "meta/ImageMetadata.cam_front_left/frame_idx"
INPUT_ID_FILE = "inputs/input_id.cloudpickle"
SAMPLES_CAST_TEMPLATE = "outputs/samples_cast/__{}__.pickle"
MAX_EXAMPLES = 10  # how many offending frames/drives to print before truncating


def required_frames_for_run_folder(run_folder: Path) -> dict[str, list[int]]:
    """Map `drive_id -> sorted frame_idx list` for one built samples run folder.

    Raises:
        FileNotFoundError: if a per-drive `samples_cast` output is absent (incomplete build).
        ValueError: if a drive appears twice with disagreeing frame sets.
    """
    drives: list[str] = load(run_folder / INPUT_ID_FILE)
    out: dict[str, list[int]] = {}
    for i, drive_id in enumerate(drives):
        path = run_folder / SAMPLES_CAST_TEMPLATE.format(i)
        if not path.is_file():
            msg = f"{run_folder} is missing {path.name} for drive {drive_id} -- build incomplete?"
            raise FileNotFoundError(msg)
        df = load(path)
        frames = sorted(set(df[FRAME_IDX_COLUMN].explode().unique().to_list()))
        if drive_id in out and out[drive_id] != frames:
            msg = f"drive {drive_id} appears twice with different frame sets"
            raise ValueError(msg)
        out[drive_id] = frames
        print(
            f"  [{i + 1}/{len(drives)}] {drive_id}: {len(frames)} frames",
            file=sys.stderr,
        )
    return out


def cmd_extract(args: argparse.Namespace) -> int:
    """Derive the required frame set and write it as JSON.

    Raises:
        ValueError: if two run folders disagree about a drive's required frames.
    """
    drives: dict[str, list[int]] = {}
    for run_folder in args.run_folders:
        print(f"reading {run_folder} ...", file=sys.stderr)
        for drive_id, frames in required_frames_for_run_folder(run_folder).items():
            if drive_id in drives and drives[drive_id] != frames:
                msg = (
                    f"drive {drive_id} has different required frames in two run folders; "
                    "train and val must not disagree"
                )
                raise ValueError(msg)
            drives[drive_id] = frames

    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_folders": [str(p) for p in args.run_folders],
        "frame_idx_column": FRAME_IDX_COLUMN,
        "n_drives": len(drives),
        "n_frames": sum(len(v) for v in drives.values()),
        "drives": drives,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print(
        f"\nwrote {args.output}: {payload['n_drives']} drives, {payload['n_frames']} frames",
        file=sys.stderr,
    )
    return 0


def _nearest(available: list[int], frame_idx: int) -> int:
    """Closest value in the sorted `available` list to `frame_idx`.

    Searches the source's real frame set rather than assuming a multiple of 10, so gaps in the
    source root widen the reported offset instead of silently mislinking.
    """
    j = bisect.bisect_left(available, frame_idx)
    candidates = [c for c in (j - 1, j) if 0 <= c < len(available)]
    return min((available[c] for c in candidates), key=lambda a: abs(a - frame_idx))


def _link_drive(
    args: argparse.Namespace, drive_id: str, frames: list[int]
) -> tuple[int, int, Counter[int], list[tuple[int, str, int, int]]]:
    """Link one drive; return (n_linked, n_skipped, offset histogram, unresolved frames).

    Raises:
        FileNotFoundError: if the source drive has no manifest to take labelled frames from.
    """
    src_manifest = args.source_root / drive_id / "manifest.json"
    if not src_manifest.is_file():
        msg = f"no manifest at {src_manifest}; cannot link drive {drive_id}"
        raise FileNotFoundError(msg)
    available = sorted(json.loads(src_manifest.read_text())["frame_indices"])
    if not available:
        msg = f"source manifest for {drive_id} lists no frames"
        raise FileNotFoundError(msg)

    dest_dir = args.dest_root / drive_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    linked = skipped = 0
    covered: list[int] = []
    hist: Counter[int] = Counter()
    unresolved: list[tuple[int, str, int, int]] = []

    for frame_idx in frames:
        dest = dest_dir / f"{frame_idx:09d}.bin"
        # An existing destination already satisfies this frame, whatever its provenance -- it may be
        # a real label written to fill an over-max_offset gap. Honour it before the offset check, or
        # such frames would be reported unresolved forever and left out of the coverage manifest.
        if dest.exists() and not args.force:
            skipped += 1
            covered.append(frame_idx)
            continue

        nearest = _nearest(available, frame_idx)
        offset = abs(nearest - frame_idx)
        if offset > args.max_offset:
            unresolved.append((offset, drive_id, frame_idx, nearest))
            continue

        if dest.exists():
            dest.unlink()
        os.link(args.source_root / drive_id / f"{nearest:09d}.bin", dest)
        linked += 1
        covered.append(frame_idx)
        hist[offset] += 1

    (dest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "n_frames": len(covered),
                "frame_indices": covered,
                # These labels are NOT from the frame they are named after: each is the nearest
                # available label, up to `max_offset` frames away. Recorded so nothing downstream
                # mistakes this root for exactly-labelled data.
                "frame_selection": "nearest-linked",
                "nearest_link_source": str(args.source_root),
                "max_offset": args.max_offset,
                "offset_histogram": {str(k): v for k, v in sorted(hist.items())},
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    return linked, skipped, hist, unresolved


def cmd_link(args: argparse.Namespace) -> int:
    """Hardlink each required frame to the nearest already-labelled frame in a source root.

    Cheaper than relabelling by two orders of magnitude (~7 min of filesystem work vs ~18 h of
    GPU), because the labels are coarse enough to be near-invariant over the <=5-frame gap. See the
    `nearest-timestep` note in this module's docstring for the measured error.
    """
    payload = json.loads(args.required.read_text())
    drives: dict[str, list[int]] = payload["drives"]

    linked = skipped = 0
    offset_hist: Counter[int] = Counter()
    worst: list[tuple[int, str, int, int]] = []

    for i, (drive_id, frames) in enumerate(sorted(drives.items())):
        n_linked, n_skipped, hist, unresolved_frames = _link_drive(
            args, drive_id, frames
        )
        linked += n_linked
        skipped += n_skipped
        offset_hist += hist
        worst.extend(unresolved_frames)
        if (i + 1) % 100 == 0:
            print(
                f"  ...{i + 1}/{len(drives)} drives, {linked} linked, {len(worst)} unresolved",
                file=sys.stderr,
            )

    unresolved = len(worst)
    total = linked + skipped + unresolved
    print(f"source     : {args.source_root}")
    print(f"dest       : {args.dest_root}")
    print(f"linked     : {linked}")
    print(f"skipped    : {skipped} (already present; --force to relink)")
    print(
        f"unresolved : {unresolved} (nearest label further than --max-offset={args.max_offset})"
    )
    print(f"total      : {total}")
    if offset_hist:
        n = sum(offset_hist.values())
        mean = sum(k * v for k, v in offset_hist.items()) / n
        print(f"\noffset histogram (frames): {dict(sorted(offset_hist.items()))}")
        print(f"mean |offset| = {mean:.2f} frames = {1000 * mean / 30:.0f} ms at 30 Hz")
    if worst:
        shown = worst[:MAX_EXAMPLES]
        print(f"\nfirst {len(shown)} of {len(worst)} unresolved:")
        for offset, drive_id, frame_idx, nearest in shown:
            print(
                f"  {drive_id}: frame {frame_idx} -> nearest {nearest} (offset {offset})"
            )
        if args.unresolved_out is not None:
            residue: dict[str, list[int]] = {}
            for _offset, drive_id, frame_idx, _nearest in worst:
                residue.setdefault(drive_id, []).append(frame_idx)
            args.unresolved_out.parent.mkdir(parents=True, exist_ok=True)
            args.unresolved_out.write_text(
                json.dumps({
                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": "residue of `link`; feed to lfg_label_drives.py --frames-from",
                    "n_drives": len(residue),
                    "n_frames": sum(len(v) for v in residue.values()),
                    "drives": {k: sorted(v) for k, v in sorted(residue.items())},
                })
            )
            print(
                f"\nwrote {args.unresolved_out} ({len(residue)} drives, {len(worst)} frames)"
            )
            print(
                "Label those exactly with lfg_label_drives.py --frames-from into --dest-root,"
            )
            print("then re-run `link` to fold them into the coverage manifests.")
        else:
            print(
                "\nThose frames still need real labelling; re-run with --unresolved-out to"
            )
            print("emit them as a --frames-from JSON.")
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    payload = json.loads(args.required.read_text())
    drives: dict[str, list[int]] = payload["drives"]

    total_required = total_missing = 0
    bad: list[tuple[str, int, int, list[int]]] = []

    for drive_id, frames in drives.items():
        required = set(frames)
        manifest_path = args.label_root / drive_id / "manifest.json"
        if manifest_path.is_file():
            labelled = set(json.loads(manifest_path.read_text())["frame_indices"])
        else:
            labelled = set()
        missing = required - labelled
        total_required += len(required)
        total_missing += len(missing)
        if missing:
            bad.append((drive_id, len(required), len(missing), sorted(missing)[:3]))

    pct = 100 * total_missing / total_required if total_required else 0.0
    print(f"label root : {args.label_root}")
    print(f"drives     : {len(drives)}  ({len(bad)} with missing labels)")
    print(f"required   : {total_required}")
    print(f"missing    : {total_missing} ({pct:.2f}%)")
    if bad:
        print(f"\nfirst {min(len(bad), args.show)} drives with gaps:")
        for drive_id, nreq, nmiss, examples in bad[: args.show]:
            print(f"  {drive_id}: {nmiss}/{nreq} missing, e.g. {examples}")
        print("\nFAIL: labels are incomplete; training will die in the dataloader.")
        return 1
    print("\nOK: every required frame has a label.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser(
        "extract", help="derive required frames from built samples folders"
    )
    p_extract.add_argument(
        "--run-folder",
        type=Path,
        action="append",
        required=True,
        dest="run_folders",
        help="a built rbyte samples run folder, e.g. <cache>/yaak/train/clip37/samples",
    )
    p_extract.add_argument("--output", type=Path, required=True)
    p_extract.set_defaults(func=cmd_extract)

    p_link = sub.add_parser(
        "link", help="hardlink required frames to the nearest already-labelled frame"
    )
    p_link.add_argument("--required", type=Path, required=True)
    p_link.add_argument(
        "--source-root", type=Path, required=True, help="e.g. .../lfg_labels/v1"
    )
    p_link.add_argument(
        "--dest-root", type=Path, required=True, help="e.g. .../lfg_labels/v1_nearest"
    )
    p_link.add_argument(
        "--max-offset",
        type=int,
        default=5,
        help="refuse to link a label more than N frames from the requested one (default 5 = 167ms)",
    )
    p_link.add_argument(
        "--force", action="store_true", help="relink even if the dest exists"
    )
    p_link.add_argument(
        "--unresolved-out",
        type=Path,
        default=None,
        help="write frames exceeding --max-offset as a --frames-from JSON for exact labelling",
    )
    p_link.set_defaults(func=cmd_link)

    p_verify = sub.add_parser(
        "verify", help="check a label root covers every required frame"
    )
    p_verify.add_argument("--required", type=Path, required=True)
    p_verify.add_argument("--label-root", type=Path, required=True)
    p_verify.add_argument("--show", type=int, default=10)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
