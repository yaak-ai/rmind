#!/usr/bin/env python
"""Offline-label yaak training frames with the released LFG model.

Implements Stage 1 of `lfg_aux_supervision_task.md`: for each sampled frame of a drive, run LFG in
current-frames-only mode (`n_future_frames_override=0`) over consecutive 15-frame windows, pool the
segmentation/motion/confidence outputs onto the trunk's 16x16 patch grid, and write one packed
1024-byte `.bin` per frame plus a per-drive `manifest.json`.

Which frames to label
---------------------
**Use `--frames-from`.** The task brief's SS1.3 claim that the training pipeline only ever loads
`frame_idx % 10 == 0` is wrong: `gather_every` strides over *surviving rows* after the `filtered`
DuckDB query, so each drive's frames are stride-10 at an arbitrary per-drive phase. Labelling the
`% 10 == 0` phase covered only 5.6% of what the 655-drive train set actually requests, and the
first full-corpus run died in the dataloader on a missing `.bin`. Generate the exact per-drive
frame lists with `scripts/lfg_required_frames.py extract` and pass the JSON here.

`--stride` remains as an escape hatch for exploratory labelling, but a label root built that way
must not be used for training. The manifest records which mode produced it.

Run with the LFG venv's interpreter (kept out of the rmind venv, see task brief SS2), e.g.:

    /nasa/tools/lfg/.venv/bin/python scripts/lfg_label_drives.py \\
        --lfg-repo /nasa/tools/lfg \\
        --checkpoint /nasa/tools/lfg/lfg_seg_motion_m3n3.pt \\
        --data-root /nasa/drives/yaak/data \\
        --output-root /nasa/drives/yaak/lfg_labels/v2 \\
        --frames-from /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \\
        --skip-existing \\
        --drive Niro096-HQ/2023-01-11--13-47-36

`--skip-existing` makes a rerun resume rather than redo, which matters for a ~18 h full-corpus job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Sequence

FRAME_SUBDIR = "frames/cam_front_left.pii.mp4/576x324"
FRAME_STRIDE = 10  # `--stride` fallback only; NOT what the training pipeline requests (see module docstring)
WINDOW_SIZE = 15  # MAX_TOTAL_FRAMES for this LFG checkpoint
CROP_SIZE = (320, 576)  # (h, w), identical to config/model/yaak/patch_policy/raw.yaml
LFG_RESOLUTION = (294, 518)  # (h, w), see task brief SS1.4
GRID_SIZE = (16, 16)
LABEL_SHAPE = (4, GRID_SIZE[0], GRID_SIZE[1])
LABEL_NBYTES = int(np.prod(LABEL_SHAPE))


def _git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_crop(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Center-crop `(..., H, W)` to `size`, matching torchvision.transforms.v2.CenterCrop."""
    th, tw = size
    h, w = x.shape[-2], x.shape[-1]
    top = round((h - th) / 2.0)
    left = round((w - tw) / 2.0)
    return x[..., top : top + th, left : left + tw]


def load_frame(path: Path) -> torch.Tensor:
    """JPEG bytes -> `(3, 294, 518)` float tensor in [0, 1], LFG resolution.

    Preprocessing must match the training pipeline byte-for-byte up to and including the crop
    (task brief SS3.2); the resize below is otherwise unconstrained since the label is pooled
    back onto the patch grid.
    """
    rgb = simplejpeg.decode_jpeg(
        path.read_bytes(), colorspace="rgb", fastdct=True, fastupsample=True
    )  # (324, 576, 3) uint8
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, 324, 576)
    x = center_crop(x, CROP_SIZE)  # (3, 320, 576)
    return F.interpolate(
        x[None], size=LFG_RESOLUTION, mode="bicubic", align_corners=False
    ).clamp_(0, 1)[0]  # (3, 294, 518)


def discover_targets_by_stride(data_root: Path, drive_id: str) -> list[Path]:
    """Exploratory mode: every FRAME_STRIDE-th raw JPEG. Not training-aligned -- see docstring."""
    frame_dir = data_root / drive_id / FRAME_SUBDIR
    frames = sorted(frame_dir.glob("*.jpg"))
    return [f for f in frames if int(f.stem) % FRAME_STRIDE == 0]


def targets_from_frame_indices(
    data_root: Path, drive_id: str, frame_indices: Sequence[int]
) -> list[Path]:
    """Resolve an explicit frame_idx list to JPEG paths, failing loudly on any absent frame.

    A missing JPEG here means the label root can never satisfy the training pipeline, so it is
    better to fail this drive now than to discover it from a dataloader worker mid-run.

    Raises:
        FileNotFoundError: if any required frame's JPEG is absent.
    """
    frame_dir = data_root / drive_id / FRAME_SUBDIR
    paths: list[Path] = []
    missing: list[int] = []
    for frame_idx in sorted(frame_indices):
        path = frame_dir / f"{frame_idx:09d}.jpg"
        if path.is_file():
            paths.append(path)
        else:
            missing.append(frame_idx)
    if missing:
        msg = (
            f"{len(missing)} of {len(frame_indices)} required JPEGs absent for {drive_id}, "
            f"e.g. {missing[:5]}"
        )
        raise FileNotFoundError(msg)
    return paths


def label_is_complete(out_dir: Path, frame_idx: int) -> bool:
    """True if this frame already has a full-size label blob (fixed size makes truncation visible)."""
    path = out_dir / f"{frame_idx:09d}.bin"
    try:
        return path.stat().st_size == LABEL_NBYTES
    except OSError:
        return False


def chunk_windows(targets: list[Path], window_size: int) -> list[list[Path]]:
    windows = []
    for start in range(0, len(targets), window_size):
        window = targets[start : start + window_size]
        if len(window) < window_size:
            window += [window[-1]] * (window_size - len(window))
        windows.append(window)
    return windows


@torch.inference_mode()
def label_window(model, paths: list[Path], device: torch.device) -> torch.Tensor:
    """Run LFG on one 15-frame window; return `(15, 4, 16, 16)` uint8 packed labels."""
    frames = torch.stack([load_frame(p) for p in paths])  # (15, 3, 294, 518)
    imgs = frames[None].to(device)  # (1, 15, 3, 294, 518)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(imgs, n_future_frames_override=0)

    seg = out["segmentation"][0].permute(0, 3, 1, 2).float()  # (15, 7, 294, 518)
    mot = out["motion"][0].permute(0, 3, 1, 2).float()  # (15, 1, 294, 518)
    conf = out["conf"][0].permute(0, 3, 1, 2).float()  # (15, 1, 294, 518)

    p = F.adaptive_avg_pool2d(seg.softmax(dim=1), GRID_SIZE)  # (15, 7, 16, 16)
    seg_label = p.argmax(dim=1).to(torch.uint8)
    seg_purity = (p.max(dim=1).values * 255).round().to(torch.uint8)
    motion = (
        (F.adaptive_avg_pool2d(mot.sigmoid(), GRID_SIZE)[:, 0] * 255)
        .round()
        .to(torch.uint8)
    )
    confidence = (
        (F.adaptive_avg_pool2d(conf.sigmoid(), GRID_SIZE)[:, 0] * 255)
        .round()
        .to(torch.uint8)
    )

    return torch.stack(
        [seg_label, seg_purity, motion, confidence], dim=1
    ).cpu()  # (15, 4, 16, 16)


def label_drive(
    model,
    device: torch.device,
    data_root: Path,
    output_root: Path,
    drive_id: str,
    lfg_sha256: str,
    script_git_sha: str,
    *,
    frame_indices_required: Sequence[int] | None = None,
    limit_windows: int | None = None,
    skip_existing: bool = False,
) -> dict:
    if frame_indices_required is None:
        selection = f"stride-{FRAME_STRIDE}"
        targets = discover_targets_by_stride(data_root, drive_id)
        if not targets:
            msg = f"no frame_idx % {FRAME_STRIDE} == 0 frames found for {drive_id}"
            raise FileNotFoundError(msg)
    else:
        selection = "required-set"
        targets = targets_from_frame_indices(
            data_root, drive_id, frame_indices_required
        )
        if not targets:
            msg_0 = f"empty required frame list for {drive_id}"
            raise ValueError(msg_0)

    out_dir = output_root / drive_id
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = targets
    n_skipped = 0
    if skip_existing:
        todo = [p for p in targets if not label_is_complete(out_dir, int(p.stem))]
        n_skipped = len(targets) - len(todo)

    windows = chunk_windows(todo, WINDOW_SIZE)
    if limit_windows is not None:
        windows = windows[:limit_windows]
        todo = todo[: limit_windows * WINDOW_SIZE]

    n_written = 0
    t0 = time.monotonic()
    for i, window_paths in enumerate(windows):
        n_real = min(WINDOW_SIZE, len(todo) - i * WINDOW_SIZE)
        packed = label_window(model, window_paths, device)  # (15, 4, 16, 16)
        for slot in range(n_real):
            frame_idx = int(window_paths[slot].stem)
            blob = packed[slot].contiguous().numpy().tobytes()
            assert len(blob) == LABEL_NBYTES, (len(blob), LABEL_NBYTES)
            (out_dir / f"{frame_idx:09d}.bin").write_bytes(blob)
            n_written += 1
        elapsed = time.monotonic() - t0
        print(
            f"[{drive_id}] window {i + 1}/{len(windows)} "
            f"({n_written}/{len(todo)} frames, {elapsed:.1f}s elapsed, "
            f"{elapsed / (i + 1):.2f}s/window)",
            file=sys.stderr,
        )

    # `lfg_required_frames.py verify` reads the manifest as this drive's coverage record, so derive
    # it from what is actually on disk rather than from what this invocation intended. That keeps it
    # honest across every combination: a full run lists everything, a --skip-existing rerun includes
    # what an earlier run wrote, and a --limit-windows run lists only the prefix it got to (so
    # `verify` correctly fails instead of passing against a partial label root).
    frame_indices = [
        frame_idx
        for frame_idx in (int(p.stem) for p in targets)
        if label_is_complete(out_dir, frame_idx)
    ]

    manifest = {
        "n_frames": len(frame_indices),
        "frame_indices": frame_indices,
        "frame_selection": selection,
        "n_written_this_run": n_written,
        "n_skipped_existing": n_skipped,
        "lfg_sha256": lfg_sha256,
        "script_git_sha": script_git_sha,
        "crop": list(CROP_SIZE),
        "lfg_resolution": list(LFG_RESOLUTION),
        "grid": list(GRID_SIZE),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lfg-repo", type=Path, required=True, help="path to the cloned LFG repo"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/nasa/drives/yaak/data")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("/nasa/drives/yaak/lfg_labels/v1")
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        help="drive id; repeatable. defaults to every drive in --frames-from",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--frames-from",
        type=Path,
        default=None,
        help=(
            "JSON from `scripts/lfg_required_frames.py extract`; labels exactly the frames the "
            "training pipeline requests. Omit only for exploratory stride-based labelling."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip frames that already have a full-size .bin, so a rerun resumes",
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="only label the first N windows per drive (smoke-test knob, not in the brief)",
    )
    args = parser.parse_args()

    required: dict[str, list[int]] | None = None
    if args.frames_from is not None:
        payload = json.loads(args.frames_from.read_text())
        required = payload["drives"]
        print(
            f"required frames from {args.frames_from}: "
            f"{payload['n_drives']} drives, {payload['n_frames']} frames",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: no --frames-from; falling back to frame_idx % {FRAME_STRIDE} == 0, which "
            "does NOT match what the training pipeline requests. Do not train on this label root.",
            file=sys.stderr,
        )

    if args.drives is None:
        if required is None:
            parser.error("--drive is required when --frames-from is not given")
        args.drives = sorted(required)

    if required is not None:
        unknown = [d for d in args.drives if d not in required]
        if unknown:
            parser.error(
                f"{len(unknown)} requested drive(s) absent from {args.frames_from}: {unknown[:3]}"
            )

    sys.path.insert(0, str(args.lfg_repo))
    from lfg.checkpoint import load_model_from_checkpoint

    device = torch.device(args.device)
    print(f"loading checkpoint {args.checkpoint} ...", file=sys.stderr)
    model, model_config, report, _metadata = load_model_from_checkpoint(
        args.checkpoint, device=device
    )
    print(f"model_config={model_config}", file=sys.stderr)
    print(f"load report: {report.to_dict()}", file=sys.stderr)

    lfg_sha256 = _sha256_file(args.checkpoint)
    script_git_sha = _git_sha(Path(__file__).resolve())
    print(f"lfg_sha256={lfg_sha256} script_git_sha={script_git_sha}", file=sys.stderr)

    failures: dict[str, str] = {}
    for i, drive_id in enumerate(args.drives):
        try:
            manifest = label_drive(
                model,
                device,
                args.data_root,
                args.output_root,
                drive_id,
                lfg_sha256,
                script_git_sha,
                frame_indices_required=None if required is None else required[drive_id],
                limit_windows=args.limit_windows,
                skip_existing=args.skip_existing,
            )
            print(
                f"[{drive_id}] {manifest['n_written_this_run']} written, "
                f"{manifest['n_skipped_existing']} already present, "
                f"{manifest['n_frames']} total "
                f"({i + 1}/{len(args.drives)} drives)",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[{drive_id}] FAILED: {exc!r}", file=sys.stderr)
            failures[drive_id] = repr(exc)

    if failures:
        failures_path = args.output_root / "failures.json"
        existing = {}
        if failures_path.exists():
            existing = json.loads(failures_path.read_text())
        existing.update(failures)
        failures_path.write_text(json.dumps(existing, indent=2))
        print(
            f"{len(failures)}/{len(args.drives)} drives failed; see {failures_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
