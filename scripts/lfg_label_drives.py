#!/usr/bin/env python
"""Offline-label yaak training frames with the released LFG model.

Implements Stage 1 of `lfg_aux_supervision_task.md`: for each sampled frame of a drive, run LFG in
current-frames-only mode (`n_future_frames_override=0`) over 15-frame windows, pool the
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

Windows are context, not just batching
--------------------------------------
LFG's `decode()` alternates per-frame attention with attention across *all* frames of the window
(`Pi3/pi3/models/pi3.py`, `i % 2` -> `hidden.reshape(B, N*hw, -1)`), so **every frame's label depends
on the other 14 frames in its window**. Two consequences:

- Which frames are labelled together matters. `--gap-split` keeps windows temporally coherent by
  breaking them at `frame_idx` discontinuities larger than N frames (2.3% of required-set windows
  otherwise span a gap of >20 frames, median 850, max 42634). Splitting at *every* discontinuity is
  the wrong trade: the required set is full of harmless 1-9 frame phase shifts and per-run chunking
  costs +23.8% windows.
- Resume must be **drive-level** (`--resume-drives`), not frame-level. `--skip-existing` drops
  individual frames from the target list, which changes the composition of every window after the
  skip point, so a resumed run yields labels computed under different context than a clean run.

Throughput
----------
`--loader-threads` (default 4) moves JPEG decode/resize and the `.bin` writes off the critical path,
leaving the job GPU-bound; `--lean-forward` (default) skips the point/camera decoders and heads,
whose outputs this script never reads. Measured on one RTX 5090: 450 ms/window for the serial
full-forward reference path, 271-293 ms/window with both (compute floor 263 ms). The two are
**bit-identical** -- they are throughput changes, not a different label definition. Pass
`--loader-threads 0 --full-forward` for the reference path.

Run with the LFG venv's interpreter (kept out of the rmind venv, see task brief SS2), e.g.:

    /nasa/tools/lfg/.venv/bin/python scripts/lfg_label_drives.py \\
        --lfg-repo /nasa/tools/lfg \\
        --checkpoint /nasa/tools/lfg/lfg_seg_motion_m3n3.pt \\
        --data-root /nasa/drives/yaak/data \\
        --output-root /nasa/drives/yaak/lfg_labels/v2 \\
        --frames-from /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \\
        --resume-drives
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import simplejpeg
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

FRAME_SUBDIR = "frames/cam_front_left.pii.mp4/576x324"
FRAME_STRIDE = 10  # `--stride` fallback only; NOT what the training pipeline requests (see module docstring)
WINDOW_SIZE = 15  # MAX_TOTAL_FRAMES for this LFG checkpoint
CROP_SIZE = (320, 576)  # (h, w), identical to config/model/yaak/patch_policy/raw.yaml
LFG_RESOLUTION = (294, 518)  # (h, w), see task brief SS1.4
GRID_SIZE = (16, 16)
LABEL_SHAPE = (4, GRID_SIZE[0], GRID_SIZE[1])
LABEL_NBYTES = int(np.prod(LABEL_SHAPE))
GAP_SPLIT_DEFAULT = 20  # frames; break windows at larger frame_idx discontinuities
LOADER_THREADS_DEFAULT = 4  # the slowest NFS drives read at ~560 ms/window serially
QUEUE_DEPTH = 3  # windows of read-ahead; 4 KiB of labels per window, so depth is cheap


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


def chunk_windows(
    targets: list[Path], window_size: int = WINDOW_SIZE, gap_split: int | None = None
) -> list[list[Path]]:
    """Group targets into windows of at most `window_size`, unpadded.

    With `gap_split`, a window never spans a `frame_idx` discontinuity larger than that many
    frames, so the model is never handed a temporally incoherent sequence (see module docstring).
    """
    if not targets:
        return []
    segments: list[list[Path]] = [targets]
    if gap_split is not None:
        segments = []
        current = [targets[0]]
        for prev, nxt in zip(targets, targets[1:], strict=False):
            if int(nxt.stem) - int(prev.stem) > gap_split:
                segments.append(current)
                current = [nxt]
            else:
                current.append(nxt)
        segments.append(current)
    return [
        segment[start : start + window_size]
        for segment in segments
        for start in range(0, len(segment), window_size)
    ]


def pad_window(window: list[Path], window_size: int) -> list[Path]:
    """Tail-pad a short window by repeating its last frame (the checkpoint wants a fixed length)."""
    return window + [window[-1]] * (window_size - len(window))


def forward_full(model, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Reference path: the model's own forward, discarding everything but the three heads used."""
    out = model(imgs, n_future_frames_override=0)
    return out["segmentation"], out["motion"], out["conf"]


def forward_labels_only(model, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`forward_full` minus the point/camera decoders and heads, whose outputs are never read.

    Verified bit-identical to `forward_full` on all three returned tensors; worth ~13% of the
    per-window GPU time. Mirrors `LFG.forward` up to that omission -- re-check it against
    `Pi3/pi3/models/pi3.py` if the LFG checkout is ever updated.
    """
    x = (imgs - model.image_mean) / model.image_std
    b, n, c, h, w = x.shape
    flat = x.reshape(b * n, c, h, w)
    hidden = (
        model.encoder.forward_features(flat)
        if hasattr(model.encoder, "forward_features")
        else model.encoder(flat, is_training=True)
    )
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]
    hidden, pos = model.decode(hidden, n, h, w)
    all_hidden, all_pos = model.autoregressive_transformer(
        hidden, n, pos, n_future_frames_override=0
    )
    conf_hidden = model.conf_decoder(all_hidden, xpos=all_pos)
    seg_hidden = model.segmentation_decoder(all_hidden, xpos=all_pos)
    motion_hidden = model.motion_decoder(all_hidden, xpos=all_pos)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        start = model.patch_start_idx
        seg = model.segmentation_head(
            [seg_hidden.float()[:, start:]], (h, w)
        ).reshape(b, n, h, w, -1)
        motion = model.motion_head([motion_hidden.float()[:, start:]], (h, w)).reshape(
            b, n, h, w, -1
        )
        conf = model.conf_head([conf_hidden.float()[:, start:]], (h, w)).reshape(
            b, n, h, w, -1
        )
    return seg, motion, conf


def pack_labels(
    seg: torch.Tensor, motion: torch.Tensor, conf: torch.Tensor
) -> torch.Tensor:
    """Pool the three heads onto the 16x16 patch grid and pack to `(n, 4, 16, 16)` uint8 on CPU."""
    seg = seg[0].permute(0, 3, 1, 2).float()  # (n, 7, 294, 518)
    motion = motion[0].permute(0, 3, 1, 2).float()  # (n, 1, 294, 518)
    conf = conf[0].permute(0, 3, 1, 2).float()  # (n, 1, 294, 518)

    p = F.adaptive_avg_pool2d(seg.softmax(dim=1), GRID_SIZE)  # (n, 7, 16, 16)
    seg_label = p.argmax(dim=1).to(torch.uint8)
    seg_purity = (p.max(dim=1).values * 255).round().to(torch.uint8)
    motion_label = (
        (F.adaptive_avg_pool2d(motion.sigmoid(), GRID_SIZE)[:, 0] * 255)
        .round()
        .to(torch.uint8)
    )
    confidence = (
        (F.adaptive_avg_pool2d(conf.sigmoid(), GRID_SIZE)[:, 0] * 255)
        .round()
        .to(torch.uint8)
    )
    return torch.stack(
        [seg_label, seg_purity, motion_label, confidence], dim=1
    ).cpu()  # (n, 4, 16, 16)


@torch.inference_mode()
def label_frames(
    model, frames: torch.Tensor, device: torch.device, *, lean: bool = True
) -> torch.Tensor:
    """Run LFG on one already-loaded window `(n, 3, 294, 518)`; return `(n, 4, 16, 16)` uint8."""
    imgs = frames[None].to(device, non_blocking=True)  # (1, n, 3, 294, 518)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        seg, motion, conf = (forward_labels_only if lean else forward_full)(model, imgs)
    return pack_labels(seg, motion, conf)


def load_window(paths: list[Path], window_size: int, *, pin: bool) -> torch.Tensor:
    """Load and tail-pad one window to `(window_size, 3, 294, 518)`."""
    frames = torch.stack([load_frame(p) for p in pad_window(paths, window_size)])
    return frames.pin_memory() if pin else frames


def iter_loaded_windows(
    windows: list[list[Path]], window_size: int, threads: int, *, pin: bool
) -> Iterator[tuple[list[Path], torch.Tensor]]:
    """Yield `(real_paths, padded_frames)` in order, reading ahead on `threads` workers.

    Ordered with a bounded look-ahead deque rather than `Executor.map` so at most QUEUE_DEPTH
    windows (~50 MB of float frames) are resident regardless of how long the drive is.
    """
    if threads <= 0:
        for window in windows:
            yield window, load_window(window, window_size, pin=pin)
        return

    with ThreadPoolExecutor(max_workers=threads) as pool:
        remaining = iter(windows)
        pending: deque = deque()

        def submit_next() -> bool:
            window = next(remaining, None)
            if window is None:
                return False
            pending.append(
                (window, pool.submit(load_window, window, window_size, pin=pin))
            )
            return True

        for _ in range(threads + QUEUE_DEPTH):
            if not submit_next():
                break
        while pending:
            window, future = pending.popleft()
            frames = future.result()
            submit_next()
            yield window, frames


class LabelWriter:
    """Background writer for 1 KiB label blobs, so NFS latency never blocks the GPU loop."""

    def __init__(self, out_dir: Path, depth: int = 8) -> None:
        self._out_dir = out_dir
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                frame_indices, packed = item
                for slot, frame_idx in enumerate(frame_indices):
                    blob = packed[slot].contiguous().numpy().tobytes()
                    assert len(blob) == LABEL_NBYTES, (len(blob), LABEL_NBYTES)
                    (self._out_dir / f"{frame_idx:09d}.bin").write_bytes(blob)
            except BaseException as exc:  # surfaced by put()/close(), never swallowed
                self._error = exc
                return

    def _check(self) -> None:
        if self._error is not None:
            raise self._error

    def put(self, frame_indices: list[int], packed: torch.Tensor) -> None:
        self._check()
        while True:
            try:
                self._queue.put((frame_indices, packed), timeout=1.0)
                return
            except queue.Full:
                # the writer thread died holding the queue full; report that, not a hang
                self._check()

    def close(self) -> None:
        self._check()
        self._queue.put(None)
        self._thread.join()
        self._check()


def drive_is_complete(
    out_dir: Path, frame_indices_required: Sequence[int], params: dict
) -> bool:
    """True if a previous run already covered this drive's required frames under `params`.

    Drive-level resume granularity is deliberate (see module docstring): skipping *frames* would
    change window composition, so a drive is either fully reusable or relabelled from scratch.
    A manifest written under different windowing/forward settings is not reusable.
    """
    manifest_path = out_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if any(manifest.get(key) != value for key, value in params.items()):
        return False
    return set(frame_indices_required) <= set(manifest.get("frame_indices", ()))


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
    resume_drives: bool = False,
    lean: bool = True,
    gap_split: int | None = GAP_SPLIT_DEFAULT,
    loader_threads: int = LOADER_THREADS_DEFAULT,
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
    params = {
        "frame_selection": selection,
        "window_size": WINDOW_SIZE,
        "gap_split": gap_split,
        "forward": "labels-only" if lean else "full",
    }
    if (
        resume_drives
        and frame_indices_required is not None
        and drive_is_complete(out_dir, frame_indices_required, params)
    ):
        manifest = json.loads((out_dir / "manifest.json").read_text())
        manifest["n_written_this_run"] = 0
        manifest["resumed"] = True
        return manifest

    out_dir.mkdir(parents=True, exist_ok=True)

    todo = targets
    n_skipped = 0
    if skip_existing:
        todo = [p for p in targets if not label_is_complete(out_dir, int(p.stem))]
        n_skipped = len(targets) - len(todo)

    windows = chunk_windows(todo, WINDOW_SIZE, gap_split)
    if limit_windows is not None:
        windows = windows[:limit_windows]

    n_written = 0
    n_todo = sum(len(w) for w in windows)
    writer = LabelWriter(out_dir)
    t0 = time.monotonic()
    try:
        for i, (window, frames) in enumerate(
            iter_loaded_windows(
                windows, WINDOW_SIZE, loader_threads, pin=device.type == "cuda"
            )
        ):
            packed = label_frames(model, frames, device, lean=lean)  # (15, 4, 16, 16)
            writer.put([int(p.stem) for p in window], packed)
            n_written += len(window)
            elapsed = time.monotonic() - t0
            print(
                f"[{drive_id}] window {i + 1}/{len(windows)} "
                f"({n_written}/{n_todo} frames, {elapsed:.1f}s elapsed, "
                f"{elapsed / (i + 1):.2f}s/window)",
                file=sys.stderr,
            )
    finally:
        writer.close()

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
        "n_written_this_run": n_written,
        "n_skipped_existing": n_skipped,
        "lfg_sha256": lfg_sha256,
        "script_git_sha": script_git_sha,
        "crop": list(CROP_SIZE),
        "lfg_resolution": list(LFG_RESOLUTION),
        "grid": list(GRID_SIZE),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **params,
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
    parser.add_argument(
        "--drives-from",
        type=Path,
        default=None,
        help="file of drive ids, one per line (comments with # allowed); union'd with --drive",
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
        "--resume-drives",
        action="store_true",
        help=(
            "skip drives already fully covered by a manifest written with the same windowing and "
            "forward settings; relabel any incomplete drive from scratch. Preferred over "
            "--skip-existing, which perturbs window composition"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "skip individual frames that already have a full-size .bin. NOTE: this changes which "
            "frames share a window and therefore the labels themselves -- prefer --resume-drives"
        ),
    )
    parser.add_argument(
        "--gap-split",
        type=int,
        default=GAP_SPLIT_DEFAULT,
        help=(
            "break windows at frame_idx discontinuities larger than this many frames "
            f"(default {GAP_SPLIT_DEFAULT}); 0 disables, reproducing the flat chunking of v1"
        ),
    )
    parser.add_argument(
        "--loader-threads",
        type=int,
        default=LOADER_THREADS_DEFAULT,
        help=(
            f"JPEG loader threads (default {LOADER_THREADS_DEFAULT}); 0 loads serially on the "
            "main thread, i.e. the reference path"
        ),
    )
    parser.add_argument(
        "--full-forward",
        dest="lean",
        action="store_false",
        help="use the model's own forward instead of the labels-only one (bit-identical, ~13% slower)",
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

    drives = list(args.drives or [])
    if args.drives_from is not None:
        drives += [
            line.strip()
            for line in args.drives_from.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not drives:
        if required is None:
            parser.error("--drive is required when --frames-from is not given")
        drives = sorted(required)
    else:
        drives = sorted(dict.fromkeys(drives))

    if required is not None:
        unknown = [d for d in drives if d not in required]
        if unknown:
            parser.error(
                f"{len(unknown)} requested drive(s) absent from {args.frames_from}: {unknown[:3]}"
            )

    if args.skip_existing and args.resume_drives:
        parser.error("--skip-existing and --resume-drives are mutually exclusive")

    gap_split = args.gap_split if args.gap_split > 0 else None

    sys.path.insert(0, str(args.lfg_repo))
    from lfg.checkpoint import load_model_from_checkpoint

    device = torch.device(args.device)
    print(f"loading checkpoint {args.checkpoint} ...", file=sys.stderr)
    model, model_config, report, _metadata = load_model_from_checkpoint(
        args.checkpoint, device=device
    )
    model.eval()
    print(f"model_config={model_config}", file=sys.stderr)
    print(f"load report: {report.to_dict()}", file=sys.stderr)

    lfg_sha256 = _sha256_file(args.checkpoint)
    script_git_sha = _git_sha(Path(__file__).resolve())
    print(
        f"lfg_sha256={lfg_sha256} script_git_sha={script_git_sha} "
        f"forward={'labels-only' if args.lean else 'full'} gap_split={gap_split} "
        f"loader_threads={args.loader_threads}",
        file=sys.stderr,
    )

    failures: dict[str, str] = {}
    t_start = time.monotonic()
    n_frames_total = 0
    for i, drive_id in enumerate(drives):
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
                resume_drives=args.resume_drives,
                lean=args.lean,
                gap_split=gap_split,
                loader_threads=args.loader_threads,
            )
            n_frames_total += manifest["n_written_this_run"]
            elapsed = time.monotonic() - t_start
            rate = n_frames_total / elapsed * 3600 if n_frames_total else 0.0
            print(
                f"[{drive_id}] {manifest['n_written_this_run']} written, "
                f"{manifest.get('n_skipped_existing', 0)} already present, "
                f"{manifest['n_frames']} total "
                f"({i + 1}/{len(drives)} drives, {elapsed / 3600:.2f}h elapsed, "
                f"{rate / 1000:.1f}k frames/h)"
                + (" [resumed]" if manifest.get("resumed") else ""),
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
            f"{len(failures)}/{len(drives)} drives failed; see {failures_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
