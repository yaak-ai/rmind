"""Step 2 converter — re-extract mp4 -> jpg with cv2.INTER_CUBIC (see
jpg_preprocessing_parity_task.md §5).

Legacy training jpgs were produced offline by ffmpeg's `scale_npp` (NPPI_INTER_CUBIC),
a CUDA kernel with no Python equivalent (see scripts/compare_frame_pixels.py, step 1).
This converter re-extracts frames using the *exact* decode + resize
`_VideoFrameSource`/`_preprocess_image` (src/rmind/scripts/benchmark_onnx.py) already use
at inference/benchmark time — `cv2.VideoCapture` sequential read, `cv2.resize(...,
INTER_CUBIC)` — so training and the benchmark's video path share one decode+resize
pipeline instead of two independently-approximate ones.

Output layout mirrors the source drive directory exactly (frames/ re-extracted, every
other entry symlinked), so `<dst-root>/<drive>` is a drop-in replacement for
`paths.data`/`data_dir` — no template or benchmark_onnx.py changes required.

Usage:
    uv run --group benchmark python scripts/convert_frames_jpg.py \\
        --src-root /nasa/drives/yaak/data --dst-root /nasa/alex/converted_jpg \\
        --interpolation cubic --quality 95 --sampling 444 \\
        --workers 4 --verify-sample 20 --resume \\
        --drive Niro122-HQ/2023-05-25--09-34-14
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_NATIVE_SIZE = (324, 576)  # (H, W) — matches benchmark_onnx.py's DEFAULT_IMAGE_SIZE
_FRAMES_RELPATH = Path("frames/cam_front_left.pii.mp4/576x324")
_VIDEO_NAME = "cam_front_left.pii.mp4"

_INTERPOLATIONS = {
    "cubic": "INTER_CUBIC",
    "area": "INTER_AREA",
    "lanczos4": "INTER_LANCZOS4",
    "linear": "INTER_LINEAR",
    "linear_exact": "INTER_LINEAR_EXACT",
}
_SAMPLING_FACTORS = {
    "444": "IMWRITE_JPEG_SAMPLING_FACTOR_444",
    "420": "IMWRITE_JPEG_SAMPLING_FACTOR_420",
}
_DONE_MARKER = ".conversion_complete"

# Some source HEVC streams contain a genuine bitstream defect (observed: a
# duplicate-POC / undecodable NALU near the tail of one drive) that both cv2's
# and the system ffmpeg CLI's software HEVC decoder drop, one frame short of
# ffprobe's container-level nb_frames. Since two independent decoders agree,
# this is a source-data property, not a converter bug — tolerate a small,
# explained drift instead of failing loudly on it, but still fail loudly if the
# gap is bigger than this (a real decode problem).
_MAX_FRAME_COUNT_DRIFT = 8


@dataclass(frozen=True)
class _Job:
    src_drive_dir: Path
    dst_drive_dir: Path
    interpolation: int
    quality: int
    sampling: int
    verify_sample: int


def _ffprobe_nb_frames(video_path: Path) -> int:
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        msg = "ffprobe not found on PATH — required for the frame-count self-check"
        raise RuntimeError(msg)

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _symlink_sidecars(src_drive_dir: Path, dst_drive_dir: Path) -> None:
    dst_drive_dir.mkdir(parents=True, exist_ok=True)
    for entry in src_drive_dir.iterdir():
        if entry.name == "frames":
            continue
        link = dst_drive_dir / entry.name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(entry.resolve())


def _existing_count(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return sum(1 for _ in frames_dir.glob("*.jpg"))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def _check_legacy_frame_count(
    job: _Job, video_path: Path, legacy_frames_dir: Path
) -> int:
    expected = _ffprobe_nb_frames(video_path)
    legacy_count = _existing_count(legacy_frames_dir)
    if legacy_count != expected:
        msg = (
            f"{job.src_drive_dir}: legacy jpg count {legacy_count} != ffprobe "
            f"nb_frames {expected} — refusing to convert against a mismatched drive"
        )
        raise RuntimeError(msg)
    return expected


@dataclass(frozen=True)
class _DecodeResult:
    decoded: int
    sizes: list[int]
    verify_maes: list[float]


def _resize_frame(
    frame_bgr: np.ndarray, job: _Job, frame_idx: int, video_path: Path
) -> np.ndarray:
    import cv2

    h, w = _NATIVE_SIZE
    resized_bgr = cv2.resize(frame_bgr, (w, h), interpolation=job.interpolation)
    if resized_bgr.shape[:2] != (h, w):
        msg = f"{video_path}: frame {frame_idx} resized to {resized_bgr.shape[:2]}, expected {(h, w)}"
        raise RuntimeError(msg)
    return resized_bgr


def _write_jpg(out_path: Path, bgr: np.ndarray, job: _Job) -> None:
    import cv2

    ok = cv2.imwrite(
        str(out_path),
        bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            job.quality,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
            job.sampling,
        ],
    )
    if not ok:
        msg = f"cv2.imwrite failed: {out_path}"
        raise RuntimeError(msg)


def _maybe_verify(
    resized_bgr: np.ndarray,
    legacy_frames_dir: Path,
    frame_idx: int,
    verify_maes: list[float],
    job: _Job,
) -> None:
    import cv2

    legacy_path = legacy_frames_dir / f"{frame_idx + 1:09d}.jpg"
    legacy_bgr = cv2.imread(str(legacy_path))
    if legacy_bgr is not None and len(verify_maes) < job.verify_sample:
        verify_maes.append(_mae(resized_bgr, legacy_bgr))


def _decode_resize_write(
    job: _Job,
    *,
    video_path: Path,
    legacy_frames_dir: Path,
    out_frames_dir: Path,
    expected: int,
) -> _DecodeResult:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        raise RuntimeError(msg)

    verify_stride = max(1, expected // max(job.verify_sample, 1))
    verify_maes: list[float] = []
    sizes: list[int] = []
    decoded = 0
    try:
        while True:
            ret, frame_bgr = (
                cap.read()
            )  # sequential decode — no CAP_PROP_POS_FRAMES seeking
            if not ret:
                break
            resized_bgr = _resize_frame(frame_bgr, job, decoded, video_path)
            out_path = out_frames_dir / f"{decoded + 1:09d}.jpg"
            _write_jpg(out_path, resized_bgr, job)
            sizes.append(out_path.stat().st_size)
            if decoded % verify_stride == 0:
                _maybe_verify(resized_bgr, legacy_frames_dir, decoded, verify_maes, job)
            decoded += 1
    finally:
        cap.release()
    return _DecodeResult(decoded=decoded, sizes=sizes, verify_maes=verify_maes)


def _log_conversion(job: _Job, result: _DecodeResult, written: int) -> None:
    sizes, verify_maes = result.sizes, result.verify_maes
    mean_size_kb = (sum(sizes) / len(sizes) / 1024) if sizes else 0.0
    logger.info(
        "converted drive",
        drive=str(job.src_drive_dir),
        n_frames=written,
        mean_size_kb=round(mean_size_kb, 2),
        verify_n=len(verify_maes),
        verify_mae_mean=round(float(np.mean(verify_maes)), 4) if verify_maes else None,
        verify_mae_max=round(float(np.max(verify_maes)), 4) if verify_maes else None,
    )


def _resume_marker(out_frames_dir: Path) -> Path:
    return out_frames_dir / _DONE_MARKER


def _already_converted(out_frames_dir: Path) -> bool:
    marker = _resume_marker(out_frames_dir)
    if not marker.is_file():
        return False
    try:
        marked_count = int(marker.read_text().strip())
    except ValueError:
        return False
    return marked_count == _existing_count(out_frames_dir)


def _check_decode_count(job: _Job, decoded: int, expected: int) -> None:
    drift = expected - decoded
    if drift == 0:
        return
    if 0 < drift <= _MAX_FRAME_COUNT_DRIFT:
        logger.warning(
            "decoded frame count short of ffprobe nb_frames — tolerated, "
            "likely a source HEVC bitstream defect (see stderr for libav "
            "'Duplicate POC'/'undecodable NALU' diagnostics); confirm by "
            "cross-checking with `ffmpeg -vsync 0 -f null -` before trusting "
            "this drive if the drift is unexpected",
            drive=str(job.src_drive_dir),
            decoded=decoded,
            ffprobe_nb_frames=expected,
            drift=drift,
        )
        return
    msg = (
        f"{job.src_drive_dir}: decoded={decoded} vs ffprobe_nb_frames={expected} "
        f"(drift={drift}) exceeds the tolerated drift of {_MAX_FRAME_COUNT_DRIFT} "
        "— self-check failed"
    )
    raise RuntimeError(msg)


def _convert_drive(job: _Job, *, resume: bool) -> None:
    video_path = job.src_drive_dir / _VIDEO_NAME
    legacy_frames_dir = job.src_drive_dir / _FRAMES_RELPATH
    out_frames_dir = job.dst_drive_dir / _FRAMES_RELPATH

    expected = _check_legacy_frame_count(job, video_path, legacy_frames_dir)

    if resume and _already_converted(out_frames_dir):
        logger.info(
            "resume: skipping, already converted",
            drive=str(job.src_drive_dir),
            n=_existing_count(out_frames_dir),
        )
        _symlink_sidecars(job.src_drive_dir, job.dst_drive_dir)
        return

    out_frames_dir.mkdir(parents=True, exist_ok=True)
    result = _decode_resize_write(
        job,
        video_path=video_path,
        legacy_frames_dir=legacy_frames_dir,
        out_frames_dir=out_frames_dir,
        expected=expected,
    )

    written = _existing_count(out_frames_dir)
    if written != result.decoded:
        msg = (
            f"{job.src_drive_dir}: written={written} != decoded={result.decoded} "
            "— self-check failed"
        )
        raise RuntimeError(msg)
    _check_decode_count(job, result.decoded, expected)

    _resume_marker(out_frames_dir).write_text(str(written))
    _symlink_sidecars(job.src_drive_dir, job.dst_drive_dir)
    _log_conversion(job, result, written)


def _resolve_interpolation(name: str) -> int:
    import cv2

    return getattr(cv2, _INTERPOLATIONS[name])


def _resolve_sampling(name: str) -> int:
    import cv2

    return getattr(cv2, _SAMPLING_FACTORS[name])


def _build_jobs(args: argparse.Namespace) -> list[_Job]:
    interpolation = _resolve_interpolation(args.interpolation)
    sampling = _resolve_sampling(args.sampling)
    return [
        _Job(
            src_drive_dir=args.src_root / drive,
            dst_drive_dir=args.dst_root / drive,
            interpolation=interpolation,
            quality=args.quality,
            sampling=sampling,
            verify_sample=args.verify_sample,
        )
        for drive in args.drive
    ]


def _run_job(job: _Job, *, resume: bool) -> tuple[str, Exception | None]:
    try:
        _convert_drive(job, resume=resume)
    except Exception as exc:
        return str(job.src_drive_dir), exc
    return str(job.src_drive_dir), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--interpolation", choices=list(_INTERPOLATIONS), default="cubic"
    )
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--sampling", choices=list(_SAMPLING_FACTORS), default="444")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-sample", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--drive",
        action="append",
        required=True,
        help="repeatable: <vehicle>/<drive_id>",
    )
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        msg = "ffprobe not found on PATH — required for the frame-count self-check"
        raise RuntimeError(msg)

    jobs = _build_jobs(args)
    logger.info("starting conversion", n_drives=len(jobs), workers=args.workers)

    failures: list[tuple[str, Exception]] = []
    # Parallelize across drives only — each drive's video is decoded sequentially.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for drive, exc in pool.map(lambda j: _run_job(j, resume=args.resume), jobs):
            if exc is not None:
                failures.append((drive, exc))

    if failures:
        for drive, exc in failures:
            logger.error("drive conversion failed", drive=drive, error=str(exc))
        msg = f"{len(failures)}/{len(jobs)} drive(s) failed"
        raise RuntimeError(msg)

    logger.info("all drives converted", n_drives=len(jobs))


if __name__ == "__main__":
    main()
