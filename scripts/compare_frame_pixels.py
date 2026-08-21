"""Step 1 pixel diagnostic — is `scale_npp` (NPPI_INTER_CUBIC) reproducible in Python?

See jpg_preprocessing_parity_task.md §4. For ~40 strided frames of one drive, this
reports three things as `tabulate` tables, in both uint8 pixel space and the model's
input space (224x224, ImageNet-normalized via `_preprocess_image`):

  1. Kernel sweep — legacy jpg (produced offline by ffmpeg's `scale_npp` =
     NPPI_INTER_CUBIC, no Python equivalent) vs the same video frame resized to
     576x324 with each of several candidate kernels. Whichever candidate lands
     closest to the legacy jpg is the best available stand-in for `scale_npp`.
  2. JPEG-only floor — the INTER_CUBIC candidate vs its own JPEG round-trip at
     (q95, 4:4:4) and (q80, 4:2:0). This is the best a re-extraction (arm B) could
     ever achieve, since re-encoding to JPEG is itself lossy.
  3. Decoder term — the same legacy jpg bytes decoded by `simplejpeg` (what
     *training* uses) vs `cv2.imread` (what *benchmark_onnx.py* uses). Measured,
     not fixed, here.

Reuses `_preprocess_image`, `_VideoFrameSource`, `_JpgFrameSource`, `_jpg_frames_dir`,
and `DEFAULT_IMAGE_SIZE` from benchmark_onnx.py rather than reimplementing any of the
crop/resize/normalize pipeline. The one deliberate exception: `_preprocess_image`
hardcodes `cv2.INTER_CUBIC` for its 1920x1080 -> DEFAULT_IMAGE_SIZE downscale, which
is exactly the parameter this script sweeps — so the sweep produces each candidate's
576x324 array itself (with the swept kernel) and then still hands it to
`_preprocess_image` for the "model input space" comparison. Because the candidates are
already at DEFAULT_IMAGE_SIZE, `_preprocess_image`'s internal same-size CUBIC resize
is a near-identity pass-through applied uniformly to every row, so it does not bias
the comparison between kernels.

Usage:
    uv run --group benchmark python scripts/compare_frame_pixels.py \\
        --data-dir /nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14
"""

# Reporting to stdout is this script's entire purpose (see benchmark_onnx.py's own
# VALIDATION CHECKS block for the same pattern) — blanket-ignore T201 rather than
# annotate every print() call individually.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import structlog
from rmind.scripts.benchmark_onnx import (
    DEFAULT_IMAGE_SIZE,
    _jpg_frames_dir,
    _JpgFrameSource,
    _preprocess_image,
    _VideoFrameSource,
)
from tabulate import tabulate

logger = structlog.get_logger(__name__)

MODEL_IMAGE_SIZE = (224, 224)  # (H, W) — patch_policy/raw.yaml's image_resize
_HEADERS = ["candidate", "MAE", "p99", "max"]

_SWEEP_CV2_NAMES = [
    "cv2.INTER_CUBIC",
    "cv2.INTER_AREA",
    "cv2.INTER_LANCZOS4",
    "cv2.INTER_LINEAR",
    "cv2.INTER_LINEAR_EXACT",
]
_SWEEP_TORCHVISION_NAMES = [
    "torchvision.Resize(antialias=True)",
    "torchvision.Resize(antialias=False)",
]
_SWEEP_NAMES = [*_SWEEP_CV2_NAMES, *_SWEEP_TORCHVISION_NAMES]
_FLOOR_SETTINGS = {
    "jpeg_roundtrip(q95,444)": (95, True),
    "jpeg_roundtrip(q80,420)": (80, False),
}
_FLOOR_NAMES = list(_FLOOR_SETTINGS)
_DECODER_LABEL = "simplejpeg vs cv2.imread (same legacy jpg bytes)"


def _cv2_kernel_constants() -> dict[str, int]:
    import cv2

    return {
        "cv2.INTER_CUBIC": cv2.INTER_CUBIC,
        "cv2.INTER_AREA": cv2.INTER_AREA,
        "cv2.INTER_LANCZOS4": cv2.INTER_LANCZOS4,
        "cv2.INTER_LINEAR": cv2.INTER_LINEAR,
        "cv2.INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT,
    }


def _cv2_resize_to_native(frame_rgb: np.ndarray, interpolation: int) -> np.ndarray:
    import cv2

    h, w = DEFAULT_IMAGE_SIZE
    return cv2.resize(frame_rgb, (w, h), interpolation=interpolation)


def _torchvision_resize_to_native(
    frame_rgb: np.ndarray, *, antialias: bool
) -> np.ndarray:
    import torch
    from torchvision.transforms import v2 as T

    h, w = DEFAULT_IMAGE_SIZE
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)  # uint8 NCHW
    resize = T.Resize([h, w], antialias=antialias)
    out = resize(tensor)[0].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(out)


def _jpeg_roundtrip(
    frame_rgb: np.ndarray, *, quality: int, sampling_444: bool
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    sampling = (
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
        if sampling_444
        else cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420
    )
    ok, buf = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling],
    )
    if not ok:
        msg = "cv2.imencode failed"
        raise RuntimeError(msg)
    decoded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        msg = "cv2.imdecode failed"
        raise RuntimeError(msg)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _simplejpeg_decode(path: Path) -> np.ndarray:
    import simplejpeg

    data = path.read_bytes()
    return simplejpeg.decode_jpeg(
        data, colorspace="rgb", fastdct=True, fastupsample=True
    )


class _RunningStats:
    """Per-frame (mean, p99, max) abs-diff stats, aggregated across frames as
    (mean-of-means, mean-of-p99s, max-of-maxes) — cheap to accumulate online
    without holding every sampled frame's full pixel-diff array in memory at once.
    """

    def __init__(self) -> None:
        self._means: list[float] = []
        self._p99s: list[float] = []
        self._maxes: list[float] = []

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        self._means.append(float(diff.mean()))
        self._p99s.append(float(np.percentile(diff, 99)))
        self._maxes.append(float(diff.max()))

    def mean_mae(self) -> float:
        return float(np.mean(self._means))

    def row(self, name: str) -> list[str | float]:
        return [
            name,
            round(self.mean_mae(), 4),
            round(float(np.mean(self._p99s)), 4),
            round(float(np.max(self._maxes)), 4),
        ]


class _Stats:
    """Bundles every running-stats table this diagnostic accumulates."""

    def __init__(self) -> None:
        self.sweep_uint8 = {n: _RunningStats() for n in _SWEEP_NAMES}
        self.sweep_model = {n: _RunningStats() for n in _SWEEP_NAMES}
        self.floor_uint8 = {n: _RunningStats() for n in _FLOOR_NAMES}
        self.floor_model = {n: _RunningStats() for n in _FLOOR_NAMES}
        self.decoder_uint8 = _RunningStats()
        self.decoder_model = _RunningStats()


class _Sources:
    """Bundles everything `_process_frame` reads a frame from."""

    def __init__(self, data_dir: Path) -> None:
        self.frames_dir = _jpg_frames_dir(data_dir)
        self.video = _VideoFrameSource(data_dir / "cam_front_left.pii.mp4")
        self.jpg = _JpgFrameSource(self.frames_dir)
        self.cv2_kernels = _cv2_kernel_constants()

    def close(self) -> None:
        self.video.close()
        self.jpg.close()


def _process_frame(frame_idx: int, sources: _Sources, stats: _Stats) -> None:
    frame_rgb = sources.video.read(frame_idx)  # native video res, uint8 RGB
    legacy_rgb = sources.jpg.read(frame_idx)  # 576x324 uint8 RGB (cv2.imread path)
    legacy_model = _preprocess_image(legacy_rgb, MODEL_IMAGE_SIZE)

    # 1. kernel sweep
    candidates = {
        name: _cv2_resize_to_native(frame_rgb, kernel)
        for name, kernel in sources.cv2_kernels.items()
    }
    for name in _SWEEP_TORCHVISION_NAMES:
        candidates[name] = _torchvision_resize_to_native(
            frame_rgb, antialias="True" in name
        )
    for name, cand in candidates.items():
        stats.sweep_uint8[name].add(cand, legacy_rgb)
        stats.sweep_model[name].add(
            _preprocess_image(cand, MODEL_IMAGE_SIZE), legacy_model
        )

    # 2. JPEG-only floor, relative to the INTER_CUBIC candidate itself
    cubic_native = candidates["cv2.INTER_CUBIC"]
    cubic_model = _preprocess_image(cubic_native, MODEL_IMAGE_SIZE)
    for name, (quality, sampling_444) in _FLOOR_SETTINGS.items():
        rt = _jpeg_roundtrip(cubic_native, quality=quality, sampling_444=sampling_444)
        stats.floor_uint8[name].add(rt, cubic_native)
        stats.floor_model[name].add(
            _preprocess_image(rt, MODEL_IMAGE_SIZE), cubic_model
        )

    # 3. decoder term: simplejpeg (training) vs cv2.imread (benchmark), same bytes
    jpg_path = sources.frames_dir / f"{frame_idx + 1:09d}.jpg"
    simplejpeg_rgb = _simplejpeg_decode(jpg_path)
    stats.decoder_uint8.add(simplejpeg_rgb, legacy_rgb)
    stats.decoder_model.add(
        _preprocess_image(simplejpeg_rgb, MODEL_IMAGE_SIZE), legacy_model
    )


def _sample_frame_indices(frames_dir: Path, num_frames: int, margin: int) -> list[int]:
    jpg_count = len(list(frames_dir.glob("*.jpg")))
    lo, hi = margin, jpg_count - margin - 1
    indices = sorted({int(i) for i in np.linspace(lo, hi, num_frames, dtype=int)})
    logger.info("sampling", n=len(indices), lo=lo, hi=hi, jpg_count=jpg_count)
    return indices


def _print_report(stats: _Stats, n_frames: int) -> None:
    print(
        f"\n=== 1. Kernel sweep — legacy jpg vs video frame resized to "
        f"{DEFAULT_IMAGE_SIZE} ({n_frames} frames) ==="
    )
    print("-- uint8 space --")
    print(
        tabulate(
            [stats.sweep_uint8[n].row(n) for n in _SWEEP_NAMES],
            headers=_HEADERS,
            floatfmt=".4f",
        )
    )
    print("-- model input space (224x224, ImageNet-normalized) --")
    print(
        tabulate(
            [stats.sweep_model[n].row(n) for n in _SWEEP_NAMES],
            headers=_HEADERS,
            floatfmt=".4f",
        )
    )
    best = min(_SWEEP_NAMES, key=lambda n: stats.sweep_uint8[n].mean_mae())
    print(f"closest kernel to legacy jpg (by uint8 MAE): {best}")

    print(
        "\n=== 2. JPEG-only floor — INTER_CUBIC candidate vs its own JPEG round-trip ==="
    )
    print("-- uint8 space --")
    print(
        tabulate(
            [stats.floor_uint8[n].row(n) for n in _FLOOR_NAMES],
            headers=_HEADERS,
            floatfmt=".4f",
        )
    )
    print("-- model input space --")
    print(
        tabulate(
            [stats.floor_model[n].row(n) for n in _FLOOR_NAMES],
            headers=_HEADERS,
            floatfmt=".4f",
        )
    )

    print(
        "\n=== 3. Decoder term — simplejpeg (training) vs cv2.imread (benchmark) on the same legacy jpg ==="
    )
    print(
        tabulate(
            [
                stats.decoder_uint8.row(_DECODER_LABEL + " [uint8]"),
                stats.decoder_model.row(_DECODER_LABEL + " [model]"),
            ],
            headers=_HEADERS,
            floatfmt=".4f",
        )
    )

    best_uint8_mae = stats.sweep_uint8[best].mean_mae()
    floor_uint8_mae = min(stats.floor_uint8[n].mean_mae() for n in _FLOOR_NAMES)
    best_model_mae = stats.sweep_model[best].mean_mae()
    floor_model_mae = min(stats.floor_model[n].mean_mae() for n in _FLOOR_NAMES)
    print("\n=== GATE ===")
    print(
        f"best kernel ({best}) uint8 MAE={best_uint8_mae:.4f} vs JPEG floor uint8 MAE={floor_uint8_mae:.4f}"
    )
    print(
        f"best kernel ({best}) model MAE={best_model_mae:.4f} vs JPEG floor model MAE={floor_model_mae:.4f}"
    )
    if best_uint8_mae <= floor_uint8_mae and best_model_mae <= floor_model_mae:
        print(
            "GATE: PASS — best kernel is at/below the JPEG floor in both spaces. "
            "scale_npp is effectively reproducible in Python. STOP: skip steps 2-4."
        )
    else:
        print(
            "GATE: FAIL — best kernel remains above the JPEG floor. Continue to step 2 (re-extraction)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14"),
    )
    parser.add_argument("--num-frames", type=int, default=40)
    parser.add_argument(
        "--margin",
        type=int,
        default=200,
        help="frames to exclude from each end of the drive (avoid boundary artifacts)",
    )
    args = parser.parse_args()

    logger.info("drive", data_dir=str(args.data_dir))
    sources = _Sources(args.data_dir)
    frame_indices = _sample_frame_indices(
        sources.frames_dir, args.num_frames, args.margin
    )
    stats = _Stats()

    try:
        for frame_idx in frame_indices:
            _process_frame(frame_idx, sources, stats)
    finally:
        sources.close()

    _print_report(stats, len(frame_indices))


if __name__ == "__main__":
    main()
