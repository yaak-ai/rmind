"""ffmpeg mjpeg quantization parity — the exact `-q:v` tables `extract_frames`
bakes into training's jpgs, reproduced without a subprocess or reference file.

Extracted out of `rmind.scripts.benchmark_onnx` (where it was validated against
real drive data — see `tests/test_benchmark_onnx_preprocessing.py`) so that
non-training consumers, notably `drivr`'s live camera pipeline, can round-trip
a frame through the same quantization without depending on script internals.
`benchmark_onnx` re-exports these names for backward compatibility with its
existing tests.
"""

import io

import numpy as np

# ffmpeg's mjpeg encoder scales `ff_mpeg1_default_intra_matrix` by `-q:v` to
# build its luma quantization table (`ff_convert_matrix` in libavcodec). This
# is that base matrix in Pillow's row-major DQT order — NOT the on-disk
# zig-zag coefficient order.
FFMPEG_MJPEG_BASE_QTABLE = (
    8, 16, 19, 22, 26, 27, 29, 34,
    16, 16, 22, 24, 27, 29, 34, 37,
    19, 22, 26, 27, 29, 34, 34, 38,
    22, 22, 26, 27, 29, 34, 37, 40,
    22, 26, 27, 29, 32, 35, 40, 48,
    26, 27, 29, 32, 35, 40, 48, 58,
    26, 27, 29, 34, 38, 46, 56, 69,
    27, 29, 35, 38, 46, 56, 69, 83,
)  # fmt: skip

# extract_frames writes yuvj420p; Pillow spells 4:2:0 as subsampling=2.
JPEG_SUBSAMPLING_420 = 2

JPEG_QV_RANGE = (2, 31)  # ffmpeg's mjpeg -q:v scale

_HWC_NDIM = 3
_RGB_CHANNELS = 3
DEFAULT_JPEG_QV = 16  # extract_frames' own -q:v, per config/dvc.yaml


def ffmpeg_mjpeg_qtable(qv: int) -> list[int]:
    """ffmpeg's mjpeg quantization table for `-q:v qv`, in Pillow's order.

    Raises:
        ValueError: if `qv` is outside ffmpeg's mjpeg -q:v range (2..31) —
            notably when a libjpeg-style 0-100 quality is passed by mistake.
    """
    lo, hi = JPEG_QV_RANGE
    if not lo <= qv <= hi:
        msg = f"jpeg_qv must be in ffmpeg's mjpeg -q:v range {lo}..{hi}, got {qv}"
        raise ValueError(msg)
    table = [min(max(base * qv // 8, 1), 255) for base in FFMPEG_MJPEG_BASE_QTABLE]
    table[0] = 8
    return table


def simulate_offline_jpeg(image_rgb: np.ndarray, *, qv: int) -> np.ndarray:
    """Round-trip an RGB frame through extract_frames' exact mjpeg settings.

    Encodes with ffmpeg's `-q:v qv` quantization table at 4:2:0 -- Pillow is
    the only encoder available here that accepts explicit tables (simplejpeg
    and cv2 both only take a libjpeg 0-100 quality, which cannot express
    them) -- then decodes with cv2, so the decoder matches training's own
    frame reads.

    Raises:
        ValueError: if `image_rgb` is not an HWC uint8 RGB array.
        RuntimeError: if the just-encoded bytes fail to decode.
    """
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != _HWC_NDIM
        or image_rgb.shape[2] != _RGB_CHANNELS
    ):
        msg = (
            "expected an HWC uint8 RGB image, got "
            f"{image_rgb.shape} of {image_rgb.dtype}"
        )
        raise ValueError(msg)

    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(
        buffer,
        format="JPEG",
        qtables=[ffmpeg_mjpeg_qtable(qv)],
        subsampling=JPEG_SUBSAMPLING_420,
    )
    decoded_bgr = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded_bgr is None:
        # we just encoded these bytes, so this can only be a bug here
        msg = f"failed to decode the jpeg re-encoded at -q:v {qv}"
        raise RuntimeError(msg)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
