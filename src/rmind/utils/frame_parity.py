"""Reproduce `extract_frames`' pixel chain as GPU tensor ops.

Training frames are produced by the `extract_frames` stage of
`/home/alex/data/dvc.yaml`, verbatim::

    ffmpeg -y -vsync 0 -threads 0 -hwaccel cuda -hwaccel_output_format cuda
           -c:v hevc_cuvid -i <video>
           -filter_complex "scale_npp=576:324,hwdownload,format=nv12"
           -f image2 -q:v 16 <out>/%09d.jpg

`rmind.scripts.benchmark_onnx`'s "ffmpeg" frame source re-runs that command and
was, per `frame_source_parity_report.md`, the closest anything got to the real
training jpgs (1.59 uint8 MAE). It is also unusable in a real-time loop: it
decodes a whole video file, shells out per 512-frame block, and writes jpgs to
disk. `drivr`'s live camera has none of those affordances.

This module is the same *pixel arithmetic* with no ffmpeg, no encoder, no
subprocess and no disk — four stages, every one of them fitted against ffmpeg's
own intermediate output rather than guessed (see `_fit` notes on each):

1. `Resampler` — the resize, done on the YUV 4:2:0 planes so chroma is filtered
   at half resolution. `scale_npp`/`scale_cuda` are a plain 4-tap Keys cubic
   with **no** antialias prefilter, which is why an antialiased resize is
   *worse* here despite being the better resampler in the abstract.
2. `expand_range` — nv12 is limited range ("tv"), the mjpeg encoder is fed
   full-range yuvj420p. ffmpeg inserts this conversion silently.
3. `quantize_planes` — the mjpeg `-q:v 16` DCT round trip, on the 4:2:0 planes
   directly. The entropy-coding half of JPEG is lossless, so reproducing the
   quantization reproduces the encoder.
4. `yuv420_to_rgb` — libjpeg's triangle chroma upsample and JPEG BT.601, i.e.
   what `cv2.imread` does to a training jpg on the way back out.

Measured on 20 frames of `Niro122-HQ/2023-05-25--09-34-14` against the real
training jpgs (uint8 MAE at 576x324) -- `tests/test_frame_parity.py` asserts
these, and every ablation is a stage of the chain being dropped:

    two adjacent real training frames (scene motion)     0.64
    ==> this module, from the same nv12 planes           1.13
    ffmpeg end-to-end, scale_cuda bicubic + mjpeg q16    1.58
      ... with F.interpolate's bicubic instead (a=-0.75) 1.31
      ... without rounding planes to uint8 before DCT    1.28
      ... with an antialiased resize                     3.19
      ... without the DCT quantize                       3.87
      ... without the tv->pc range conversion            6.88

The chain lands *below* ffmpeg's own baseline: this module's Catmull-Rom is
evidently closer to `scale_npp` than `scale_cuda`'s kernel is, and `scale_npp`
is what actually produced the training data. So 1.59 was never a floor.

Cost, warmed up, on GPU-resident planes: **1.51 ms/frame** at 1920x1080 ->
576x324 on an RTX 5090, or 1.60 ms including `uyvy_to_yuv420` on a packed 4:2:2
buffer (which is what a live V4L2 camera hands over). `quantize_planes` is ~0.8
ms of that. Note the constant tables are `lru_cache`d per (device, dtype): they
are 64 floats each, but rebuilding them per frame costs a host-to-device sync
and measured 4.4 ms/frame rather than 1.51. Requires torch only, so it imports
on the Orin.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from rmind.utils.jpeg import DEFAULT_JPEG_QV, ffmpeg_mjpeg_qtable

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FFMPEG_MJPEG_INTRA_QUANT_BIAS",
    "NPP_CUBIC_A",
    "Resampler",
    "Resamplers",
    "expand_range",
    "quantize_planes",
    "quantize_rgb",
    "to_training_frame",
    "uyvy_to_yuv420",
    "yuv420_to_rgb",
]

# Keys cubic parameter. `scale_npp`/`scale_cuda` use Catmull-Rom (a = -1/2);
# torch's `F.interpolate(mode="bicubic")` hardcodes a = -3/4, which is a
# measurably worse match (1.31 vs 1.13 end-to-end) — hence `Resampler` rather
# than a call to `F.interpolate`. _fit: `kernel_fit.py` compared every candidate
# against `scale_cuda`'s own output plane-for-plane; a = -1/2 won at 0.51 uint8
# MAE against a = -0.75's 0.63.
NPP_CUBIC_A = -0.5

# ffmpeg's mjpeg quantizer is NOT round-to-nearest. `dct_quantize_c` in
# libavcodec/mpegvideo_enc.c computes `(|coef| * qmat + bias) >> shift`, i.e.
# truncation with a bias, and `ff_mpv_encode_init` sets
# `intra_quant_bias = 3 << (QUANT_BIAS_SHIFT - 3)` = 96/256 for MJPEG.
# Getting this wrong is the difference between 1.13 and ~2.4 MAE: round-to-
# nearest reproduces the encoder's output to only 1.14 MAE in the plane domain,
# where 3/8 reproduces it to 0.13. _fit: `dct_fit.py` swept the bias against
# ffmpeg's real encode/decode of the identical input planes.
FFMPEG_MJPEG_INTRA_QUANT_BIAS = 3 / 8

# BT.601 limited ("tv") range, which is what the training videos carry
# (`ffprobe` reports `color_range=tv`, and every other field `unknown`).
_LIMITED_Y_OFFSET, _LIMITED_Y_SCALE = 16.0, 255.0 / 219.0
_LIMITED_C_SCALE = 255.0 / 224.0

# JPEG is always full-range BT.601, whatever the source video was tagged.
# libjpeg's jdcolor.c coefficients; reproduces `cv2.imread` on a real training
# jpg to 0.12 uint8 MAE (_fit: `decode_fit.py`, same file both sides).
_YCC_TO_RGB = ((1.0, 0.0, 1.402), (1.0, -0.344136286, -0.714136286), (1.0, 1.772, 0.0))

_BLOCK = 8  # JPEG's DCT block size
_UYVY_BYTES_PER_PIXEL = 2  # packed 4:2:2: one chroma byte and one luma byte
_HWC_NDIM = 3
# A Keys cubic is piecewise over |x| < 1 and |x| < 2 — 4 taps, and NOT
# widened by the downscale ratio (see NPP_CUBIC_A).
_CUBIC_INNER, _CUBIC_SUPPORT = 1, 2


class Resampler:
    """Separable Keys-cubic resize with precomputed gather indices and weights.

    Geometry is fixed for the life of a camera, so the taps are built once and
    every frame is two gathers and two weighted sums. Half-pixel centers
    (matching libswscale/NPP and `align_corners=False`), edges handled by index
    clamping, which is the replication NPP does.

    Chroma and luma need separate instances — that is the point of resizing in
    4:2:0 rather than RGB.
    """

    def __init__(
        self,
        in_hw: tuple[int, int],
        out_hw: tuple[int, int],
        *,
        a: float = NPP_CUBIC_A,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.in_hw = tuple(in_hw)
        self.out_hw = tuple(out_hw)
        # dim 2 is height, dim 3 is width for the [N, C, H, W] this is applied to
        self._taps = tuple(
            self._axis(int(src), int(dst), a=a, device=device, dtype=dtype)
            for src, dst in zip(in_hw, out_hw, strict=True)
        )

    @staticmethod
    def _keys_cubic(x: torch.Tensor, a: float) -> torch.Tensor:
        x = x.abs()
        x2 = x * x
        x3 = x2 * x
        near = (a + 2) * x3 - (a + 3) * x2 + 1
        far = a * x3 - 5 * a * x2 + 8 * a * x - 4 * a
        return torch.where(
            x < _CUBIC_INNER,
            near,
            torch.where(x < _CUBIC_SUPPORT, far, torch.zeros_like(x)),
        )

    @classmethod
    def _axis(
        cls,
        src: int,
        dst: int,
        *,
        a: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = src / dst
        support = float(_CUBIC_SUPPORT)  # deliberately NOT scaled by `ratio`
        # float64 for the tap geometry only — it is O(dst * taps), computed once,
        # and half-pixel center arithmetic is where rounding actually shows up.
        centers = (
            torch.arange(dst, device=device, dtype=torch.float64) + 0.5
        ) * ratio - 0.5
        taps = int(2 * support) + 1
        first = torch.floor(centers - support + 0.5).long()
        index = first[:, None] + torch.arange(taps, device=device)[None, :]
        weight = cls._keys_cubic(index.double() - centers[:, None], a)
        weight /= weight.sum(dim=1, keepdim=True)
        return index.clamp(0, src - 1), weight.to(dtype)

    def __call__(self, planes: torch.Tensor) -> torch.Tensor:
        """Resize [N, C, H, W] to `out_hw`. Float in, float out, unclamped.

        Raises:
            ValueError: if `planes` is not the size this was built for. The taps
                are precomputed, so a mismatch cannot be handled silently.
        """
        if planes.shape[-2:] != self.in_hw:
            msg = (
                f"expected planes of size {self.in_hw}, got {tuple(planes.shape[-2:])}"
            )
            raise ValueError(msg)
        (h_index, h_weight), (w_index, w_weight) = self._taps
        x = (planes[..., :, w_index] * w_weight).sum(-1)  # [N, C, H, W_out]
        x = x.transpose(-2, -1)  # [N, C, W_out, H]
        x = (x[..., :, h_index] * h_weight).sum(-1)  # [N, C, W_out, H_out]
        return x.transpose(-2, -1)


def uyvy_to_yuv420(uyvy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed UYVY 4:2:2 -> planar 4:2:0, as `(luma, chroma)`.

    A V4L2 UYVY buffer arrives as [H, W, 2] uint8 with the byte order
    U Y0 V Y1 — chroma at half horizontal resolution, full vertical. Training's
    HEVC source is 4:2:0, so chroma is box-averaged down vertically to match.
    Doing this instead of letting OpenCV convert to BGR is what keeps the whole
    chain in the domain training's resize happened in.

    Returns:
        luma `[1, 1, H, W]` and chroma `[1, 2, H // 2, W // 2]`, both float, in
        the source's own range (i.e. still limited — see `expand_range`).

    Raises:
        ValueError: if the buffer is not `[H, W, 2]`, or has odd dimensions
            (4:2:0 needs to halve both axes).
    """
    if uyvy.ndim != _HWC_NDIM or uyvy.shape[-1] != _UYVY_BYTES_PER_PIXEL:
        msg = (
            f"expected a packed UYVY buffer of shape [H, W, 2], got {tuple(uyvy.shape)}"
        )
        raise ValueError(msg)
    height, width, _ = uyvy.shape
    if height % 2 or width % 2:
        msg = f"UYVY 4:2:0 conversion needs even dimensions, got {height}x{width}"
        raise ValueError(msg)

    packed = uyvy.float()
    luma = packed[..., 1].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # chroma bytes alternate U, V along the row: U at even x, V at odd x
    chroma_422 = packed[..., 0].reshape(height, width // 2, 2)  # [H, W/2, (U, V)]
    chroma_422 = chroma_422.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W/2]
    chroma = F.avg_pool2d(chroma_422, kernel_size=(2, 1))  # 4:2:2 -> 4:2:0
    return luma, chroma


def nv12_to_planes(
    buffer: torch.Tensor, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat NV12 bytes -> `(luma [1, 1, H, W], chroma [1, 2, H/2, W/2])`, float.

    NV12 is a full-resolution luma plane followed by interleaved Cb/Cr at half
    resolution in both axes — the layout `hwdownload,format=nv12` produces, and
    so the layout `benchmark_onnx`'s ffmpeg-decoded frames arrive in.

    Raises:
        ValueError: if the buffer is not exactly `height * width * 3 // 2` bytes.
    """
    expected = height * width * 3 // 2
    flat = buffer.reshape(-1)
    if flat.numel() != expected:
        msg = f"expected {expected} NV12 bytes for {height}x{width}, got {flat.numel()}"
        raise ValueError(msg)
    luma = flat[: height * width].reshape(1, 1, height, width).float()
    chroma = flat[height * width :].reshape(height // 2, width // 2, 2)
    return luma, chroma.permute(2, 0, 1).unsqueeze(0).float()


def expand_range(
    luma: torch.Tensor, chroma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Limited ("tv") -> full ("pc") range, what ffmpeg's nv12 -> yuvj420p does.

    Load-bearing and easy to omit: skipping it costs 6.88 vs 1.13 uint8 MAE,
    more than every other stage of this module combined. Reproduces ffmpeg's own
    conversion exactly on luma and to 0.5 MAE on chroma.
    """
    return (
        (luma - _LIMITED_Y_OFFSET) * _LIMITED_Y_SCALE,
        (chroma - 128.0) * _LIMITED_C_SCALE + 128.0,
    )


@lru_cache(maxsize=8)
def _dct_matrix(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix `D`, so a block transforms as `D @ x @ D.T`.

    Cached: it is 64 constants, but building it means a host-side float64
    computation and a device transfer, which is a per-frame sync in a real-time
    loop and was measurably most of this module's cost before caching.
    """
    n = torch.arange(_BLOCK, device=device, dtype=torch.float64)
    k = n.view(_BLOCK, 1)
    scale = 0.5 * torch.where(k == 0, torch.full_like(k, 2**-0.5), torch.ones_like(k))
    return (torch.cos(torch.pi * (2 * n + 1) * k / (2 * _BLOCK)) * scale).to(dtype)


@lru_cache(maxsize=32)
def _qtable_tensor(
    qv: int, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """`ffmpeg_mjpeg_qtable` as an 8x8 device tensor. Cached, as above."""
    return torch.tensor(ffmpeg_mjpeg_qtable(qv), device=device, dtype=dtype).reshape(
        _BLOCK, _BLOCK
    )


@lru_cache(maxsize=8)
def _ycc_to_rgb_matrix(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """`_YCC_TO_RGB` shaped as a 1x1 convolution kernel. Cached, as above."""
    return torch.tensor(_YCC_TO_RGB, device=device, dtype=dtype).reshape(3, 3, 1, 1)


def quantize_planes(
    luma: torch.Tensor,
    chroma: torch.Tensor,
    *,
    qv: int = DEFAULT_JPEG_QV,
    bias: float = FFMPEG_MJPEG_INTRA_QUANT_BIAS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip 4:2:0 planes through ffmpeg's mjpeg `-q:v qv` quantization.

    The model has only ever seen DCT-quantized pixels, and this is the single
    largest recoverable term: dropping it costs 3.87 vs 1.13 uint8 MAE.

    Runs on the *planes*, not on RGB, because that is where the real encoder
    runs — quantizing an RGB frame means subsampling chroma a second time, which
    measurably hurts (3.45 vs 2.87 at an intermediate stage of the fit). It also
    needs no encoder at all: JPEG's entropy coding is lossless, so quantization
    is the whole of what a round trip does to the pixels. Unlike
    `torchvision.io.encode_jpeg`/nvjpeg this can express ffmpeg's actual table,
    which a libjpeg 0-100 quality provably cannot (see `rmind.utils.jpeg`).

    Args:
        luma: `[N, 1, H, W]` float in 0..255, full range.
        chroma: `[N, 2, H/2, W/2]` float in 0..255, full range.
        qv: ffmpeg's mjpeg quantizer scale, 2..31. Training used 16.
        bias: quantizer bias; see `FFMPEG_MJPEG_INTRA_QUANT_BIAS`.

    Returns:
        The same shapes, rounded and clamped to 0..255 but still float.

    Raises:
        ValueError: if `luma` is not float32 or better, or if `qv` is outside
            ffmpeg's 2..31 range (notably when a libjpeg 0-100 quality is passed
            by mistake, which `ffmpeg_mjpeg_qtable` rejects).
    """
    if luma.dtype not in {torch.float32, torch.float64}:
        # a DC coefficient runs to ~1024 before quantization; fp16 cannot
        # represent `floor(coef / q + bias)` there without changing the level
        msg = f"quantize_planes needs float32 or better, got {luma.dtype}"
        raise ValueError(msg)
    qtable = _qtable_tensor(qv, luma.device, luma.dtype)
    return (
        _quantize(luma, qtable, bias).round().clamp(0, 255),
        _quantize(chroma, qtable, bias).round().clamp(0, 255),
    )


def _quantize(planes: torch.Tensor, qtable: torch.Tensor, bias: float) -> torch.Tensor:
    n, c, height, width = planes.shape
    pad_h, pad_w = (-height) % _BLOCK, (-width) % _BLOCK
    if pad_h or pad_w:
        # 324 is 40.5 blocks tall; JPEG pads the partial MCU by edge replication
        planes = F.pad(planes, (0, pad_w, 0, pad_h), mode="replicate")
    blocks_h, blocks_w = planes.shape[-2] // _BLOCK, planes.shape[-1] // _BLOCK

    dct = _dct_matrix(planes.device, planes.dtype)
    blocks = (
        planes
        .reshape(n, c, blocks_h, _BLOCK, blocks_w, _BLOCK)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(-1, _BLOCK, _BLOCK)
        - 128.0  # JPEG's level shift
    )
    coefficients = dct @ blocks @ dct.T
    # ffmpeg truncates toward zero with `bias`, rather than rounding to nearest
    levels = torch.floor(coefficients.abs() / qtable + bias)
    dequantized = torch.sign(coefficients) * levels * qtable
    out = (dct.T @ dequantized @ dct) + 128.0

    out = (
        out
        .reshape(n, c, blocks_h, blocks_w, _BLOCK, _BLOCK)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(n, c, blocks_h * _BLOCK, blocks_w * _BLOCK)
    )
    return out[..., :height, :width]


def quantize_rgb(rgb: torch.Tensor, *, qv: int = DEFAULT_JPEG_QV) -> torch.Tensor:
    """Round-trip an RGB frame through the mjpeg quantization. Fallback path.

    Prefer `quantize_planes`: this has to *make* 4:2:0 chroma by subsampling
    full-resolution RGB, which is a second resample of data that was 4:2:0 in the
    first place. Measured on top of the current cv2 resize it reaches 3.94 uint8
    MAE where the full plane-domain chain reaches 1.13 — worth having when only
    RGB is available (a V4L2 driver that will not hand over its raw buffer, or a
    capture loop that has already converted), not worth choosing.

    Args:
        rgb: `[N, 3, H, W]`, uint8 or float in 0..255.
        qv: ffmpeg's mjpeg quantizer scale, 2..31.

    Returns:
        `[N, 3, H, W]` float in 0..255, rounded and clamped.
    """
    planar = rgb.float()
    matrix = _rgb_to_ycc_matrix(planar.device, planar.dtype)
    ycc = F.conv2d(planar, matrix)
    luma = ycc[:, :1]
    chroma = F.avg_pool2d(ycc[:, 1:] + 128.0, 2)
    luma, chroma = quantize_planes(luma, chroma, qv=qv)
    return yuv420_to_rgb(luma, chroma).round().clamp(0, 255)


@lru_cache(maxsize=8)
def _rgb_to_ycc_matrix(device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """The forward BT.601 matrix, i.e. the inverse of `_YCC_TO_RGB`. Cached."""
    inverse = torch.linalg.inv(torch.tensor(_YCC_TO_RGB, dtype=torch.float64))
    return inverse.to(device=device, dtype=dtype).reshape(3, 3, 1, 1)


def yuv420_to_rgb(luma: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
    """Full-range 4:2:0 planes -> RGB `[N, 3, H, W]`, as libjpeg's decoder does.

    Chroma is upsampled with a triangle filter, which is exactly libjpeg's
    `h2v2_fancy_upsample` and exactly `align_corners=False` bilinear at 2x
    (verified equal; nearest-neighbour instead costs 0.31 MAE). Together with
    the BT.601 matrix this reproduces `cv2.imread` on a real training jpg to
    0.12 uint8 MAE.
    """
    upsampled = F.interpolate(
        chroma, scale_factor=2, mode="bilinear", align_corners=False
    )
    upsampled = upsampled[..., : luma.shape[-2], : luma.shape[-1]]
    ycc = torch.cat([luma, upsampled], dim=1)
    ycc = torch.cat([ycc[:, :1], ycc[:, 1:] - 128.0], dim=1)
    # a 1x1 convolution rather than einsum: same arithmetic, ~10x faster here
    return F.conv2d(ycc, _ycc_to_rgb_matrix(ycc.device, ycc.dtype))


@dataclass(frozen=True)
class Resamplers:
    """The luma/chroma resampler pair for one 4:2:0 geometry.

    A pair rather than two arguments because they are only ever correct together:
    a chroma resampler built for a different target than its luma partner
    silently produces a frame whose colour is offset from its brightness. Build
    with `resamplers_for`.
    """

    luma: Resampler
    chroma: Resampler

    def __call__(
        self, luma: torch.Tensor, chroma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize 4:2:0 planes, chroma at half resolution throughout."""
        return self.luma(luma), self.chroma(chroma)


def to_training_frame(
    luma: torch.Tensor,
    chroma: torch.Tensor,
    *,
    resamplers: Resamplers,
    limited_range: bool,
    qv: int | None = DEFAULT_JPEG_QV,
) -> torch.Tensor:
    """4:2:0 planes at capture resolution -> training-format RGB, uint8.

    The whole fitted chain. Build the resamplers once (geometry is fixed for a
    camera) and call this per frame::

        luma, chroma = uyvy_to_yuv420(frame)
        resamplers = resamplers_for(luma.shape[-2:], (324, 576), device="cuda")
        rgb = to_training_frame(
            luma, chroma, resamplers=resamplers, limited_range=True
        )

    Args:
        luma: `[N, 1, H, W]` float, source range.
        chroma: `[N, 2, H/2, W/2]` float, source range.
        resamplers: built by `resamplers_for` for this source geometry.
        limited_range: whether the source is limited ("tv") range. No default:
            guessing wrong costs more than every other stage combined, and the
            answer comes from the source, not from this module.
        qv: mjpeg quantizer scale, or None to skip the DCT round trip.

    Returns:
        `[N, 3, H_out, W_out]` uint8 RGB.
    """
    luma, chroma = resamplers(luma, chroma)
    if limited_range:
        luma, chroma = expand_range(luma, chroma)
    # the encoder is fed uint8 planes, and rounding here rather than carrying
    # float into the DCT is worth 1.13 vs 1.28 MAE
    luma, chroma = luma.round().clamp(0, 255), chroma.round().clamp(0, 255)
    if qv is not None:
        luma, chroma = quantize_planes(luma, chroma, qv=qv)
    return yuv420_to_rgb(luma, chroma).round().clamp(0, 255).to(torch.uint8)


def resamplers_for(
    luma_hw: Sequence[int],
    out_hw: Sequence[int],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Resamplers:
    """The resampler pair for a 4:2:0 source whose luma plane is `luma_hw`."""
    in_h, in_w = (int(v) for v in luma_hw)
    out_h, out_w = (int(v) for v in out_hw)
    return Resamplers(
        luma=Resampler((in_h, in_w), (out_h, out_w), device=device, dtype=dtype),
        chroma=Resampler(
            (in_h // 2, in_w // 2),
            ((out_h + 1) // 2, (out_w + 1) // 2),
            device=device,
            dtype=dtype,
        ),
    )
