"""`rmind.utils.frame_parity` vs the real thing, on real drive data.

Every stage of that module is a fit against a specific piece of ffmpeg or
libjpeg behaviour, so every stage gets a test that compares it against *that*
piece rather than against the end of the chain — a single end-to-end number
cannot tell a broken quantizer bias from a broken resize kernel, and the fit
history is a sequence of exactly that confusion being resolved.

`frame_source_parity_report.md` records the measurements these pin. The
tolerances are set from measured agreement with headroom for a different drive
or ffmpeg build; a structural regression (wrong table, round-to-nearest instead
of ffmpeg's 3/8 bias, missing range conversion, chroma resized in RGB) moves
them by 2-6x, well past the margins here.

Set RMIND_BENCHMARK_DRIVE_DIR to point at a drive other than the default.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404 — fixed argv, no shell
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rmind.utils.frame_parity import (
    FFMPEG_MJPEG_INTRA_QUANT_BIAS,
    Resampler,
    expand_range,
    nv12_to_planes,
    quantize_planes,
    resamplers_for,
    to_training_frame,
    uyvy_to_yuv420,
    yuv420_to_rgb,
)
from rmind.utils.jpeg import DEFAULT_JPEG_QV

cv2 = pytest.importorskip("cv2")

NATIVE_HW = (1080, 1920)
TARGET_HW = (324, 576)
FIRST_FRAME = 2910
NUM_FRAMES = 8  # enough to average out per-frame scene variation, quick to decode

# ── measured baselines (see the module docstring of frame_parity) ─────────────
# this module's chain, from the same nv12 planes ffmpeg's own baseline gets
CHAIN_MAE = 1.13
# ffmpeg end-to-end with scale_cuda bicubic + mjpeg q16 — the report's "floor"
FFMPEG_BASELINE_MAE = 1.58
# two adjacent real training frames, i.e. one frame of scene motion
SCENE_MOTION_MAE = 0.64

_DRIVE_DIR_CANDIDATES = (
    Path("/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14"),
    Path.home() / "data" / "Niro122-HQ" / "2023-05-25--09-34-14",
)


def _drive_dir() -> Path | None:
    override = os.environ.get("RMIND_BENCHMARK_DRIVE_DIR")
    candidates = (Path(override),) if override else _DRIVE_DIR_CANDIDATES
    return next((d for d in candidates if d.is_dir()), None)


requires_drive = pytest.mark.skipif(
    _drive_dir() is None,
    reason="needs a real drive dir (set RMIND_BENCHMARK_DRIVE_DIR)",
)
requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="needs ffmpeg on PATH"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device for hwaccel decode"
)

_HWACCEL = ("-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "hevc_cuvid")
# resolved rather than spelled, so every subprocess call below runs an absolute path
_FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# tolerances, all set from measured agreement with headroom — see each test
RESAMPLER_VS_SCALE_CUDA = 1.0  # fitted at 0.51
RANGE_LUMA_TOLERANCE = 0.05  # fitted at 0.00 (exact)
RANGE_CHROMA_TOLERANCE = 1.0  # fitted at ~0.5 (rounding)
QUANTIZER_VS_FFMPEG = 0.6  # fitted at 0.13; round-to-nearest instead measures 1.14
DECODE_VS_LIBJPEG = 0.5  # fitted at 0.12
ABLATION_MARGIN = 1.15  # every dropped stage measured 2-6x worse


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def _select(chain: str) -> str:
    last = FIRST_FRAME + NUM_FRAMES - 1
    return rf"select='between(n\,{FIRST_FRAME}\,{last})'{chain}"


def _raw_frames(
    video: Path, chain: str, hw: tuple[int, int], pix_fmt: str, *, hwaccel: bool = False
) -> list[bytes]:
    """Decode NUM_FRAMES starting at FIRST_FRAME as raw planar bytes."""
    height, width = hw
    command = [
        _FFMPEG, "-v", "error", *(_HWACCEL if hwaccel else ()), "-i", str(video),
        "-filter_complex", _select(chain), "-fps_mode", "passthrough",
        "-frames:v", str(NUM_FRAMES), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]  # fmt: skip
    raw = subprocess.run(command, capture_output=True, check=True).stdout  # noqa: S603
    size = height * width * 3 // 2
    assert len(raw) == size * NUM_FRAMES, (len(raw), size * NUM_FRAMES)
    return [raw[k * size : (k + 1) * size] for k in range(NUM_FRAMES)]


def _decode_jpg_planes(path: Path) -> bytes:
    """A jpg's 4:2:0 planes as libjpeg's decoder produced them, before RGB."""
    return subprocess.run(  # noqa: S603
        [
            _FFMPEG,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuvj420p",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout


def _planar_420(buffer: bytes, hw: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Split fully-planar (I420/yuvj420p) bytes into Y, Cb, Cr."""
    height, width = hw
    luma_size, chroma_size = height * width, height * width // 4
    flat = np.frombuffer(buffer, np.uint8)
    return (
        flat[:luma_size].reshape(height, width),
        flat[luma_size : luma_size + chroma_size].reshape(height // 2, width // 2),
        flat[luma_size + chroma_size :].reshape(height // 2, width // 2),
    )


@pytest.fixture(scope="module")
def video() -> Path:
    drive = _drive_dir()
    assert drive is not None  # guarded by requires_drive
    path = drive / "cam_front_left.pii.mp4"
    if not path.is_file():
        pytest.skip(f"no source video at {path}")
    return path


@pytest.fixture(scope="module")
def training_jpgs() -> list[Path]:
    """The real extract_frames output these frames were trained on."""
    drive = _drive_dir()
    assert drive is not None
    height, width = TARGET_HW
    directory = drive / "frames" / "cam_front_left.pii.mp4" / f"{width}x{height}"
    # extract_frames' image2 muxer numbers frames 1-based, so frame N is file N+1
    paths = [directory / f"{FIRST_FRAME + k + 1:09d}.jpg" for k in range(NUM_FRAMES)]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"no extracted frames under {directory}")
    return paths


@pytest.fixture(scope="module")
def target_rgb(training_jpgs: list[Path]) -> list[np.ndarray]:
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in training_jpgs]


@pytest.fixture(scope="module")
def native_nv12(video: Path) -> list[bytes]:
    """The decoder's own output at capture resolution — this module's input."""
    return _raw_frames(video, ",format=nv12", NATIVE_HW, "nv12")


@pytest.fixture(scope="module")
def scaled_nv12(video: Path) -> list[bytes]:
    """What scale_cuda produces, i.e. ffmpeg's own resize of the same frames."""
    height, width = TARGET_HW
    chain = f",scale_cuda={width}:{height}:interp_algo=bicubic,hwdownload,format=nv12"
    return _raw_frames(video, chain, TARGET_HW, "nv12", hwaccel=True)


@pytest.fixture(scope="module")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── stage 1: the resize kernel ────────────────────────────────────────────────


@requires_drive
@requires_ffmpeg
@requires_cuda
def test_resampler_matches_scale_cuda(
    native_nv12: list[bytes], scaled_nv12: list[bytes], device: str
) -> None:
    """`Resampler` must reproduce scale_cuda's own luma plane.

    Compared plane-for-plane rather than at the end of the chain, so this
    isolates the resampling kernel from the range conversion, the quantizer and
    the colour transform. Catmull-Rom (a = -1/2) measured 0.51 uint8 MAE here
    against `F.interpolate`'s hardcoded a = -3/4 at 0.63 — which is exactly why
    `Resampler` exists rather than a call to `F.interpolate`.
    """
    resampler = Resampler(NATIVE_HW, TARGET_HW, device=device)
    errors = []
    for native, scaled in zip(native_nv12, scaled_nv12, strict=True):
        luma, _ = nv12_to_planes(
            torch.frombuffer(bytearray(native), dtype=torch.uint8).to(device),
            *NATIVE_HW,
        )
        actual = (
            resampler(luma).round().clamp(0, 255)[0, 0].to(torch.uint8).cpu().numpy()
        )
        expected, _ = nv12_to_planes(
            torch.frombuffer(bytearray(scaled), dtype=torch.uint8), *TARGET_HW
        )
        errors.append(_mae(actual, expected[0, 0].to(torch.uint8).numpy()))

    assert np.mean(errors) < RESAMPLER_VS_SCALE_CUDA, (
        f"resampler drifted from scale_cuda: {np.mean(errors)}"
    )


def test_resampler_rejects_a_size_it_was_not_built_for() -> None:
    resampler = Resampler((64, 64), (32, 32))
    with pytest.raises(ValueError, match="expected planes of size"):
        resampler(torch.zeros(1, 1, 48, 48))


# ── stage 2: the limited -> full range conversion ─────────────────────────────


@requires_drive
@requires_ffmpeg
@requires_cuda
def test_expand_range_matches_ffmpegs_nv12_to_yuvj420p(
    video: Path, scaled_nv12: list[bytes], device: str
) -> None:
    """`expand_range` must reproduce the conversion ffmpeg inserts silently.

    ffmpeg converts nv12 (limited) to yuvj420p (full) on the way into the mjpeg
    encoder. Asking it for both pix_fmts off the identical filter chain isolates
    that one step.
    """
    height, width = TARGET_HW
    chain = f",scale_cuda={width}:{height}:interp_algo=bicubic,hwdownload,format=nv12"
    expected_frames = _raw_frames(video, chain, TARGET_HW, "yuvj420p", hwaccel=True)

    luma_errors, chroma_errors = [], []
    for limited, expected in zip(scaled_nv12, expected_frames, strict=True):
        luma, chroma = nv12_to_planes(
            torch.frombuffer(bytearray(limited), dtype=torch.uint8).to(device),
            *TARGET_HW,
        )
        luma, chroma = expand_range(luma, chroma)
        luma = luma.round().clamp(0, 255)[0, 0].to(torch.uint8).cpu().numpy()
        chroma = chroma.round().clamp(0, 255)[0].to(torch.uint8).cpu().numpy()

        want_y, want_cb, want_cr = _planar_420(expected, TARGET_HW)
        luma_errors.append(_mae(luma, want_y))
        chroma_errors.append(_mae(chroma, np.stack([want_cb, want_cr])))

    # luma matched ffmpeg exactly when fitted; chroma to ~0.5 (rounding)
    assert np.mean(luma_errors) < RANGE_LUMA_TOLERANCE, (
        f"luma range drift: {np.mean(luma_errors)}"
    )
    assert np.mean(chroma_errors) < RANGE_CHROMA_TOLERANCE, (
        f"chroma range drift: {np.mean(chroma_errors)}"
    )


# ── stage 3: the mjpeg quantizer ──────────────────────────────────────────────


@requires_drive
@requires_ffmpeg
@requires_cuda
def test_quantize_planes_matches_the_real_mjpeg_encoder(
    video: Path, device: str
) -> None:
    """`quantize_planes` must reproduce ffmpeg's encoder, on identical planes.

    The whole test is in the plane domain: feed the encoder's own input planes
    in, read the encoded jpg's planes back out, and compare. That leaves the
    quantizer as the only variable — no resize, no colour transform.
    """
    height, width = TARGET_HW
    chain = f",scale_cuda={width}:{height}:interp_algo=bicubic,hwdownload,format=nv12"
    source = _raw_frames(video, chain, TARGET_HW, "yuvj420p", hwaccel=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(  # noqa: S603
            [
                _FFMPEG,
                "-v",
                "error",
                "-y",
                *_HWACCEL,
                "-i",
                str(video),
                "-filter_complex",
                _select(chain),
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(NUM_FRAMES),
                "-f",
                "image2",
                "-q:v",
                str(DEFAULT_JPEG_QV),
                str(Path(tmp) / "%09d.jpg"),
            ],
            check=True,
        )
        encoded = [
            _planar_420(_decode_jpg_planes(Path(tmp) / f"{k + 1:09d}.jpg"), TARGET_HW)
            for k in range(NUM_FRAMES)
        ]

    errors = []
    for raw, (want_y, want_cb, want_cr) in zip(source, encoded, strict=True):
        got_y, got_cb, got_cr = _planar_420(raw, TARGET_HW)
        luma = torch.from_numpy(got_y.copy()).to(device).float()[None, None]
        chroma = (
            torch.from_numpy(np.stack([got_cb, got_cr]).copy()).to(device).float()[None]
        )
        luma, chroma = quantize_planes(luma, chroma)
        errors.append(
            _mae(luma[0, 0].to(torch.uint8).cpu().numpy(), want_y)
            + _mae(
                chroma[0].to(torch.uint8).cpu().numpy(), np.stack([want_cb, want_cr])
            )
        )

    # fitted at 0.13 mean per plane; round-to-nearest instead of ffmpeg's 3/8
    # bias measured 1.14, so this margin distinguishes them decisively
    assert np.mean(errors) < QUANTIZER_VS_FFMPEG, (
        f"quantizer drifted from ffmpeg: {np.mean(errors)}"
    )


def test_quantizer_bias_is_ffmpegs_mjpeg_bias() -> None:
    """ffmpeg sets intra_quant_bias = 3 << (QUANT_BIAS_SHIFT - 3) for MJPEG."""
    assert pytest.approx((3 << (8 - 3)) / 256) == FFMPEG_MJPEG_INTRA_QUANT_BIAS


def test_quantize_planes_rejects_half_precision() -> None:
    """fp16 cannot hold a DC coefficient's quantization level."""
    luma = torch.zeros(1, 1, 16, 16, dtype=torch.float16)
    chroma = torch.zeros(1, 2, 8, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="float32 or better"):
        quantize_planes(luma, chroma)


@pytest.mark.parametrize("qv", [0, 1, 32, 50, 95, 100])
def test_quantize_planes_rejects_a_libjpeg_quality(qv: int) -> None:
    """A 0-100 libjpeg quality must be refused, not silently accepted.

    This is the bug the report documents: `-q:v 16` read as a libjpeg quality is
    the single worst point in the whole 0-100 sweep.
    """
    luma, chroma = torch.zeros(1, 1, 16, 16), torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match=r"range 2\.\.31"):
        quantize_planes(luma, chroma, qv=qv)


# ── stage 4: the decoder's chroma upsample and colour transform ───────────────


@requires_drive
@requires_ffmpeg
def test_yuv420_to_rgb_matches_libjpegs_decoder(
    training_jpgs: list[Path], target_rgb: list[np.ndarray], device: str
) -> None:
    """`yuv420_to_rgb` must reproduce `cv2.imread` on the identical file.

    Reading one real training jpg's 4:2:0 planes and converting them here, then
    comparing against libjpeg's own decode of that same file, leaves the
    triangle upsample and the BT.601 matrix as the only variables.
    """
    errors = []
    for path, expected in zip(training_jpgs, target_rgb, strict=True):
        raw = subprocess.run(  # noqa: S603
            [
                _FFMPEG,
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuvj420p",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
        y, cb, cr = _planar_420(raw, TARGET_HW)
        luma = torch.from_numpy(y.copy()).to(device).float()[None, None]
        chroma = torch.from_numpy(np.stack([cb, cr]).copy()).to(device).float()[None]
        actual = yuv420_to_rgb(luma, chroma).round().clamp(0, 255)
        actual = actual[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        errors.append(_mae(actual, expected))

    assert np.mean(errors) < DECODE_VS_LIBJPEG, (
        f"decode tail drifted from libjpeg: {np.mean(errors)}"
    )


# ── the whole chain, and the ablations that show each stage earns its place ───


@requires_drive
@requires_ffmpeg
@requires_cuda
def test_chain_beats_the_ffmpeg_baseline(
    native_nv12: list[bytes], target_rgb: list[np.ndarray], device: str
) -> None:
    """End to end, from the decoder's planes to the training jpg.

    The claim `frame_source_parity_report.md` needs to keep being true: this
    module gets closer to the real training frames than re-running
    extract_frames' own ffmpeg command does, because its Catmull-Rom is a better
    match for `scale_npp` (what training used) than `scale_cuda` is.
    """
    resamplers = resamplers_for(NATIVE_HW, TARGET_HW, device=device)
    errors = []
    for native, expected in zip(native_nv12, target_rgb, strict=True):
        luma, chroma = nv12_to_planes(
            torch.frombuffer(bytearray(native), dtype=torch.uint8).to(device),
            *NATIVE_HW,
        )
        rgb = to_training_frame(luma, chroma, resamplers=resamplers, limited_range=True)
        errors.append(_mae(rgb[0].permute(1, 2, 0).cpu().numpy(), expected))

    mae = float(np.mean(errors))
    assert mae < FFMPEG_BASELINE_MAE, f"chain regressed past ffmpeg's baseline: {mae}"
    assert mae > SCENE_MOTION_MAE * 0.5, (
        f"MAE of {mae} is below half a frame of scene motion — suspect the "
        "comparison is against itself rather than against the training jpgs"
    )


@requires_drive
@requires_ffmpeg
@requires_cuda
@pytest.mark.parametrize("ablation", ["no_quantize", "no_range", "antialiased_resize"])
def test_every_stage_earns_its_place(
    native_nv12: list[bytes], target_rgb: list[np.ndarray], device: str, ablation: str
) -> None:
    """Dropping any fitted stage must make the result measurably worse.

    Without this, a stage could be silently broken — or silently removed as a
    "simplification" — and the single end-to-end number would be the only
    warning. Measured during the fit: no_quantize 3.87, antialiased_resize 3.19,
    no_range 6.88, against the full chain's 1.13.
    """
    resamplers = resamplers_for(NATIVE_HW, TARGET_HW, device=device)

    errors = []
    for native, expected in zip(native_nv12, target_rgb, strict=True):
        luma, chroma = nv12_to_planes(
            torch.frombuffer(bytearray(native), dtype=torch.uint8).to(device),
            *NATIVE_HW,
        )
        if ablation == "antialiased_resize":
            # `F.interpolate`'s antialias=True scales the filter support with
            # the downscale ratio — a proper prefilter, and the right thing in
            # general. It is wrong *here* because scale_npp does not do it.
            height, width = TARGET_HW
            luma = torch.nn.functional.interpolate(
                luma,
                size=TARGET_HW,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            chroma = torch.nn.functional.interpolate(
                chroma,
                size=(height // 2, width // 2),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            luma, chroma = expand_range(luma, chroma)
            luma, chroma = luma.round().clamp(0, 255), chroma.round().clamp(0, 255)
            luma, chroma = quantize_planes(luma, chroma)
            rgb = yuv420_to_rgb(luma, chroma).round().clamp(0, 255).to(torch.uint8)
        else:
            rgb = to_training_frame(
                luma,
                chroma,
                resamplers=resamplers,
                limited_range=ablation != "no_range",
                qv=None if ablation == "no_quantize" else DEFAULT_JPEG_QV,
            )
        errors.append(_mae(rgb[0].permute(1, 2, 0).cpu().numpy(), expected))

    assert float(np.mean(errors)) > CHAIN_MAE * ABLATION_MARGIN, (
        f"ablation {ablation!r} did not degrade the result — that stage is "
        "either not doing anything or the full chain has regressed to match it"
    )


# ── the UYVY entry point drivr's live camera uses ─────────────────────────────


def test_uyvy_to_yuv420_deinterleaves_the_documented_byte_order() -> None:
    """UYVY is U Y0 V Y1: luma in byte 1, chroma alternating U/V in byte 0.

    A swapped Cb/Cr or a mis-strided deinterleave is the classic way this fails,
    and on a live camera it shows up as a uniformly green frame rather than as a
    number — so it is worth pinning against a hand-built buffer.
    """
    # two pixels per chroma pair, two rows so the 4:2:2 -> 4:2:0 average is real
    uyvy = torch.tensor(
        [
            [[10, 100], [200, 110], [20, 120], [210, 130]],
            [[30, 140], [220, 150], [40, 160], [230, 170]],
        ],
        dtype=torch.uint8,
    )
    luma, chroma = uyvy_to_yuv420(uyvy)

    assert luma.shape == (1, 1, 2, 4)
    torch.testing.assert_close(
        luma[0, 0], torch.tensor([[100.0, 110, 120, 130], [140.0, 150, 160, 170]])
    )
    assert chroma.shape == (1, 2, 1, 2)
    # Cb from the even columns (10, 20 / 30, 40), averaged down the two rows
    torch.testing.assert_close(chroma[0, 0], torch.tensor([[20.0, 30.0]]))
    # Cr from the odd columns (200, 210 / 220, 230)
    torch.testing.assert_close(chroma[0, 1], torch.tensor([[210.0, 220.0]]))


@pytest.mark.parametrize(
    "shape", [(4, 4), (4, 4, 3), (4, 4, 1)], ids=["2d", "three_channel", "one_channel"]
)
def test_uyvy_to_yuv420_rejects_a_buffer_that_is_not_packed_uyvy(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="packed UYVY"):
        uyvy_to_yuv420(torch.zeros(shape, dtype=torch.uint8))


def test_uyvy_to_yuv420_rejects_odd_dimensions() -> None:
    with pytest.raises(ValueError, match="even dimensions"):
        uyvy_to_yuv420(torch.zeros((5, 4, 2), dtype=torch.uint8))


def test_nv12_to_planes_rejects_a_wrongly_sized_buffer() -> None:
    with pytest.raises(ValueError, match="expected 6 NV12 bytes"):
        nv12_to_planes(torch.zeros(8, dtype=torch.uint8), 2, 2)
