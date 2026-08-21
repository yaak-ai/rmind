"""Verify benchmark_onnx.py's image preprocessing matches the real training pipeline.

Pipeline under test, for the image branch only:
    1. a raw frame at training's native JPEG resolution (config/dataset/yaak/
       train.yaml:10 decodes frames pre-extracted at 576x324)
    2. the FULL episode_builder.input_transform, built from
       config/experiment/yaak/control_transformer/pretrain.yaml — this is what
       training actually runs the raw frame through (Rearrange -> CenterCrop ->
       Resize -> ToDtype -> Normalize)
    3. the CUT episode_builder.input_transform, built by applying
       config/export/onnx.yaml's hparams_jq (the actual hparams_jq
       export_onnx.py uses to produce the ONNX file) to the same pretrain
       hparams — this is what the exported/benchmarked model actually runs
       the raw frame through (just a Remapper; the image branch is deleted)
    4. benchmark_onnx.py's own _preprocess_image, run on the same raw frame

We expect (2) == (3 then 4): the cut model's Remapper passes the raw image
through untouched (step 3), so benchmark_onnx.py must externally reproduce
everything training's real pipeline (step 2) used to do to it (step 4).
"""

import os
from pathlib import Path
from typing import cast

import cv2
import hydra
import jq  # ty:ignore[unresolved-import]
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image, JpegImagePlugin
from torch.testing import assert_close

from rmind.scripts import benchmark_onnx

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
NATIVE_IMAGE_SIZE = (324, 576)  # (H, W) — training's fixed JPEG frame resolution
FINAL_IMAGE_SIZE = (256, 256)  # (H, W) — what the exported ONNX model expects


@pytest.fixture(scope="module")
def pretrain_hparams() -> dict:
    """Step 2's source: the full, unstripped architecture training runs."""
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=yaak/control_transformer/pretrain"],
        )
    return cast("dict", OmegaConf.to_container(cfg.model, resolve=True))


@pytest.fixture(scope="module")
def cut_hparams(pretrain_hparams: dict) -> dict:
    """Step 3's source: pretrain hparams patched by export/onnx.yaml's
    hparams_jq — the actual jq program export_onnx.py applies to produce the
    ONNX file benchmark_onnx.py loads.
    """
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="export/onnx")
    jq_program = str(cfg.model.hparams_jq)
    return jq.compile(jq_program).input_value(pretrain_hparams).first()


def test_export_hparams_jq_deletes_image_preprocessing(cut_hparams: dict) -> None:
    """Confirms export/onnx.yaml's hparams_jq is what actually removes image
    preprocessing from the model graph — i.e. why benchmark_onnx.py must
    reproduce it externally in the first place.
    """
    args = cut_hparams["episode_builder"]["input_transform"]["_args_"]
    assert len(args) == 1, "expected the ModuleDict (_args_[1]) to be deleted"


@pytest.fixture(scope="module")
def full_image_transform(pretrain_hparams: dict) -> torch.nn.Module:
    """Step 2: the real image preprocessing chain training runs."""
    episode_builder_cfg = OmegaConf.create(pretrain_hparams["episode_builder"])
    episode_builder = hydra.utils.instantiate(
        episode_builder_cfg, _recursive_=True, _convert_="all"
    )
    return episode_builder.input_transform[1].get("image").eval()


@pytest.fixture(scope="module")
def cut_input_transform(cut_hparams: dict) -> torch.nn.Module:
    """Step 3: the exported/benchmarked model's own input_transform — just the
    Remapper, since the image ModuleDict was deleted (see test above).
    """
    episode_builder_cfg = OmegaConf.create(cut_hparams["episode_builder"])
    episode_builder = hydra.utils.instantiate(
        episode_builder_cfg, _recursive_=True, _convert_="all"
    )
    return episode_builder.input_transform.eval()


@pytest.fixture
def native_frame() -> np.ndarray:
    """A random HWC uint8 frame at training's native resolution."""
    rng = np.random.default_rng(0)
    h, w = NATIVE_IMAGE_SIZE
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_cut_input_transform_passes_image_through_unchanged(
    cut_input_transform: torch.nn.Module, native_frame: np.ndarray
) -> None:
    """Confirms step 3 is a no-op on image data — i.e. benchmark_onnx.py's
    _preprocess_image (step 4) is the ONLY thing standing in for training's
    real crop/resize/normalize chain (step 2) in the exported/benchmarked model.
    """
    raw = torch.from_numpy(native_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 3]
    with torch.inference_mode():
        out = cut_input_transform({"data": {"cam_front_left": raw}})
    assert torch.equal(out["image"]["cam_front_left"], raw)


def test_benchmark_preprocessing_matches_training(
    full_image_transform: torch.nn.Module,
    cut_input_transform: torch.nn.Module,
    native_frame: np.ndarray,
) -> None:
    """(2) == (3 then 4): benchmark_onnx.py's preprocessing, applied after the
    cut model's own (no-op) input_transform, should reproduce what training's
    full input_transform does to the same raw frame.

    _preprocess_image must use the torchvision.transforms.v2 *classes*
    (Resize/CenterCrop), not the .functional module — the two aren't
    numerically identical even with matching interpolation/antialias args,
    which used to leave a ~1/255-per-pixel rounding residual (~0.017 after
    ImageNet normalize) large enough to fail this at any reasonably tight
    tolerance. With the classes, only floating-point noise remains.
    """
    raw = torch.from_numpy(native_frame).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 3]
    with torch.inference_mode():
        expected = full_image_transform(raw)  # step 2
        cut_image = cut_input_transform({"data": {"cam_front_left": raw}})["image"][
            "cam_front_left"
        ]  # step 3

    actual_chw = benchmark_onnx._preprocess_image(  # noqa: SLF001 — step 4
        cut_image.squeeze(0).squeeze(0).numpy(), FINAL_IMAGE_SIZE
    )  # CHW float32, already normalized
    actual = actual_chw[np.newaxis, np.newaxis]  # [1, 1, 3, H, W]

    assert_close(torch.from_numpy(actual), expected, atol=2e-5, rtol=0)


# ── "video_jpeg" source: extract_frames' mjpeg quantization ───────────────────
#
# These assert against real extract_frames output, which is the only ground
# truth for the quantization tables (no synthetic fixture can stand in for
# "what ffmpeg -q:v 16 actually wrote"). Point RMIND_BENCHMARK_DRIVE_DIR at a
# drive directory to override the search.

DCT_BLOCK_COEFFICIENTS = 64
PINNED_DC_QUANTIZER = 8  # ffmpeg never scales the DC term
# the "ffmpeg" baseline measured 1.59 uint8 MAE against real training frames on
# Niro122 2023-05-25; this leaves headroom for a different drive or ffmpeg build
MAX_BASELINE_MAE = 2.5

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


@pytest.fixture(scope="module")
def training_jpgs() -> list[Path]:
    """Real extract_frames output: <drive>/frames/<video>/<W>x<H>/*.jpg."""
    drive = _drive_dir()
    assert drive is not None  # guarded by requires_drive
    h, w = benchmark_onnx.DEFAULT_IMAGE_SIZE
    frames = sorted(
        (drive / "frames" / "cam_front_left.pii.mp4" / f"{w}x{h}").glob("*.jpg")
    )
    if not frames:
        pytest.skip(f"no extracted frames under {drive}")
    return frames


@requires_drive
def test_video_jpeg_uses_extract_frames_quantization(training_jpgs: list[Path]) -> None:
    """_simulate_offline_jpeg's tables must be extract_frames' own tables.

    This is the parity the module comment always claimed but the
    implementation did not have: `-q:v 16` used to be passed straight to
    simplejpeg as a libjpeg 0-100 *quality*, which produces a completely
    different table — the worst match in the whole 0-100 sweep — so
    "video_jpeg" over-compressed badly. Pillow's `quantization` readback is
    already in the same natural order `qtables=` takes, so this compares the
    exact list _simulate_offline_jpeg encodes with.
    """
    expected = benchmark_onnx._ffmpeg_mjpeg_qtable(benchmark_onnx._DEFAULT_JPEG_QV)  # noqa: SLF001
    for path in training_jpgs[:5]:
        with Image.open(path) as im:
            tables = cast(
                "dict[int, list[int]]",
                im.quantization,  # ty:ignore[unresolved-attribute]
            )
            assert len(tables) == 1, (
                f"{path.name}: extract_frames emits one table shared by all "
                f"components, got {len(tables)}"
            )
            assert list(tables[0]) == expected, f"DQT mismatch in {path.name}"
            assert (
                JpegImagePlugin.get_sampling(im) == benchmark_onnx._JPEG_SUBSAMPLING_420  # noqa: SLF001
            ), f"{path.name}: extract_frames writes yuvj420p"


@pytest.mark.parametrize("qv", [2, 8, 16, 24, 31])
def test_ffmpeg_mjpeg_qtable_scaling(qv: int) -> None:
    """q[0] pinned at 8, the rest scaled from ff_mpeg1_default_intra_matrix.

    Verified byte-identical to `ffmpeg -q:v N -pix_fmt yuvj420p` for every N
    in 2..31; this locks in the shape of the rule without shelling out.
    """
    table = benchmark_onnx._ffmpeg_mjpeg_qtable(qv)  # noqa: SLF001
    base = benchmark_onnx._FFMPEG_MJPEG_BASE_QTABLE  # noqa: SLF001
    assert len(table) == DCT_BLOCK_COEFFICIENTS
    assert table[0] == PINNED_DC_QUANTIZER
    assert table[1:] == [min(max(b * qv // 8, 1), 255) for b in base[1:]]


@pytest.mark.parametrize("qv", [1, 0, 32, 95])
def test_ffmpeg_mjpeg_qtable_rejects_libjpeg_style_quality(qv: int) -> None:
    """A libjpeg-style 0-100 quality must not be silently accepted as -q:v.

    Passing one through as if it were ffmpeg's scale is precisely the bug this
    replaced, so values outside 2..31 are a hard error.
    """
    with pytest.raises(ValueError, match="jpeg_qv"):
        benchmark_onnx._ffmpeg_mjpeg_qtable(qv)  # noqa: SLF001


@requires_drive
def test_video_jpeg_moves_frames_toward_training_data(
    training_jpgs: list[Path],
) -> None:
    """The round-trip must close the gap to real extract_frames output.

    Direction is the whole point and is what the libjpeg-quality bug
    inverted: over-compressing put "video_jpeg" *further* from the training
    frames (5.59 vs 4.31 uint8 MAE) than doing nothing at all. Measured at
    model-input level, so it covers what actually reaches the network.
    """
    drive = _drive_dir()
    assert drive is not None
    capture = cv2.VideoCapture(str(drive / "cam_front_left.pii.mp4"))
    if not capture.isOpened():
        pytest.skip(f"cannot open video under {drive}")

    frame_indices = [2910, 2970, 3030, 3090, 3150]
    try:
        decoded = []
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok:
                pytest.skip(f"video too short for frame {index}")
            decoded.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    def preprocess(image: np.ndarray, *, jpeg_qv: int | None = None) -> np.ndarray:
        return benchmark_onnx._preprocess_image(  # noqa: SLF001
            image, FINAL_IMAGE_SIZE, jpeg_qv=jpeg_qv
        )

    # the real extracted jpg is the reference every source is chasing; resolve
    # it the way _JpgFrameSource does rather than by position in the listing
    frames_dir = training_jpgs[0].parent
    references = []
    for index in frame_indices:
        name = benchmark_onnx._JPG_FILENAME_PATTERN.format(  # noqa: SLF001
            idx=index + benchmark_onnx._JPG_INDEX_OFFSET  # noqa: SLF001
        )
        image_bgr = cv2.imread(str(frames_dir / name))
        if image_bgr is None:
            pytest.skip(f"missing extracted frame {name}")
        references.append(preprocess(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))

    def gap(images: list[np.ndarray]) -> float:
        return float(
            np.mean([
                np.abs(a - b).mean() for a, b in zip(images, references, strict=True)
            ])
        )

    without = gap([preprocess(f) for f in decoded])
    with_jpeg = gap([
        preprocess(f, jpeg_qv=benchmark_onnx._DEFAULT_JPEG_QV)  # noqa: SLF001
        for f in decoded
    ])

    # measured ~11% closer at 224x224 and ~14% at uint8 576x324; require a
    # clear improvement without pinning the exact figure
    assert with_jpeg < without * 0.95, (
        f"jpeg round-trip did not move frames toward training data: "
        f"{without:.5f} -> {with_jpeg:.5f}"
    )


def _read_training_frames(
    frames_dir: Path, frame_indices: list[int]
) -> list[np.ndarray]:
    """Real extract_frames output for the given video frame indices."""
    frames = []
    for index in frame_indices:
        name = benchmark_onnx._JPG_FILENAME_PATTERN.format(  # noqa: SLF001
            idx=index + benchmark_onnx._JPG_INDEX_OFFSET  # noqa: SLF001
        )
        image_bgr = cv2.imread(str(frames_dir / name))
        if image_bgr is None:
            pytest.skip(f"missing extracted frame {name}")
        frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return frames


@requires_drive
def test_ffmpeg_source_is_the_closest_to_the_training_frames(
    training_jpgs: list[Path],
) -> None:
    """The "ffmpeg" source is the baseline, so nothing may beat it.

    It re-runs extract_frames' own command (NVDEC decode, GPU resize in NV12,
    real mjpeg), substituting only scale_cuda for the unavailable scale_npp,
    so its distance to the real training jpgs is the floor every other source
    is approximating. Asserts the full ordering — ffmpeg < video_jpeg < video —
    which is what makes each source's extra fidelity step worth having.
    """
    drive = _drive_dir()
    assert drive is not None
    video_path = drive / "cam_front_left.pii.mp4"
    if not video_path.exists():
        pytest.skip(f"no video under {drive}")

    frame_indices = [2910, 2940, 2970, 3000]
    height, width = benchmark_onnx.DEFAULT_IMAGE_SIZE

    try:
        source = benchmark_onnx._FfmpegFrameSource(video_path)  # noqa: SLF001
        by_ffmpeg = [source.read(i) for i in frame_indices]
    except (RuntimeError, ValueError) as e:  # no ffmpeg, or no CUDA device
        pytest.skip(f"ffmpeg frame source unavailable: {e}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        pytest.skip(f"cannot open {video_path}")
    try:
        decoded = []
        for index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok:
                pytest.skip(f"video too short for frame {index}")
            decoded.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    references = _read_training_frames(training_jpgs[0].parent, frame_indices)

    def mae(images: list[np.ndarray]) -> float:
        return float(
            np.mean([
                np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
                for a, b in zip(images, references, strict=True)
            ])
        )

    downscaled = [
        cv2.resize(f, (width, height), interpolation=cv2.INTER_CUBIC) for f in decoded
    ]
    gaps = {
        "ffmpeg": mae(by_ffmpeg),
        "video_jpeg": mae([
            benchmark_onnx._simulate_offline_jpeg(  # noqa: SLF001
                f,
                qv=benchmark_onnx._DEFAULT_JPEG_QV,  # noqa: SLF001
            )
            for f in downscaled
        ]),
        "video": mae(downscaled),
    }

    assert gaps["ffmpeg"] < gaps["video_jpeg"] < gaps["video"], (
        f"baseline ordering broken: {gaps}"
    )
    # measured ~1.59 / 3.80 / 4.42 on Niro122 2023-05-25; two adjacent real
    # training frames differ by ~1.34, so the floor is near one frame of motion
    assert gaps["ffmpeg"] < MAX_BASELINE_MAE, (
        f"ffmpeg baseline regressed: {gaps['ffmpeg']:.3f}"
    )
