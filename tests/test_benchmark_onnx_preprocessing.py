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

from pathlib import Path
from typing import cast

import hydra
import jq  # ty:ignore[unresolved-import]
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
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
