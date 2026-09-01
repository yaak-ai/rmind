"""Convert a finished run's checkpoint into a warm-start for a DIFFERENT
architecture (currently: 1-camera -> 3-camera `CausalFrameTransformer` arms).

`PatchPolicy` has no camera-specific parameters -- every camera runs through
the same frozen `image_encoder` and the same `patch_projection`
(`PatchPolicy._frame_tokens`) -- so widening the camera count changes exactly
one tensor's shape: `encoder.intra_position_embedding.weight`, from
`(tokens_per_frame_src, dim_model)` to `(tokens_per_frame_dst, dim_model)`.
Everything else transfers verbatim. This script downloads the source
artifact, tiles that one tensor (`tile_intra_position_embedding`), loads the
result `strict=True` into a model instantiated from the TARGET experiment
config, and writes a checkpoint that is a full-resume ("loops"-shaped) file
starting fresh at epoch 0 / global_step 0, with a fresh optimizer and the
target config's own LR schedule:

    rmind-warm-start-ckpt \
        --artifact yaak/rmind/model-do8m9ot8:v2 \
        --experiment yaak/patch_policy/dinov2_dinowm_causal_3cam \
        --out /abs/path/do8m9ot8_v2_3cam.ckpt

Then:

    just train-unsafe \
        experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam \
        ckpt_path=/abs/path/do8m9ot8_v2_3cam.ckpt

This is data, not code: the output checkpoint deliberately omits `"loops"`
and sets `"optimizer_states"`/`"lr_schedulers"` to `[]` (present but empty),
which is the one shape PyTorch Lightning's `ckpt_path` full-resume accepts as
"fresh run, pretrained weights" --
`restore_optimizers_and_schedulers` KeyErrors if either key is absent
entirely, but zips against an empty list restore nothing when they are
present and empty; `restore_loops` is skipped outright when `"loops"` is
missing. Dropping `ModelCheckpoint`'s callback state is deliberate too:
keeping it would carry over the source run's `best_model_score` and suppress
checkpoint saving in the new run.

`--experiment` must resolve to an already-generated config (`just
generate-config`), and instantiating the target model downloads the frozen
DINOv2 backbone and the pinned tokenizer artifacts, so this step needs
network.
"""

import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException
from hydra.utils import instantiate
from structlog import get_logger
from torch import Tensor

import rmind  # noqa: F401  # registers the `eval` OmegaConf resolver
from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.models.patch_policy import PatchPolicy
from rmind.utils._wandb import LoadableFromArtifact

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
INTRA_POSITION_KEY = "encoder.intra_position_embedding.weight"


def tile_intra_position_embedding(src: Tensor, *, tokens_per_frame: int) -> Tensor:
    """Widen a `(tokens_per_frame_src, dim_model)` intra-frame position table to
    `tokens_per_frame` by tiling the per-camera patch rows and keeping the speed
    row (slot 0) fixed.

    `PatchPolicy._frame_tokens` lays a frame out as
    `[speed, cam_0 patches, cam_1 patches, ...]`, so tiling the patch block onto
    each new camera slot -- rather than randomly initializing it -- means the
    LAST row (the readout, `patch_policy.py`'s
    `rearrange(...)[:, :, -1]`) keeps carrying the trained readout role instead
    of landing on an untrained row: `new[-1] == src[-1]` for any tiling factor.

    Raises:
        ValueError: if `tokens_per_frame - 1` isn't a positive multiple of
            `src`'s patches-per-frame.
    """
    patches = src.shape[0] - 1
    target = tokens_per_frame - 1
    if target < patches or target % patches != 0:
        msg = (
            f"tokens_per_frame - 1 ({target}) must be a positive multiple of "
            f"the source's patches-per-frame ({patches})"
        )
        raise ValueError(msg)
    return torch.cat([src[:1], src[1:].repeat(target // patches, 1)], dim=0)


def _compose_target_model(experiment: str) -> PatchPolicy:
    try:
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name="train", overrides=[f"experiment={experiment}"])
    except MissingConfigException as e:
        msg = (
            "config compose failed -- have you run `just generate-config` "
            "(ytt) in this checkout? it generates config/dataset/yaak/*_3cam.yaml "
            "and friends, which `compose` needs even though this script never "
            "touches the datamodule"
        )
        raise RuntimeError(msg) from e
    return instantiate(cfg.model)


def _remap_state_dict(
    state_dict: dict[str, Tensor], *, tokens_per_frame: int
) -> dict[str, Tensor]:
    remapped = dict(state_dict)
    src = state_dict.get(INTRA_POSITION_KEY)
    if src is not None and src.shape[0] != tokens_per_frame:
        remapped[INTRA_POSITION_KEY] = tile_intra_position_embedding(
            src, tokens_per_frame=tokens_per_frame
        )
    return remapped


def _backfill_preprocessing_buffers(
    model: PatchPolicy, remapped: dict[str, Tensor]
) -> dict[str, Tensor]:
    """Fill `input_transform.*` buffers absent from the source checkpoint with
    the freshly-instantiated target's own values.

    `input_transform` is a fixed preprocessing pipeline (Rearrange/CenterCrop/
    Resize/ToDtype/Normalize) with no trained parameters, ever -- any buffer
    under it is a constant declared straight in the experiment config (e.g.
    `rmind.components.norm.ImageNormalize`'s `mean`/`std`, the ImageNet
    constants). `ImageNormalize` pre-registers them as buffers so they move to
    device once instead of being rebuilt every forward (see its docstring);
    that postdates some source runs' commits, whose `input_transform` used the
    plain, stateless `torchvision.transforms.v2.Normalize` and so has NO
    `input_transform.*` keys in its checkpoint at all. Backfilling from the
    target is exact, not an approximation -- both experiments declare the same
    constants -- and is scoped to `input_transform.` buffers only, so any
    OTHER missing key (a real architecture/weight drift) still fails the
    `strict=True` load below.
    """
    buffer_names = set(dict(model.named_buffers()))
    filled = dict(remapped)
    added = []
    for key, value in model.state_dict().items():
        if (
            key not in filled
            and key.startswith("input_transform.")
            and key in buffer_names
        ):
            filled[key] = value
            added.append(key)
    if added:
        logger.warning(
            "backfilling preprocessing buffers absent from the source "
            "checkpoint (pre-ImageNormalize source run -- config constants, "
            "not trained state)",
            keys=added,
        )
    return filled


def _self_check(out: Path, *, tokens_per_frame: int) -> None:
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    expected_keys = {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "pytorch-lightning_version",
        "hyper_parameters",
    }
    if set(ckpt) != expected_keys:
        msg = f"expected exactly {sorted(expected_keys)}, got {sorted(ckpt)}"
        raise AssertionError(msg)
    if "loops" in ckpt:
        msg = "'loops' must be absent -- its presence would resume mid-schedule"
        raise AssertionError(msg)
    shape = ckpt["state_dict"][INTRA_POSITION_KEY].shape
    if shape != (tokens_per_frame, shape[1]):
        msg = f"{INTRA_POSITION_KEY} has shape {shape}, expected first dim {tokens_per_frame}"
        raise AssertionError(msg)


def convert(*, artifact: str, experiment: str, out: Path) -> None:
    ckpt_path = LoadableFromArtifact.download_wandb_artifact(artifact)
    logger.info("downloaded source artifact", artifact=artifact, path=str(ckpt_path))
    source = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state_dict: dict[str, Tensor] = source["state_dict"]

    model = _compose_target_model(experiment)
    trunk = model.encoder
    if not isinstance(trunk, CausalFrameTransformer):
        msg = (
            f"target experiment {experiment!r}'s encoder is a "
            f"{type(trunk).__name__}, not a CausalFrameTransformer -- this "
            "conversion only handles the decoder-only (causal) arm's "
            "tokens_per_frame-keyed positional embedding"
        )
        raise TypeError(msg)
    factorization = getattr(trunk, "intra_position_factorization", "flat")
    if factorization != "flat":
        msg = (
            f"target experiment {experiment!r} uses "
            f"intra_position_factorization={factorization!r}: there is no "
            f"{INTRA_POSITION_KEY} to tile, and the factorized arms train from "
            "scratch by decision -- see task-intra-position-factorization.md §0 "
            "('Clean re-init'). Warm-starting one would need an ANOVA-style "
            "decomposition of the trained flat table, which is deliberately not "
            "built. Use a `flat` target, or train the factorized arm from scratch"
        )
        raise ValueError(msg)

    tokens_per_frame = trunk.tokens_per_frame
    # `optimizer_states: []` below already forces a fresh optimizer, which is
    # what makes this safe: SelectiveAdamW builds its param groups from SORTED
    # name sets and torch maps saved state onto them POSITIONALLY, so resuming
    # optimizer state across any change to the position parameters' names or
    # count silently mis-assigns Adam moments.
    logger.warning(
        "the output checkpoint carries NO optimizer state, by design -- never "
        "`ckpt_path`-full-resume across a change to the intra-frame position "
        "parameterization (SelectiveAdamW maps optimizer state positionally)"
    )
    remapped = _remap_state_dict(source_state_dict, tokens_per_frame=tokens_per_frame)
    remapped = _backfill_preprocessing_buffers(model, remapped)

    src_shape = source_state_dict[INTRA_POSITION_KEY].shape
    dst_shape = remapped[INTRA_POSITION_KEY].shape
    model.load_state_dict(remapped, strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "state_dict": remapped,
        "optimizer_states": [],
        "lr_schedulers": [],
        "pytorch-lightning_version": pl.__version__,
        "hyper_parameters": dict(model.hparams),
    }
    torch.save(checkpoint, out)
    logger.info(
        "wrote warm-start checkpoint",
        path=str(out),
        mb=round(out.stat().st_size / 1e6, 1),
        intra_position_embedding_shape={
            "src": tuple(src_shape),
            "dst": tuple(dst_shape),
        },
        tensor_count=len(remapped),
    )

    _self_check(out, tokens_per_frame=tokens_per_frame)
    logger.info("self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        required=True,
        help="source wandb model artifact, e.g. yaak/rmind/model-<id>:v0",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="target hydra experiment, e.g. yaak/patch_policy/dinov2_dinowm_causal_3cam",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    convert(artifact=args.artifact, experiment=args.experiment, out=args.out)


if __name__ == "__main__":
    main()
