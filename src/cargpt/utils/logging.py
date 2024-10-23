import logging  # noqa: A005
import sys
from typing import Annotated

import matplotlib as mpl
import wandb
from jaxtyping import Shaped
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import Tensor


def setup_logging() -> None:
    logging.getLogger("xformers").setLevel(logging.ERROR)

    _ = logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message} | {extra}</level>",
                "colorize": True,
            }
        ]
    )


@rank_zero_only
def log_depth_images(
    logger,
    global_step: int,
    *,
    prefix: str = "",
    captions: Annotated[list[str], 2],
    input: Shaped[Tensor, "2 3 h w"],
    warped: Shaped[Tensor, "2 3 h w"],
    auto_mask: Shaped[Tensor, "1 h w"],
    self_mask: Shaped[Tensor, "1 h w"],
    disparity: Shaped[Tensor, "2 h w"],
    projected_disparity: Shaped[Tensor, "1 h w"],
    computed_disparity: Shaped[Tensor, "1 h w"],
) -> None:
    assert isinstance(logger, WandbLogger)

    disparity_colormap = mpl.cm.get_cmap(name="pink", lut=10000)

    def colorize(tensor):
        return disparity_colormap(tensor.cpu().numpy())

    disparity_cm = colorize(disparity / disparity.amax((1, 2), keepdim=True))
    projected_disparity_cm = colorize(
        projected_disparity / projected_disparity.amax((1, 2), keepdim=True)
    )
    computed_disparity_cm = colorize(computed_disparity / disparity[1].max())

    ref_captions = [captions[0]]
    warped_captions = [f"{captions[1]} [gt]", f"{captions[1]} [warped]"]
    data = {}

    data = {
        "input": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(input, captions, strict=True)
        ],
        "warped": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(warped, warped_captions, strict=True)
        ],
        "auto_mask": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(auto_mask, ref_captions, strict=True)
        ],
        "self_mask": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(self_mask, ref_captions, strict=True)
        ],
        "disparity": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(disparity_cm, captions, strict=True)
        ],
        "projected_disparity": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(projected_disparity_cm, ref_captions, strict=True)
        ],
        "computed_disparity": [
            wandb.Image(img, caption=caption)
            for (img, caption) in zip(computed_disparity_cm, ref_captions, strict=True)
        ],
    }

    logger.experiment.log(
        data={f"{prefix}/{k}": v for k, v in data.items()}, step=global_step
    )
