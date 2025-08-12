from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger

from rmind.config import HydraConfig

logger = get_logger(__name__)


@dataclass
class Config:
    model: HydraConfig[LightningModule]
    args: Sequence[Any]
    path: Path


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # pyright: ignore[reportCallIssue]

    logger.debug("instantiating", target=config.model.target)
    args = instantiate(config.args, _recursive_=True, _convert_="all")
    model = config.model.instantiate().eval()
    logger.debug(f"instantiated\n{ModelSummary(model)}")  # noqa: G004

    model(*args)
    logger.debug("torch exporting")
    exported_program = torch.export.export(mod=model, args=tuple(args), strict=True)

    logger.debug("onnx exporting")
    model = torch.onnx.export(
        model=exported_program,
        f=config.path,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=True,
    )

    logger.debug("exported", path=config.path.resolve().as_posix())


if __name__ == "__main__":
    main()
