import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pydantic.dataclasses import dataclass
from pytorch_lightning import LightningModule
from structlog import get_logger

from rmind.config import HydraConfig

logger = get_logger(__name__)


def model_summary(model: torch.nn.Module) -> dict[str, Any]:
    """
    Get comprehensive model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

    size_mb = (param_size + buffer_size) / (1024**2)
    size_gb = size_mb / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": size_mb,
        "size_gb": size_gb,
    }


@dataclass
class Config:
    model: HydraConfig[LightningModule]
    args: Sequence[Any]
    path: Path


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model, _recursive_=True, _convert_="all")
    # model = torch.compile(model, mode="default", fullgraph=True)  # noqa: ERA001
    logger.info("model", target=cfg.model.model_name)
    args = instantiate(cfg.args, _recursive_=True, _convert_="all")
    # batch 1
    images = args[0]["data"]["cam_front_left"][0][0, ...]
    # batch 6
    # images = args[0]["data"]["cam_front_left"][0]  # noqa: ERA001
    logger.info("resolution", resolution=images.shape)
    # result = model(images, images)  # noqa: ERA001
    exported_program = torch.export.export(
        mod=model, args=(images, images), strict=True
    )

    model = torch.onnx.export(
        model=exported_program,
        f=cfg.f,
        artifacts_dir=cfg["artifacts_dir"],
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=True,
        report=True,
        dump_exported_program=True,
    )
    sys.exit(0)

    # config = Config(**OmegaConf.to_container(cfg, resolve=True))  # noqa: ERA001
    # logger.debug("instantiating", target=config.model.target)  # noqa: ERA001
    # args = instantiate(config.args, _recursive_=True, _convert_="all")  # noqa: ERA001
    # model = config.model.instantiate().eval()  # noqa: ERA001
    # logger.debug(f"instantiated\n{ModelSummary(model)}")  # noqa: ERA001

    # model(*args)  # noqa: ERA001
    # logger.debug("torch exporting")  # noqa: ERA001
    # exported_program = torch.export.export(mod=model, args=tuple(args), strict=True)  # noqa: ERA001

    # logger.debug("onnx exporting")  # noqa: ERA001
    # model = torch.onnx.export(  # noqa: ERA001, RUF100
    #     model=exported_program,  # noqa: ERA001
    #     f=config.path,  # noqa: ERA001
    #     artifacts_dir=cfg["artifacts_dir"],  # noqa: ERA001
    #     dynamo=True,  # noqa: ERA001
    #     external_data=False,  # noqa: ERA001
    #     optimize=True,  # noqa: ERA001
    #     verify=True,  # noqa: ERA001
    # )  # noqa: ERA001, RUF100

    # logger.debug("exported", path=config.path.resolve().as_posix())  # noqa: ERA001


if __name__ == "__main__":
    main()
