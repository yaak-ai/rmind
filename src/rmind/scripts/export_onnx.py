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
    print(model_summary(model))  # noqa: T201
    logger.info("model", target=cfg.model.model.model_name)
    images = instantiate(cfg.input, _recursive_=True, _convert_="all")
    logger.info("required input", resolution=images.shape)
    # result = model(images, None)  # noqa: ERA001
    dynamo_kwargs = instantiate(cfg.dynamo_kwargs)
    exported_program = torch.export.export(mod=model, args=(images,), **dynamo_kwargs)
    onnx_kwargs = instantiate(cfg.onnx_kwargs)
    model = torch.onnx.export(model=exported_program, **onnx_kwargs)

    logger.debug("exported", path=onnx_kwargs["f"])


if __name__ == "__main__":
    main()
