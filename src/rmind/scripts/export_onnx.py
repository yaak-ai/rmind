from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import AfterValidator, BaseModel, ConfigDict
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger
from torch.utils._pytree import tree_flatten_with_path  # noqa: PLC2701

from rmind.config import HydraConfig

logger = get_logger(__name__)


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    model: HydraConfig[LightningModule]
    args: Annotated[Sequence[Any], AfterValidator(instantiate)]
    f: Path
    opset_version: int | None = None
    dynamo: Literal[True] = True
    external_data: bool = False
    optimize: bool = True
    verify: bool = True
    report: bool = True
    artifacts_dir: Path = Path.cwd()
    input_names: Sequence[str] | None = None
    output_names: Sequence[str] | None = None


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]

    logger.debug("instantiating", target=config.model.target)
    args = instantiate(config.args, _recursive_=True, _convert_="all")
    model = config.model.instantiate().eval()
    logger.debug(f"model summary:\n{ModelSummary(model)}")  # noqa: G004

    logger.debug("model eager forward")
    _ = model(*args)

    logger.debug("torch exporting")
    exported_program = torch.export.export(mod=model, args=tuple(args), strict=True)

    if config.output_names is None:
        out_spec = exported_program.call_spec.out_spec
        dummy = out_spec.unflatten([None] * out_spec.num_leaves)
        paths, _ = tree_flatten_with_path(dummy)
        config.output_names = [".".join(mk.key for mk in path) for path, _ in paths]  # ty:ignore[unresolved-attribute]
        logger.debug("inferred output_names", output_names=config.output_names)

    logger.debug("onnx exporting")
    model = torch.onnx.export(
        model=exported_program, **config.model_dump(exclude={"model"})
    )

    logger.debug(
        "exported",
        model=config.f.resolve().as_posix(),
        artifacts=config.artifacts_dir.resolve().as_posix(),
    )


if __name__ == "__main__":
    main()
