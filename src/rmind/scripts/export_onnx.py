from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import hydra
import torch
import torch.fx.experimental._config as _fx_config  # noqa: PLC2701
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import AfterValidator, BaseModel, ConfigDict
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger
from torch.utils._pytree import tree_flatten_with_path  # noqa: PLC2701

from rmind.config import HydraConfig
from rmind.utils.patch import monkeypatched

# tensordict's global dicts mutated during export tracing cause a spurious
# "pending unbacked symbol u0" error even though the exported graph is valid
# (same workaround as tests/test_export.py::_soft_pending_unbacked).
_fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]

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

    # Eager forward populates cached buffers (e.g. attention mask) that are not
    # trace-friendly. Must run before torch.export — see EpisodeBuilder._build_attention_mask_tensor.
    # The second pass (with _is_exporting_flag=True) also serves as the source
    # of truth for output-name inference below.
    eager_out: Any = None
    for patch in (False, True):
        with monkeypatched(
            obj=(obj := torch.compiler),
            name=(name := "_is_exporting_flag"),
            patch=patch,
        ):
            logger.debug("model eager forward", **{f"{obj.__name__}.{name}": patch})
            eager_out = model(*args)

    if config.output_names is None:
        # Infer from eager output, not `out_spec.unflatten([None]*N)`: TensorDict's
        # pytree unflatten short-circuits to None on any-None leaves, collapsing to
        # output_names=[''] which makes torch.onnx prune subgraphs as unreached.
        paths_and_leaves, _ = tree_flatten_with_path(eager_out)
        config.output_names = [
            ".".join(mk.key for mk in path)  # ty:ignore[unresolved-attribute]
            for path, _ in paths_and_leaves
        ]
        logger.debug("inferred output_names", output_names=config.output_names)

    logger.debug("torch exporting")
    exported_program = torch.export.export(mod=model, args=tuple(args), strict=True)

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
