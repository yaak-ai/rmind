from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
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
    upload_to_wandb: bool = False


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _upload_to_wandb(config: Config) -> None:
    # Log the exported `.onnx` to the same wandb run that produced the source
    # checkpoint, so it lands as a sibling output artifact in the run's lineage.
    import wandb  # noqa: PLC0415

    source_ref = getattr(config.model, "artifact", None)
    if source_ref is None:
        msg = (
            "upload_to_wandb=True requires a model loaded via load_from_wandb_artifact"
        )
        raise ValueError(msg)

    api = wandb.Api()
    source = api.artifact(source_ref, type="model")
    source_run = source.logged_by()
    if source_run is None:
        msg = f"could not resolve the wandb run that produced {source_ref!r}"
        raise ValueError(msg)

    logger.debug(
        "resuming source run",
        entity=source_run.entity,
        project=source_run.project,
        id=source_run.id,
    )
    run = wandb.init(
        entity=source_run.entity,
        project=source_run.project,
        id=source_run.id,
        resume="must",
        job_type="export-onnx",
    )

    artifact = wandb.Artifact(
        name=f"{source.name.split(':')[0]}-onnx",
        type="onnx",
        metadata={
            "source_artifact": f"{source.entity}/{source.project}/{source.name}",
            "target": config.model.model_dump()["_target_"],
            "opset_version": config.opset_version,
            "dynamo": config.dynamo,
            "external_data": config.external_data,
            "optimize": config.optimize,
            "input_names": list(config.input_names) if config.input_names else None,
            "output_names": (
                list(config.output_names) if config.output_names else None
            ),
            "torch_version": torch.__version__,
            "onnx_version": _pkg_version("onnx"),
            "onnxruntime_version": _pkg_version("onnxruntime"),
        },
    )
    artifact.add_file(local_path=config.f.as_posix())
    if config.external_data:
        # weights spill into sibling files sharing the .onnx stem
        for path in config.f.parent.glob(f"{config.f.name}*"):
            if path != config.f:
                artifact.add_file(local_path=path.as_posix())

    _ = run.log_artifact(artifact)
    run.finish()
    logger.debug("uploaded onnx artifact", artifact=artifact.name, run=run.id)


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]

    # Name the .onnx after the checkpoint artifact it is generated from, e.g.
    # yaak/rmind/model-1vm7nc5i:v9 -> model-1vm7nc5i_v9.onnx (keeping run.dir).
    if (source_ref := getattr(config.model, "artifact", None)) is not None:
        stem = source_ref.rsplit("/", 1)[-1].replace(":", "_")
        config.f = config.f.with_name(f"{stem}.onnx")
        logger.debug("output path", f=config.f.as_posix())

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
        model=exported_program,
        **config.model_dump(exclude={"model", "upload_to_wandb"}),
    )

    logger.debug(
        "exported",
        model=config.f.resolve().as_posix(),
        artifacts=config.artifacts_dir.resolve().as_posix(),
    )

    if config.upload_to_wandb:
        logger.debug("uploading to wandb")
        _upload_to_wandb(config)


if __name__ == "__main__":
    main()
