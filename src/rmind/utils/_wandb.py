from pathlib import Path
from typing import Annotated, Any, Self

import jq  # ty:ignore[unresolved-import]
import torch
from deepdiff import DeepDiff
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import OmegaConf
from pydantic import BeforeValidator, ConfigDict, validate_call
from pytorch_lightning.core.saving import _load_state, pl_load  # noqa: PLC2701
from pytorch_lightning.utilities.migration.utils import (
    _pl_migrate_checkpoint,  # noqa: PLC2701
    pl_legacy_patch,
)
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from structlog import get_logger

logger = get_logger(__name__)


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(
        cls, artifact: str, filename: str = "model.ckpt", **kwargs: Any
    ) -> Self:
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )

        artifact_dir = artifact_obj.download()
        ckpt_path = Path(artifact_dir) / filename

        return cls.load_from_checkpoint(ckpt_path, **kwargs)  # ty:ignore[unresolved-attribute]

    # NOTE: shared by every `pl.LightningModule` mixing this in (e.g.
    # `ControlTransformer`, `PatchPolicy`) so `hparams_jq` can patch a
    # checkpoint's saved hyperparameters (dropped/renamed fields, forced
    # inference-time flags like `sample_codes=false`, ...) ahead of
    # `cls(**hparams)` -- without every model reimplementing this override.
    @_restricted_classmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def load_from_checkpoint(  # noqa: PLR0913
        cls,  # noqa: N805
        checkpoint_path: _PATH,
        *,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: _PATH | None = None,
        strict: bool | None = None,
        hparams_jq: Annotated[jq._Program, BeforeValidator(jq.compile)] | None = None,
        weights_only: bool | None = False,
        **kwargs: Any,
    ) -> Self:
        if hparams_jq is None:
            return super().load_from_checkpoint(  # ty:ignore[unresolved-attribute]
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                weights_only=weights_only,
                strict=strict,
                **kwargs,
            )

        with pl_legacy_patch():
            checkpoint = pl_load(
                checkpoint_path, map_location=map_location, weights_only=weights_only
            )

        # convert legacy checkpoints to the new format
        checkpoint = _pl_migrate_checkpoint(checkpoint, checkpoint_path=checkpoint_path)

        hparams = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]  # ty:ignore[unresolved-attribute]
        hparams_container = OmegaConf.to_container(
            OmegaConf.create(hparams), resolve=False, throw_on_missing=False
        )
        hparams_container_updated = hparams_jq.input_value(hparams_container).first()

        for diff in (
            DeepDiff(hparams_container, hparams_container_updated, view="tree")
            .pretty()
            .splitlines()
        ):
            logger.debug("hparams updated", diff=diff)

        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = OmegaConf.create(  # ty:ignore[unresolved-attribute]
            hparams_container_updated
        )

        model = _load_state(cls, checkpoint, strict=strict, **kwargs)
        state_dict = checkpoint["state_dict"]
        if not state_dict:
            rank_zero_warn(
                f"The state dict in {checkpoint_path!r} contains no parameters."
            )
            return model  # ty:ignore[invalid-return-type]

        device = next(
            (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
            torch.tensor(0),
        ).device

        return model.to(device)  # ty:ignore[invalid-return-type, unresolved-attribute]
