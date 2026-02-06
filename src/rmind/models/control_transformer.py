from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal, Self, override

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, InstanceOf, validate_call
from pytorch_lightning.core.saving import _load_state, pl_load  # noqa: PLC2701
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.migration.utils import (
    _pl_migrate_checkpoint,  # noqa: PLC2701
    pl_legacy_patch,
)
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from structlog import get_logger
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.mask import AttentionMaskBuilder, WandbAttentionMaskLegend
from rmind.components.objectives.base import PredictionKey
from rmind.config import HydraConfig
from rmind.utils._wandb import LoadableFromArtifact

logger = get_logger(__name__)


class LRSchedulerHydraConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    interval: Literal["epoch", "step"]
    scheduler: HydraConfig[LRScheduler]


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    episode_builder: Module
    encoder: Module
    objectives: ModuleDict
    optimizer: HydraConfig[Optimizer] | None = None
    lr_scheduler: LRSchedulerHydraConfig | None = None

    @validate_call
    def __init__(
        self,
        *,
        episode_builder: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        objectives: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
    ) -> None:
        super().__init__()

        hparams = {}

        if isinstance(episode_builder, HydraConfig):
            hparams["episode_builder"] = episode_builder.model_dump()
            episode_builder = episode_builder.instantiate()

        self.episode_builder = episode_builder

        if isinstance(encoder, HydraConfig):
            hparams["encoder"] = encoder.model_dump()
            encoder = encoder.instantiate()

        self.encoder = encoder

        if isinstance(objectives, HydraConfig):
            hparams["objectives"] = objectives.model_dump()
            objectives = objectives.instantiate()

        self.objectives = objectives

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()

        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()

        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @override
    @_restricted_classmethod
    def load_from_checkpoint(
        cls,  # noqa: N805
        checkpoint_path: _PATH,
        *,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: _PATH | None = None,
        strict: bool | None = None,
        hparams_updaters: Sequence[
            Callable[[Mapping[Any, Any]], Mapping[Any, Any]]
            | Callable[[Mapping[Any, Any]], None]
        ]
        | None = None,
        weights_only: bool | None = False,
        **kwargs: Any,
    ) -> Self:  # ty:ignore[invalid-method-override]
        match hparams_updaters:
            case [] | None:
                return super().load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    map_location=map_location,
                    hparams_file=hparams_file,
                    weights_only=weights_only,
                    strict=strict,
                    **kwargs,
                )

            case _:
                with pl_legacy_patch():
                    checkpoint = pl_load(
                        checkpoint_path,
                        map_location=map_location,
                        weights_only=weights_only,
                    )

                # convert legacy checkpoints to the new format
                checkpoint = _pl_migrate_checkpoint(
                    checkpoint, checkpoint_path=checkpoint_path
                )

                # update hparams
                hparams = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
                for fn in hparams_updaters:
                    match result := fn(hparams):
                        case DictConfig():
                            hparams = result

                        case None:
                            pass  # modified inplace

                        case _:
                            raise NotImplementedError
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

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

    @override
    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_packed, mask=episode.attention_mask
        )

        metrics = TensorDict({
            name: objective.compute_metrics(episode, embedding=embedding)  # ty:ignore[call-non-callable]
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            from wandb import Image  # noqa: PLC0415

            img = Image(
                AttentionMaskBuilder
                .build(
                    index=episode.index.to_dict(),
                    timestep=episode.timestep.to_dict(),
                    legend=WandbAttentionMaskLegend,
                )
                .float()
                .unsqueeze(0)
                .cpu()
            )
            self.logger.log_image("masks/shared", [img], step=step)

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.detach().items(
                    include_nested=True, leaves_only=True
                )
                if not any(part.startswith("_") for part in k)
            },
            sync_dist=True,
        )

        return {"loss": metrics["loss", "total"]} | metrics.select(
            *(
                (obj_name, "_artifacts")
                for obj_name, metric in metrics.items()
                if "_artifacts" in metric
            )
        ).to_dict()

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_packed, mask=episode.attention_mask
        )
        metrics = TensorDict({
            name: objective.compute_metrics(episode, embedding=embedding)  # ty:ignore[call-non-callable]
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

        if not self.trainer.sanity_checking:
            self.log_dict(
                {
                    "/".join(["val", *k]): v
                    for k, v in metrics.items(include_nested=True, leaves_only=True)
                    if not any(part.startswith("_") for part in k)
                },  # ty:ignore[invalid-argument-type]
                sync_dist=True,
            )

        return {"loss": metrics["loss", "total"]} | metrics.select(
            *(
                (obj_name, "_artifacts")
                for obj_name, metric in metrics.items()
                if "_artifacts" in metric
            )
        ).to_dict()

    @override
    def predict_step(self, batch: dict[str, Any]) -> TensorDict:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_packed, mask=episode.attention_mask
        )
        attention_rollout = None
        if (
            (
                PredictionKey.ATTENTION_ROLLOUT in set(PredictionKey)
                and obj.supports_attention_rollout
            )
            for obj in self.objectives.values()
        ):
            attention_rollout = self.encoder.compute_attention_rollout(  # ty:ignore[call-non-callable]
                src=episode.embeddings_packed,
                mask=episode.attention_mask,
                head_fusion="max",
                discard_ratio=0.9,
            )

        return TensorDict({
            name: objective.predict(
                episode=episode,
                embedding=embedding,
                keys=set(PredictionKey),
                tokenizers=self.episode_builder.tokenizers,
                attention_rollout=attention_rollout
                if objective.supports_attention_rollout
                else None,
            )  # ty:ignore[call-non-callable]
            for name, objective in self.objectives.items()
        }).auto_batch_size_(1)

    @override
    def forward(self, batch: TensorTree) -> TensorTree | TensorDict:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_packed, mask=episode.attention_mask
        )

        outputs = {
            name: objective(episode, embedding=embedding)
            for name, objective in self.objectives.items()
        }

        return TensorDict(outputs) if not torch.compiler.is_exporting() else outputs

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer is not None:
            from rmind.components import optimizers  # noqa: PLC0415

            match self.optimizer.target:
                case optimizers.SelectiveAdamW:
                    optimizer = self.optimizer.instantiate(module=self)

                case _:
                    optimizer = self.optimizer.instantiate(params=self.parameters())

        else:
            msg = "optimizer not specified"
            raise ValueError(msg)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler.scheduler.instantiate(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler} | self.lr_scheduler.model_dump(
                exclude={"scheduler"}
            )

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return {"optimizer": optimizer}
