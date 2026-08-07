from typing import Annotated, Any, ClassVar, Literal, override

import pytorch_lightning as pl
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.objectives.base import ObjectivePredictionKey
from rmind.config import HydraConfig
from rmind.utils._wandb import LoadableFromArtifact

INTERNAL_STEP_OUTPUT_KEY = "_internal"


class LRSchedulerHydraConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    interval: Literal["epoch", "step"]
    scheduler: HydraConfig[LRScheduler]


class PredictionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    objectives: set[ObjectivePredictionKey] = Field(default_factory=set)


class ControlTransformer(LoadableFromArtifact, pl.LightningModule):
    episode_builder: Module
    encoder: Module
    objectives: ModuleDict
    optimizer: HydraConfig[Optimizer] | None = None
    lr_scheduler: LRSchedulerHydraConfig | None = None
    prediction_config: PredictionConfig

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        episode_builder: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        objectives: HydraConfig[ModuleDict] | InstanceOf[ModuleDict],
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        prediction_config: Annotated[
            PredictionConfig, Field(default_factory=PredictionConfig)
        ],
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

        self.encoder: HydraConfig[Module] | InstanceOf[Module] = encoder

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

        self.prediction_config = prediction_config

        self.save_hyperparameters(hparams)

    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )

        metrics = TensorDict({
            name: objective.compute_metrics(episode=episode, embedding=embedding)  # ty:ignore[call-non-callable]
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

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

        outputs = {"loss": metrics["loss", "total"]} | metrics.select(
            *(
                (obj_name, "_artifacts")
                for obj_name, metric in metrics.items()
                if "_artifacts" in metric
            )
        ).to_dict()

        if self.current_epoch == 0 and batch_idx == 0:
            outputs[INTERNAL_STEP_OUTPUT_KEY] = {"episode": episode.detach()}

        return outputs

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        metrics = TensorDict({
            name: objective.compute_metrics(episode=episode, embedding=embedding)  # ty:ignore[call-non-callable]
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
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        objectives_predictions = {
            name: objective.predict(
                episode=episode,
                embedding=embedding,
                keys=frozenset(self.prediction_config.objectives),
                tokenizers=self.episode_builder.tokenizers,
            )  # ty:ignore[call-non-callable]
            for name, objective in self.objectives.items()
        }
        return TensorDict(objectives_predictions).auto_batch_size_(1)  # ty:ignore[invalid-argument-type]

    @override
    def forward(self, batch: TensorTree) -> TensorTree | TensorDict:
        episode = self.episode_builder(batch)
        embedding = self.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )

        outputs = {
            name: objective(episode=episode, embedding=embedding)
            for name, objective in self.objectives.items()
        }

        return TensorDict(outputs)  # ty:ignore[invalid-argument-type]

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
