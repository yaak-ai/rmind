from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, override

import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from lightning_fabric.utilities.types import (
    _MAP_LOCATION_TYPE,  # pyright: ignore[reportPrivateUsage]
    _PATH,  # pyright: ignore[reportPrivateUsage]
)
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, ImportString
from pytorch_lightning.core.saving import (
    _load_state,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    pl_load,  # pyright: ignore[reportPrivateImportUsage]
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.migration.utils import (
    _pl_migrate_checkpoint,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    pl_legacy_patch,
)
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
)
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module  # noqa: TC002

from rmind.components.mask import WandbAttentionMaskLegend
from rmind.components.objectives.base import PredictionResultKey
from rmind.utils._wandb import LoadableFromArtifact

if TYPE_CHECKING:
    from rmind.components.episode import EpisodeBuilder
    from rmind.components.objectives import ObjectiveScheduler
    from rmind.utils import ModuleDict

logger = get_logger(__name__)


class HydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")

    target: ImportString[type[T]] = Field(alias="_target_")
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(by_alias=True), **kwargs)


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    def __init__(self, **_kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_builder: Module = instantiate(self.hparams.input_builder)  # pyright: ignore[reportAttributeAccessIssue]
        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)  # pyright: ignore[reportAttributeAccessIssue]
        self.encoder: Module = instantiate(self.hparams.encoder)  # pyright: ignore[reportAttributeAccessIssue]
        self.objectives: ModuleDict = instantiate(self.hparams.objectives)  # pyright: ignore[reportAttributeAccessIssue]
        self.objective_scheduler: ObjectiveScheduler | None = instantiate(
            self.hparams.get("objective_scheduler")
        )
        if self.objective_scheduler is not None and (
            (specified := self.objectives.keys())
            != (scheduled := set(self.objective_scheduler.objectives))
        ):
            msg = f"objective scheduler enabled but {specified} != {scheduled}"
            raise ValueError(msg)

    @override
    @_restricted_classmethod
    def load_from_checkpoint(  # pyright: ignore[reportIncompatibleMethodOverride]
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
        **kwargs: Any,
    ) -> Self:
        match hparams_updaters:
            case [] | None:
                return super().load_from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    map_location=map_location,
                    hparams_file=hparams_file,
                    strict=strict,
                    **kwargs,
                )

            case _:
                with pl_legacy_patch():
                    checkpoint = pl_load(checkpoint_path, map_location=map_location)

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
                    return model  # pyright: ignore[reportReturnType]

                device = next(
                    (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
                    torch.tensor(0),
                ).device

                return model.to(device)  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]

    @override
    def training_step(self, batch: TensorDict, _batch_idx: int) -> Tensor:
        all_objectives = tuple(self.objectives.keys())
        scheduled_objectives = (
            all_objectives
            if self.objective_scheduler is None
            else tuple(self.objective_scheduler.sample())
        )

        match strategy := self.trainer.strategy:
            case SingleDeviceStrategy():
                objectives_to_compute = scheduled_objectives

            case DDPStrategy():
                # compute all objectives since we need a static graph
                objectives_to_compute = all_objectives

            case _:
                msg = f"Don't know the correct way to handle {strategy}"
                raise NotImplementedError(msg)

        input = self.input_builder.forward(batch)
        episode = self.episode_builder.forward(input)

        metrics = TensorDict(
            {
                name: self.objectives[name].forward(episode, self.encoder)
                for name in objectives_to_compute
            },
            device=input.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues]  # noqa: SIM118
        losses.select(*(set(metrics.keys()) - set(scheduled_objectives))).zero_()  # pyright: ignore[reportArgumentType]
        loss_total = losses.sum(reduce=True)

        metrics["loss", "total"] = loss_total

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            from wandb import Image  # noqa: PLC0415

            for name, objective in self.objectives.items():
                mask = objective.build_attention_mask(episode.index, episode.timestep)
                img = Image(
                    mask.with_legend(WandbAttentionMaskLegend).mask.unsqueeze(0)
                )
                self.logger.log_image(f"masks/{name}", [img], step=step)

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.detach().items(
                    include_nested=True, leaves_only=True
                )
            },
            sync_dist=True,
        )

        return loss_total

    @override
    def validation_step(self, batch: TensorDict, _batch_idx: int) -> Tensor:
        input = self.input_builder.forward(batch)
        episode = self.episode_builder.forward(input)
        metrics = TensorDict(
            {
                name: objective.forward(episode, self.encoder)
                for name, objective in self.objectives.items()
            },
            device=input.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues]  # noqa: SIM118
        metrics["loss", "total"] = sum(
            losses.values(include_nested=True, leaves_only=True)
        )

        if not self.trainer.sanity_checking:
            self.log_dict(
                {
                    "/".join(["val", *k]): v
                    for k, v in metrics.items(include_nested=True, leaves_only=True)
                },
                sync_dist=True,
            )

        return metrics["loss", "total"]

    @override
    def predict_step(self, batch: TensorDict) -> TensorDict:
        input = self.input_builder.forward(batch)
        episode = self.episode_builder.forward(input)

        predictions = TensorDict({
            name: objective.predict(
                episode=episode,
                encoder=self.encoder,
                result_keys=frozenset((
                    PredictionResultKey.GROUND_TRUTH,
                    PredictionResultKey.PREDICTION_VALUE,
                    PredictionResultKey.PREDICTION_STD,
                    PredictionResultKey.PREDICTION_PROBS,
                    PredictionResultKey.SCORE_LOGPROB,
                    PredictionResultKey.SCORE_L1,
                    PredictionResultKey.SUMMARY_EMBEDDINGS,
                )),
                tokenizers=self.episode_builder.tokenizers,
            )
            for name, objective in self.objectives.items()
        })

        return TensorDict(
            {"input": input, "predictions": predictions}  # pyright: ignore[reportArgumentType]
        )

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        result = {}

        if (cfg := self.hparams.get("optimizer")) is not None:
            from rmind.components import optimizers  # noqa: PLC0415

            match get_class(cfg._target_):
                case optimizers.SelectiveAdamW:
                    result["optimizer"] = instantiate(cfg, module=self)

                case _:
                    result["optimizer"] = instantiate(cfg, params=self.parameters())

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=result["optimizer"])
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

        logger.debug("configure_optimizers", result=result)

        return result  # pyright: ignore[reportReturnType]
