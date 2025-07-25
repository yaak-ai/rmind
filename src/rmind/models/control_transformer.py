from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, overload, override

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import get_class
from hydra_once import instantiate
from lightning_fabric.utilities.types import (
    _MAP_LOCATION_TYPE,  # pyright: ignore[reportPrivateUsage]
    _PATH,  # pyright: ignore[reportPrivateUsage]
)
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig
from pydantic import InstanceOf, validate_call
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
from rbyte.config import HydraConfig
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.mask import WandbAttentionMaskLegend
from rmind.components.objectives import ObjectiveScheduler
from rmind.components.objectives.base import Objective, PredictionResultKey
from rmind.utils._wandb import LoadableFromArtifact

logger = get_logger(__name__)


def maybe_instantiate(value: Any | HydraConfig[Any]) -> Any:
    match value:
        case HydraConfig():
            return instantiate(value.model_dump(by_alias=True))

        case _:
            return value


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    @validate_call
    def __init__(
        self,
        *,
        input_builder: InstanceOf[Module] | HydraConfig[Module],
        episode_builder: InstanceOf[Module] | HydraConfig[Module],
        objectives: InstanceOf[ModuleDict] | HydraConfig[ModuleDict],
        objective_scheduler: InstanceOf[ObjectiveScheduler]
        | HydraConfig[ObjectiveScheduler]
        | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_builder: Module = maybe_instantiate(input_builder)
        self.episode_builder: Module = maybe_instantiate(episode_builder)
        self.objectives: ModuleDict = maybe_instantiate(objectives)

        if not mit.all_equal(
            (
                objective.encoder
                for objective in self.objectives.values()
                if hasattr(objective, "encoder")
            ),
            key=id,
        ):
            logger.warning("objectives have different encoders")

        self.objective_scheduler: ObjectiveScheduler | None = maybe_instantiate(
            objective_scheduler
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

        input = self.input_builder(batch.to_dict())
        episode = self.episode_builder.forward(input)

        metrics = TensorDict({
            name: self.objectives[name].compute_metrics(episode)  # pyright: ignore[reportCallIssue]
            for name in objectives_to_compute
        })

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
                mask = objective.build_attention_mask(  # pyright: ignore[reportCallIssue]
                    episode.index, episode.timestep, legend=WandbAttentionMaskLegend
                )
                img = Image(mask.mask.unsqueeze(0))
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
        input = self.input_builder.forward(batch.to_dict())
        episode = self.episode_builder.forward(input)
        metrics = TensorDict({
            name: objective.compute_metrics(episode)  # pyright: ignore[reportCallIssue]
            for name, objective in self.objectives.items()
        })

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues]  # noqa: SIM118
        loss_total = losses.sum(reduce=True)
        metrics["loss", "total"] = loss_total

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
            name: objective.predict(  # pyright: ignore[reportCallIssue]
                episode=episode,
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

    @overload
    def forward(self, batch: TensorDict | TensorClass) -> TensorDict: ...

    @overload
    def forward(self, batch: TensorTree) -> TensorTree: ...

    @override
    def forward(
        self, batch: TensorDict | TensorClass | TensorTree
    ) -> TensorDict | TensorTree:
        if isinstance(batch, TensorClass):
            batch = batch.to_tensordict()

        input = self.input_builder(batch)
        if isinstance(input, TensorDict):
            input = input.auto_batch_size_(2)

        episode = self.episode_builder(input)

        outputs = tree_map(
            lambda objective: objective.forward(episode),
            self.objectives.to_dict(),
            is_leaf=lambda x: isinstance(x, Objective),
        )

        return TensorDict(outputs) if not torch.compiler.is_exporting() else outputs

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
