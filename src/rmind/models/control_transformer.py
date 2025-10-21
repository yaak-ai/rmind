from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any, ClassVar, Literal, Self, override

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import (
    _MAP_LOCATION_TYPE,  # pyright: ignore[reportPrivateUsage]
    _PATH,  # pyright: ignore[reportPrivateUsage]
)
from lightning_utilities.core.rank_zero import rank_zero_warn
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, InstanceOf, validate_call
from pytorch_lightning.core.saving import (
    _load_state,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    pl_load,  # pyright: ignore[reportPrivateImportUsage]
)
from pytorch_lightning.loggers import WandbLogger
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
from torch.nn import Module
from torch.nn.modules.module import (
    _IncompatibleKeys,  # pyright: ignore[reportPrivateUsage]
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.mask import WandbAttentionMaskLegend
from rmind.components.objectives.base import PredictionResultKey
from rmind.config import HydraConfig
from rmind.utils._wandb import LoadableFromArtifact

logger = get_logger(__name__)


class LRSchedulerHydraConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    interval: Literal["epoch", "step"]
    scheduler: HydraConfig[LRScheduler]


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    episode_builder: Module
    encoder: Module | None
    objectives: ModuleDict
    optimizer: HydraConfig[Optimizer] | None = None
    lr_scheduler: LRSchedulerHydraConfig | None = None

    @validate_call
    def __init__(
        self,
        *,
        episode_builder: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module] | None = None,
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
        if self.encoder is not None:
            for objective in self.objectives.values():
                if hasattr(objective, "encoder") and objective.encoder is None:  # pyright: ignore[reportUnnecessaryComparison]
                    objective.encoder = self.encoder

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()

        self.optimizer = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()

        self.lr_scheduler = lr_scheduler

        self.save_hyperparameters(hparams)

    @override
    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        shared_encoder_objectives = {
            k
            for k, v in self.objectives.items()
            if getattr(v, "encoder", None) is self.encoder
        }

        shared_encoder_objective_keys = set()
        state_dict = super().state_dict(*args, **kwargs)

        for k in state_dict:
            match k.split(".", maxsplit=3):
                case ["objectives", objective, "encoder", *_] if (
                    objective in shared_encoder_objectives
                ):
                    shared_encoder_objective_keys.add(k)

                case _:
                    pass

        return {
            k: v
            for k, v in state_dict.items()
            if k not in shared_encoder_objective_keys
        }

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        incompatible_keys = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        if incompatible_keys.missing_keys:
            encoder_keys = [k for k in state_dict if k.startswith("encoder.")]
            shared_encoder_objectives = {
                k
                for k, v in self.objectives.items()
                if getattr(v, "encoder", None) is self.encoder
            }
            shared_encoder_objective_keys = map(
                ".".join,
                product(("objectives",), shared_encoder_objectives, encoder_keys),
            )
            missing_keys = set(incompatible_keys.missing_keys) - set(
                shared_encoder_objective_keys
            )
            incompatible_keys = incompatible_keys._replace(
                missing_keys=list(missing_keys)
            )

        # mimic `super().load_state_dict` `strict` handling
        error_msgs: list[str] = []
        if strict:
            if unexpected_keys := incompatible_keys.unexpected_keys:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    )
                )

            if missing_keys := incompatible_keys.missing_keys:
                error_msgs.append(
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    )
                )

        if error_msgs:
            msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
                self.__class__.__name__, "\n\t".join(error_msgs)
            )
            raise RuntimeError(msg)

        return incompatible_keys

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
    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        episode = self.episode_builder(batch)

        metrics = TensorDict({
            name: objective.compute_metrics(episode)  # pyright: ignore[reportCallIssue]
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

        return metrics["loss", "total"]

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        episode = self.episode_builder(batch)
        metrics = TensorDict({
            name: objective.compute_metrics(episode)  # pyright: ignore[reportCallIssue]
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
                },
                sync_dist=True,
            )

        return metrics["loss", "total"]

    @override
    def predict_step(self, batch: dict[str, Any]) -> TensorDict:
        episode = self.episode_builder(batch)

        return TensorDict({
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
        }).auto_batch_size_(1)

    @override
    def forward(self, batch: TensorTree) -> TensorTree | TensorDict:
        episode = self.episode_builder(batch)

        outputs = {
            name: objective(episode) for name, objective in self.objectives.items()
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

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}  # pyright: ignore[reportReturnType]

        return {"optimizer": optimizer}
