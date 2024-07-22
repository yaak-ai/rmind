from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, cast

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from lightning_fabric.plugins.io.torch_io import pl_load
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.core.saving import _load_state
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from tensordict import TensorDict
from torch.nn import Module  # noqa: TCH002
from typing_extensions import override
from yaak_datasets import Batch

from cargpt.components.episode import EpisodeBuilder, Modality
from cargpt.components.mask import WandbAttentionMaskLegend
from cargpt.components.objectives import ObjectiveScheduler
from cargpt.components.objectives.common import ObjectiveName, PredictionResultKey
from cargpt.utils._wandb import LoadableFromArtifact
from cargpt.utils.containers import ModuleDict


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.batch_transforms = instantiate(self.hparams.get("batch_transforms", None))

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)  # pyright: ignore[reportAttributeAccessIssue]
        self.encoder: Module = instantiate(self.hparams.encoder)  # pyright: ignore[reportAttributeAccessIssue]
        self.objectives: ModuleDict = instantiate(self.hparams.objectives)  # pyright: ignore[reportAttributeAccessIssue]
        self.objective_scheduler: ObjectiveScheduler | None = instantiate(
            self.hparams.get("objective_scheduler")
        )
        if self.objective_scheduler is not None and (
            (specified := set(self.objectives.keys()))
            != (scheduled := {x.value for x in self.objective_scheduler.objectives})
        ):
            msg = f"objective scheduler enabled but {specified} != {scheduled}"
            raise ValueError(msg)

    @override
    @_restricted_classmethod
    def load_from_checkpoint(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
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
                from pytorch_lightning.utilities.migration import pl_legacy_patch  # noqa: I001, PLC0415
                from pytorch_lightning.utilities.migration.utils import (  # noqa: PLC0415
                    _pl_migrate_checkpoint,
                )
                from pytorch_lightning.utilities.rank_zero import rank_zero_warn  # noqa: PLC0415

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
    def training_step(self, batch: TensorDict, *args):
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

        metrics = TensorDict(
            {
                name: self.objectives[name](batch, self.episode_builder, self.encoder)
                for name in objectives_to_compute
            },
            batch_size=[],
            device=batch.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues]
        losses.select(*(set(objectives_to_compute) - set(scheduled_objectives))).zero_()

        metrics["loss", "total"] = sum(  # pyright: ignore[reportArgumentType]
            losses.values(include_nested=True, leaves_only=True)
        )

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            from wandb import Image  # noqa: PLC0415

            episode = self.episode_builder.build_episode(batch)
            objectives = (
                all_objectives
                if self.objective_scheduler is None
                else self.objective_scheduler.objectives
            )
            # TODO: batch log mask images
            for obj in map(str, objectives):
                objective = self.objectives[obj]
                mask = objective._build_attention_mask(episode.index, episode.timestep)
                img = Image(mask.with_legend(WandbAttentionMaskLegend).data)
                self.logger.log_image(f"masks/{obj}", [img], step=step)

        self.log_dict(
            {
                "/".join(["train", *k]): v
                for k, v in metrics.items(include_nested=True, leaves_only=True)
            },
            sync_dist=True,
        )

        return metrics["loss", "total"]

    @override
    def validation_step(self, batch: TensorDict, *args):
        metrics = TensorDict(
            {
                name: objective(batch, self.episode_builder, self.encoder)
                for name, objective in self.objectives.items()
            },
            batch_size=[],
            device=batch.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues]
        metrics["loss", "total"] = sum(  # pyright: ignore[reportArgumentType]
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
    def predict_step(self, batch: TensorDict):
        predictions = TensorDict.from_dict({
            name: objective.predict(
                batch,
                episode_builder=self.episode_builder,
                encoder=self.encoder,
                # TODO: attention
                result_keys=(
                    PredictionResultKey.GROUND_TRUTH,
                    PredictionResultKey.PREDICTION,
                    PredictionResultKey.PREDICTION_PROBS,
                    PredictionResultKey.SCORE_LOGPROB,
                    PredictionResultKey.SCORE_L1,
                ),
            )
            for name, objective in self.objectives.items()
        })

        return TensorDict.from_dict(
            {"inputs": batch, "predictions": predictions}, batch_size=batch.batch_size
        )

    @override
    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking and isinstance(self.logger, WandbLogger):
            from torchmetrics.functional import (  # noqa: PLC0415
                pairwise_cosine_similarity as similarity_fn,
            )
            from wandb import Image  # noqa: PLC0415

            embeddings = TensorDict.from_dict(
                {
                    attr: {
                        k: v.weight
                        for k, v in cast(
                            ModuleDict, getattr(self.episode_builder, attr)
                        ).flatten()
                        if isinstance(v, torch.nn.Embedding) and v.num_embeddings > 1
                    }
                    for attr in ("embeddings", "position_encoding")
                },
                batch_size=[],
            )

            # TODO: need access to episode index to build full episode position embedding
            similarities = embeddings.apply(similarity_fn)

            self.logger.log_image(
                key=f"embeddings/{similarity_fn.__name__}",
                images=[
                    Image(v, caption=".".join(mit.always_iterable(k)))
                    for k, v in similarities.items(  # pyright: ignore[reportAttributeAccessIssue]
                        include_nested=True, leaves_only=True
                    )
                ],
                step=self.trainer.global_step,
            )

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        result = {}

        if (cfg := self.hparams.get("optimizer")) is not None:
            from cargpt.components.optimizers import SelectiveAdamW  # noqa: PLC0415

            result["optimizer"] = (
                instantiate(cfg, module=self)
                if get_class(cfg._target_) is SelectiveAdamW
                else instantiate(cfg, params=self.parameters())
            )

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=result["optimizer"])
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

        logger.debug("configure_optimizers", result=result)

        return result

    def _populate_logit_bias(self):
        # https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023_paper.pdf
        # Section 3.2

        import polars as pl  # noqa: PLC0415

        samples = self.trainer.datamodule.train_dataloader().dataset.samples  # pyright: ignore[reportAttributeAccessIssue]

        sample_logit_bias_losses = defaultdict(list)
        delta_logit_bias_losses = defaultdict(list)

        for k, mod in self.objectives.flatten():
            if hasattr(mod, "logit_bias") and mod.logit_bias is None:
                match k:
                    case (
                        ObjectiveName.COPYCAT,
                        "streams",
                        "memory_extraction",
                        "losses",
                        modality,
                        name,
                    ):
                        delta_logit_bias_losses[(modality, name)].append((k, mod))

                    case (*_, modality, name):
                        sample_logit_bias_losses[(modality, name)].append((k, mod))

                    case _:
                        raise NotImplementedError

        cols = {
            (Modality.CONTINUOUS, "gas_pedal"): "vehicle_motion.gas_pedal_normalized",
            (
                Modality.CONTINUOUS,
                "brake_pedal",
            ): "vehicle_motion.brake_pedal_normalized",
            (
                Modality.CONTINUOUS,
                "steering_angle",
            ): "vehicle_motion.steering_angle_normalized",
            (Modality.CONTINUOUS, "speed"): "vehicle_motion.speed",
            (Modality.DISCRETE, "turn_signal"): "vehicle_state.turn_signal",
        }

        for k_loss, losses in sample_logit_bias_losses.items():
            values = (
                samples.select(pl.col(cols[k_loss]).explode())
                .to_torch()
                .to(self.device)
            )
            tokenizer = self.episode_builder.tokenizers.get(k_loss)
            labels = tokenizer(values)
            freq = torch.bincount(
                labels.flatten(),
                weights=None,
                minlength=self.episode_builder.embeddings.get(k_loss).weight.shape[0],
            )

            logit_bias = ((freq + 1) / freq.sum()).log()

            for k_module, loss in losses:
                logger.debug(
                    "setting logit bias (sample-based)",
                    module=".".join(k_module),
                    loss=loss.__class__.__name__,
                )

                loss.logit_bias = logit_bias

        for k_loss, losses in delta_logit_bias_losses.items():
            values = (
                samples.select(
                    pl.col(cols[k_loss])
                    .arr.to_list()
                    .list.diff(null_behavior="drop")
                    .explode()
                )
                .to_torch()
                .to(self.device)
            )

            for k_module, loss in losses:
                logger.debug(
                    "setting logit bias (delta-based)",
                    module=".".join(k_module),
                    loss=loss.__class__.__name__,
                )

                match k_module:
                    case (*k_objective, "losses", _modality, _name):
                        objective = self.objectives.get(tuple(k_objective))
                        tokenizer = objective.delta_tokenizers.get(k_loss)
                        labels = tokenizer(values)
                        freq = torch.bincount(
                            labels.flatten(),
                            weights=None,
                            minlength=objective.heads.get(k_loss).out_features,
                        )
                        logit_bias = ((freq + 1) / freq.sum()).log()

                        loss.logit_bias = logit_bias

                    case _:
                        raise NotImplementedError

    @override
    def on_fit_start(self) -> None:
        self._populate_logit_bias()

    @override
    def on_before_batch_transfer(self, batch: Batch, dataloader_idx: int) -> TensorDict:  # pyright: ignore[reportGeneralTypeIssues]
        table = batch.table

        return TensorDict.from_dict({
            Modality.IMAGE: batch.frame,
            Modality.CONTINUOUS: {
                "speed": table["vehicle_motion.speed"],
                "gas_pedal": table["vehicle_motion.gas_pedal_normalized"],
                "brake_pedal": table["vehicle_motion.brake_pedal_normalized"],
                "steering_angle": table["vehicle_motion.steering_angle_normalized"],
            },
            Modality.DISCRETE: {"turn_signal": table["vehicle_state.turn_signal"]},
        }).auto_batch_size_(batch_dims=2)

    @override
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        for transform in self.batch_transforms or ():
            batch = batch.update(batch.select(*transform.select).apply(transform.apply))

        return batch
