from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, override

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
from torch.nn import Module  # noqa: TC002

from cargpt.components.episode import EpisodeBuilder, Modality, PositionEncoding
from cargpt.components.mask import WandbAttentionMaskLegend
from cargpt.components.objectives import ObjectiveScheduler
from cargpt.components.objectives.base import ObjectiveName, PredictionResultKey
from cargpt.utils._wandb import LoadableFromArtifact
from cargpt.utils.containers import ModuleDict


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.input_transforms = instantiate(self.hparams.get("input_transforms", None))

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
                from pytorch_lightning.utilities.migration.utils import pl_legacy_patch  # noqa: I001, PLC0415
                from pytorch_lightning.utilities.migration.utils import (  # noqa: PLC0415
                    _pl_migrate_checkpoint,
                )
                from lightning_utilities.core.rank_zero import rank_zero_warn  # noqa: PLC0415

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
        input = self._build_input(batch)

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
                name: self.objectives[name](input, self.episode_builder, self.encoder)
                for name in objectives_to_compute
            },
            batch_size=[],
            device=input.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
        losses.select(*(set(objectives_to_compute) - set(scheduled_objectives))).zero_()

        metrics["loss", "total"] = sum(  # pyright: ignore[reportArgumentType]
            losses.values(include_nested=True, leaves_only=True)
        )

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            from wandb import Image  # noqa: PLC0415

            episode = self.episode_builder.build_episode(input)
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
        input = self._build_input(batch)

        metrics = TensorDict(
            {
                name: objective(input, self.episode_builder, self.encoder)
                for name, objective in self.objectives.items()
            },
            batch_size=[],
            device=input.device,
        )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType]
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
        input = self._build_input(batch)

        predictions = TensorDict.from_dict(
            {
                name: objective.predict(
                    input,
                    episode_builder=self.episode_builder,
                    encoder=self.encoder,
                    # TODO: attention
                    result_keys=frozenset((
                        PredictionResultKey.GROUND_TRUTH,
                        PredictionResultKey.PREDICTION,
                        PredictionResultKey.PREDICTION_STD,
                        PredictionResultKey.PREDICTION_PROBS,
                        PredictionResultKey.SCORE_LOGPROB,
                        PredictionResultKey.SCORE_L1,
                    )),
                )
                for name, objective in self.objectives.items()
            },
            batch_size=input.batch_size,
        )

        return TensorDict.from_dict(
            {"inputs": input, "predictions": predictions}, batch_size=input.batch_size
        )

    @override
    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking and isinstance(self.logger, WandbLogger):
            from torchmetrics.functional import (  # noqa: PLC0415
                pairwise_cosine_similarity as similarity_fn,
            )
            from wandb import Image  # noqa: PLC0415

            # TODO: need access to episode index to build full episode position embedding
            similarity_keys = (
                ("embeddings", Modality.CONTINUOUS),
                ("embeddings", Modality.DISCRETE),
                ("position_encoding", PositionEncoding.IMAGE),
                ("position_encoding", PositionEncoding.OBSERVATIONS),
                ("position_encoding", PositionEncoding.TIMESTEP),
            )

            similarities = (
                TensorDict.from_module(self.episode_builder)
                .select(*similarity_keys)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
                .apply(similarity_fn, inplace=False)
                .cpu()
            )

            self.logger.log_image(
                key=f"embeddings/{similarity_fn.__name__}",
                images=[
                    Image(v, caption=".".join(k[:-1]))
                    for k, v in similarities.items(True, True)
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

        sample_logit_bias_losses: list[tuple[tuple[str, ...], Module]] = []
        delta_logit_bias_losses: list[tuple[tuple[str, ...], Module]] = []

        for k, mod in self.objectives.tree_flatten_with_path():
            if hasattr(mod, "logit_bias") and mod.logit_bias is None:
                match k:
                    case (objective, "losses", *_):
                        match objective:
                            case (
                                ObjectiveName.FORWARD_DYNAMICS
                                | ObjectiveName.INVERSE_DYNAMICS
                                | ObjectiveName.RANDOM_MASKED_HINDSIGHT_CONTROL
                                | ObjectiveName.POLICY
                            ):
                                dest = sample_logit_bias_losses

                            case ObjectiveName.MEMORY_EXTRACTION:
                                dest = delta_logit_bias_losses

                            case _:
                                raise NotImplementedError

                        dest.append((k, mod))

                    case _:
                        raise NotImplementedError

        cols = {
            (Modality.CONTINUOUS, "gas_pedal"): "VehicleMotion.gas_pedal_normalized",
            (
                Modality.CONTINUOUS,
                "brake_pedal",
            ): "VehicleMotion.brake_pedal_normalized",
            (
                Modality.CONTINUOUS,
                "steering_angle",
            ): "VehicleMotion.steering_angle_normalized",
            (Modality.CONTINUOUS, "speed"): "VehicleMotion.speed",
            (Modality.DISCRETE, "turn_signal"): "VehicleState.turn_signal",
        }

        by_key_modality_name = lambda x: x[0][-2:]  # noqa: E731

        for k_loss, losses in mit.map_reduce(
            sample_logit_bias_losses, keyfunc=by_key_modality_name
        ).items():
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

        for k_loss, losses in mit.map_reduce(
            delta_logit_bias_losses, keyfunc=by_key_modality_name
        ).items():
            values = (
                samples.select(
                    pl.col(cols[k_loss]).list.diff(null_behavior="drop").explode()
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

    def _build_input(self, batch: Any) -> TensorDict:
        data = batch.data
        input = (
            TensorDict.from_dict(
                {
                    Modality.IMAGE: {"cam_front_left": data["cam_front_left"]},
                    Modality.CONTINUOUS: {
                        "speed": data["VehicleMotion.speed"],
                        "gas_pedal": data["VehicleMotion.gas_pedal_normalized"],
                        "brake_pedal": data["VehicleMotion.brake_pedal_normalized"],
                        "steering_angle": data[
                            "VehicleMotion.steering_angle_normalized"
                        ],
                    },
                    Modality.DISCRETE: {
                        "turn_signal": data["VehicleState.turn_signal"]
                    },
                },
                device=self.device,
            )
            .auto_batch_size_(batch_dims=2)
            .refine_names("b", "t")
        )

        for transform in self.input_transforms or ():
            input = input.update(input.select(*transform.select).apply(transform.apply))

        return input
