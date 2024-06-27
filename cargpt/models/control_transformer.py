from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from lightning_fabric.plugins.io.torch_io import pl_load
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.core.saving import _load_state  # noqa: PLC2701
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from tensordict import TensorDict
from torch.nn import Module  # noqa: TCH002
from typing_extensions import override

from cargpt.components.episode import EpisodeBuilder, Modality
from cargpt.components.mask import WandbAttentionMaskLegend
from cargpt.components.objectives import ObjectiveScheduler
from cargpt.components.objectives.common import ObjectiveName, PredictionResultKey
from cargpt.utils._wandb import LoadableFromArtifact
from cargpt.utils.containers import ModuleDict

try:
    from yaak_datasets import Batch
except ImportError:
    from typing import Any

    Batch = Any


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

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
                    _pl_migrate_checkpoint,  # noqa: PLC2701
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
    def training_step(self, batch: Batch, *args):  # pyright: ignore[reportInvalidTypeForm]
        inputs = self._build_input(batch)

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
                name: self.objectives[name](inputs, self.episode_builder, self.encoder)
                for name in objectives_to_compute
            },
            batch_size=[],
            device=inputs.device,
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

            episode = self.episode_builder.build_episode(inputs)
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
    def validation_step(self, batch: Batch, *args):  # pyright: ignore[reportInvalidTypeForm]
        inputs = self._build_input(batch)

        metrics = TensorDict(
            {
                name: objective(inputs, self.episode_builder, self.encoder)
                for name, objective in self.objectives.items()
            },
            batch_size=[],
            device=inputs.device,
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
    def predict_step(self, batch: Batch):  # pyright: ignore[reportInvalidTypeForm]
        inputs = self._build_input(batch)

        predictions = TensorDict.from_dict({
            name: objective.predict(
                inputs,
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
            {"inputs": inputs, "predictions": predictions}, batch_size=batch.batch_size
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
                        for k, v in getattr(self.episode_builder, attr).flatten()
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

    def _build_input(self, batch: Batch) -> TensorDict:  # pyright: ignore[reportInvalidTypeForm]
        frames = batch.frames
        meta = batch.meta
        shapes = [
            frames.get_item_shape(k)
            for k in frames.keys(include_nested=True, leaves_only=True)
        ]

        # include timestep as batch dim
        batch_size = mit.one({(b, t) for (b, t, *_) in shapes})

        return TensorDict.from_dict(
            {
                Modality.IMAGE: frames,
                Modality.CONTINUOUS: {
                    "speed": meta["VehicleMotion_speed"],
                    "gas_pedal": meta["VehicleMotion_gas_pedal_normalized"],
                    "brake_pedal": meta["VehicleMotion_brake_pedal_normalized"],
                    "steering_angle": meta["VehicleMotion_steering_angle_normalized"],
                },
                Modality.DISCRETE: {
                    "turn_signal": meta["VehicleState_turn_signal"],
                    "incident": meta["incident_type"],
                },
            },
            batch_size=batch_size,
            device=frames.device,
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

        from polars import col  # noqa: PLC0415

        dataset = self.trainer.datamodule.train_dataloader().dataset  # pyright: ignore[reportAttributeAccessIssue]

        metadata_df_cols = {
            "VehicleMotion_gas_pedal_normalized": "gas_pedal",
            "VehicleMotion_brake_pedal_normalized": "brake_pedal",
            "VehicleMotion_steering_angle_normalized": "steering_angle",
            "VehicleMotion_speed": "speed",
            "VehicleState_turn_signal": "turn_signal",
            "incident_type": "incident",
        }

        sample_logit_bias_module_keys = {
            (ObjectiveName.FORWARD_DYNAMICS, "losses", Modality.CONTINUOUS, "speed"),
            (
                ObjectiveName.FORWARD_DYNAMICS,
                "losses",
                Modality.DISCRETE,
                "turn_signal",
            ),
            (
                ObjectiveName.INVERSE_DYNAMICS,
                "losses",
                Modality.CONTINUOUS,
                "gas_pedal",
            ),
            (
                ObjectiveName.INVERSE_DYNAMICS,
                "losses",
                Modality.CONTINUOUS,
                "brake_pedal",
            ),
            (
                ObjectiveName.INVERSE_DYNAMICS,
                "losses",
                Modality.CONTINUOUS,
                "steering_angle",
            ),
            (
                ObjectiveName.RANDOM_MASKED_HINDSIGHT_CONTROL,
                "losses",
                Modality.CONTINUOUS,
                "gas_pedal",
            ),
            (
                ObjectiveName.RANDOM_MASKED_HINDSIGHT_CONTROL,
                "losses",
                Modality.CONTINUOUS,
                "brake_pedal",
            ),
            (
                ObjectiveName.RANDOM_MASKED_HINDSIGHT_CONTROL,
                "losses",
                Modality.CONTINUOUS,
                "steering_angle",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "policy",
                "losses",
                Modality.CONTINUOUS,
                "gas_pedal",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "policy",
                "losses",
                Modality.CONTINUOUS,
                "brake_pedal",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "policy",
                "losses",
                Modality.CONTINUOUS,
                "steering_angle",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "policy",
                "losses",
                Modality.DISCRETE,
                "incident",
            ),
        }

        sample_logit_bias_losses = mit.map_reduce(
            iterable=(
                (k, loss)
                for (k, loss) in (
                    (k, self.objectives.get(k, default=None))
                    for k in sample_logit_bias_module_keys
                )
                if loss is not None and loss.logit_bias is None
            ),
            # group by (modality, name)
            keyfunc=lambda x: x[0][-2:],
        )

        if sample_logit_bias_losses:
            sample_cols = {
                k: v
                for (k, v) in metadata_df_cols.items()
                if v in {name for (_, name) in sample_logit_bias_losses.keys()}
            }

            sample_df = dataset._metadata.select(*sample_cols.keys()).rename(
                sample_cols
            )

            samples = TensorDict.from_dict(
                {
                    (modality, k): sample_df[k].to_numpy(
                        zero_copy_only=False, writable=False
                    )
                    for (modality, k) in sample_logit_bias_losses.keys()
                },
                device=self.device,
                batch_size=[],
            )

            sample_labels = samples.unsqueeze(0).named_apply(
                lambda k, v: self.episode_builder.tokenizers.get(k)(v), nested_keys=True
            )

            sample_bincounts = sample_labels.named_apply(
                lambda k, v: torch.bincount(
                    v.flatten(),
                    weights=None,
                    minlength=self.episode_builder.embeddings.get(k).weight.shape[0],
                ),
                nested_keys=True,
                batch_size=[],
            )

            sample_logit_bias = sample_bincounts.apply(
                lambda x: ((x + 1) / x.sum()).log()
            )

            for loss_key, losses in sample_logit_bias_losses.items():
                for module_key, loss in losses:
                    logger.debug(
                        "setting logit bias (sample-based)",
                        module=".".join(module_key),
                        loss=loss.__class__.__name__,
                    )

                    loss.logit_bias = sample_logit_bias[loss_key]

        delta_logit_bias_module_keys = {
            (
                ObjectiveName.COPYCAT,
                "streams",
                "memory_extraction",
                "losses",
                Modality.CONTINUOUS,
                "gas_pedal",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "memory_extraction",
                "losses",
                Modality.CONTINUOUS,
                "brake_pedal",
            ),
            (
                ObjectiveName.COPYCAT,
                "streams",
                "memory_extraction",
                "losses",
                Modality.CONTINUOUS,
                "steering_angle",
            ),
        }

        delta_logit_bias_losses = mit.map_reduce(
            iterable=(
                (k, loss)
                for (k, loss) in (
                    (k, self.objectives.get(k, default=None))
                    for k in delta_logit_bias_module_keys
                )
                if loss is not None and loss.logit_bias is None
            ),
            # group by (modality, name)
            keyfunc=lambda x: x[0][-2:],
        )

        if delta_logit_bias_losses:
            delta_cols = {
                k: v
                for (k, v) in metadata_df_cols.items()
                if v in {name for (_, name) in delta_logit_bias_losses.keys()}
            }

            ref_camera = dataset._cfg.samples.alignment.ref_camera
            delta_metadata_df = dataset._metadata.select(
                "drive_id", f"{ref_camera}/ImageMetadata_frame_idx", *delta_cols.keys()
            ).rename(delta_cols)

            clip_metadata_df = (
                dataset._clips.lazy()
                .explode(f"{ref_camera}/ImageMetadata_frame_idx")
                .join(
                    delta_metadata_df.lazy(),
                    on=("drive_id", f"{ref_camera}/ImageMetadata_frame_idx"),
                )
                .group_by("clip_id")
                .all()
            )

            delta_df = clip_metadata_df.select(
                col(delta_cols.values()).list.diff(null_behavior="drop").explode()
            ).collect()

            deltas = TensorDict.from_dict(
                {
                    (modality, k): delta_df[k].to_numpy(
                        zero_copy_only=False, writable=False
                    )
                    for (modality, k) in delta_logit_bias_losses.keys()
                },
                device=self.device,
                batch_size=[],
            )

            # TODO/HACK: technically for each loss requiring delta-based
            # logit bias we'd need to use the loss' parent objective's delta
            # detokenizers, but for now we know there's only one such objective
            # (copycat memory_extraction)
            memory_extraction = self.objectives.copycat.streams.memory_extraction
            delta_labels = deltas.named_apply(
                lambda k, v: memory_extraction.delta_tokenizers.get(k)(v),
                nested_keys=True,
            )

            delta_bincounts = delta_labels.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                lambda k, v: torch.bincount(
                    v,
                    weights=None,
                    minlength=memory_extraction.heads.get(k).out_features,
                ),
                nested_keys=True,
                batch_size=[],
            )

            delta_logit_bias = delta_bincounts.apply(
                lambda x: ((x + 1) / x.sum()).log()
            )

            for loss_key, losses in delta_logit_bias_losses.items():
                for module_key, loss in losses:
                    logger.debug(
                        "setting logit bias (delta-based)",
                        module=".".join(module_key),
                        loss=loss.__class__.__name__,
                    )

                    loss.logit_bias = delta_logit_bias[loss_key]

    @override
    def on_fit_start(self) -> None:
        self._populate_logit_bias()
