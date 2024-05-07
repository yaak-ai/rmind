from os import PathLike
from typing import Self

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_fabric.plugins.io.torch_io import pl_load
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.core.saving import _load_state  # noqa: PLC2701
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from tensordict import TensorDict
from torch.nn import Module  # noqa: TCH002
from typing_extensions import override
from wandb import Image
from yaak_datasets import Batch

from cargpt.components.episode import (
    EpisodeBuilder,
    Modality,
)
from cargpt.components.mask import WandbAttentionMaskLegend
from cargpt.components.objectives import ObjectiveScheduler
from cargpt.components.objectives.common import ObjectiveName, PredictionResultKey
from cargpt.utils._wandb import LoadableFromArtifact
from cargpt.utils.containers import ModuleDict


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
        checkpoint_path: str | PathLike[str],
        **kwargs,
    ) -> Self:
        match kwargs:
            case {"hparams_updaters": hparams_updaters, **rest} if not rest:
                # relevant parts of super().load_from_checkpoint
                checkpoint = pl_load(checkpoint_path)  # pyright: ignore
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

                model = _load_state(cls, checkpoint, strict=False)
                state_dict = checkpoint["state_dict"]
                device = next(
                    (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
                    torch.tensor(0),
                ).device

                return model.to(device)  # pyright: ignore

            case _ if "hparams_updaters" not in kwargs:
                return super().load_from_checkpoint(
                    checkpoint_path=checkpoint_path,  # pyright: ignore
                    **kwargs,
                )

            case _:
                msg = "`hparams_updaters` cannot be combined with other kwargs"
                raise NotImplementedError(msg)

    def _step(self, batch: Batch) -> TensorDict:
        inputs = self._build_input(batch)

        selected_objectives = (
            self.objectives.keys()
            if self.objective_scheduler is None
            else self.objective_scheduler.sample()
        )

        # TODO: currently this does full episode construction for each objective -- optimize?
        metrics = TensorDict(
            {
                name: self.objectives[name](inputs, self.episode_builder, self.encoder)
                for name in selected_objectives
            },
            batch_size=[],
            device=inputs.device,
        )

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            episode = self.episode_builder.build_episode(inputs)
            objectives = (
                self.objectives.keys()
                if self.objective_scheduler is None
                else self.objective_scheduler.objectives
            )
            for obj in map(str, objectives):
                objective = self.objectives[obj]
                mask = objective._build_attention_mask(episode.index, episode.timestep)
                img = Image(mask.with_legend(WandbAttentionMaskLegend).data)
                self.logger.log_image(
                    f"masks/{obj}",
                    [img],
                    step=step,
                )

        losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # pyright: ignore
        metrics[("loss", "total")] = sum(losses.values(True, True))  # pyright: ignore

        return metrics

    @override
    def training_step(self, batch: Batch, *args):  # pyright: ignore[reportIncompatibleMethodOverride]
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["train", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    @override
    def validation_step(self, batch: Batch, *args):  # pyright: ignore[reportIncompatibleMethodOverride]
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["val", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    @override
    def predict_step(self, batch: Batch):  # pyright: ignore[reportIncompatibleMethodOverride]
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
                ),
            )
            for name, objective in self.objectives.items()
        })

        return TensorDict.from_dict(
            {
                "inputs": inputs,
                "predictions": predictions,
            },
            batch_size=batch.batch_size,  # pyright: ignore[reportAttributeAccessIssue]
        )

    def _build_input(self, batch: Batch) -> TensorDict:
        frames = batch.frames
        meta = batch.meta
        shapes = [
            frames.get_item_shape(k)
            for k in frames.keys(include_nested=True, leaves_only=True)  # pyright: ignore
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
                },
            },
            batch_size=batch_size,
            device=frames.device,
        )

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())  # pyright: ignore[reportAttributeAccessIssue]
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=optimizer)
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

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
        }

        sample_logit_bias_module_keys = {
            (
                ObjectiveName.FORWARD_DYNAMICS,
                "losses",
                Modality.CONTINUOUS,
                "speed",
            ),
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
                        zero_copy_only=False,
                        writable=False,
                    )
                    for (modality, k) in sample_logit_bias_losses.keys()
                },
                device=self.device,
                batch_size=[],
            )

            sample_labels = samples.unsqueeze(0).named_apply(
                lambda k, v: self.episode_builder.tokenizers.get(k)(v),
                nested_keys=True,
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
                        module=module_key,
                        loss=loss.__class__,
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
                "drive_id",
                f"{ref_camera}/ImageMetadata_frame_idx",
                *delta_cols.keys(),
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
                        zero_copy_only=False,
                        writable=False,
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
                        module=module_key,
                        loss=loss.__class__,
                    )

                    loss.logit_bias = delta_logit_bias[loss_key]

    @override
    def on_fit_start(self) -> None:
        self._populate_logit_bias()
