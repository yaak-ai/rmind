from enum import StrEnum, auto

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from tensordict import TensorDict
from torch.nn import Module, ModuleDict  # noqa: TCH002
from wandb import Image
from yaak_datasets import Batch

from cargpt.components.episode import (
    EpisodeBuilder,
    Modality,
)
from cargpt.components.mask import (
    WandbAttentionMaskLegend,
)
from cargpt.utils._wandb import LoadableFromArtifact


class Objective(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    COPYCAT = auto()


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    hparams: AttributeDict

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.encoder: Module = instantiate(self.hparams.encoder)
        self.objectives: ModuleDict = instantiate(self.hparams.objectives)

    def _step(self, batch: Batch) -> TensorDict:
        inputs = self._build_input(batch)

        # TODO: currently this does full episode construction for each objective -- optimize?
        metrics = TensorDict(
            {
                name: objective(inputs, self.episode_builder, self.encoder)
                for name, objective in self.objectives.items()
            },
            batch_size=[],
            device=inputs.device,
        )

        if (
            isinstance(self.logger, WandbLogger)
            and (step := self.trainer.global_step) == 0
        ):
            for k, v in metrics.items():
                img = Image(v["mask"].with_legend(WandbAttentionMaskLegend).data)  # pyright: ignore
                self.logger.log_image(
                    f"masks/{k}",
                    [img],
                    step=step,
                )

        metrics = metrics.exclude(*((k, "mask") for k in metrics.keys()))  # pyright: ignore
        losses = metrics.select(*((k, "loss") for k in metrics.keys()))
        metrics[("loss", "total")] = sum(losses.values(True, True))

        return metrics

    def training_step(self, batch: Batch, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["train", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

    def validation_step(self, batch: Batch, _batch_idx: int):
        metrics = self._step(batch)

        self.log_dict({
            "/".join(["val", *k]): v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
        })

        return metrics["loss", "total"]

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

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=optimizer)
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

        return result

    def _populate_focal_loss_alpha(self):
        try:
            module = self.objectives.copycat.streams.memory_extraction
        except AttributeError:
            pass
        else:
            from polars import col  # noqa: PLC0415

            from cargpt.components.loss import AlphaFocalLoss  # noqa: PLC0415

            if losses := {
                k: v
                for k, v in module.losses.continuous.items()
                if isinstance(v, AlphaFocalLoss) and v.alpha is None
            }:
                col_map = {f"VehicleMotion_{k}_normalized": k for k in losses.keys()}

                dataset = self.trainer.datamodule.train_dataloader().dataset  # pyright: ignore
                ref_camera = dataset._cfg.samples.alignment.ref_camera
                metadata_df = dataset._metadata.select(
                    "drive_id",
                    f"{ref_camera}/ImageMetadata_frame_idx",
                    *col_map.keys(),
                ).rename(col_map)

                clip_metadata_df = (
                    dataset._clips.lazy()
                    .explode(f"{ref_camera}/ImageMetadata_frame_idx")
                    .join(
                        metadata_df.lazy(),
                        on=("drive_id", f"{ref_camera}/ImageMetadata_frame_idx"),
                    )
                    .group_by("clip_id")
                    .all()
                )

                delta_df = clip_metadata_df.select(
                    col(col_map.values()).list.diff(null_behavior="drop").explode()
                ).collect()

                deltas = TensorDict.from_dict(
                    {
                        ("continuous", k): v.to_numpy(allow_copy=False, writable=False)
                        for k, v in delta_df.to_dict().items()
                    },
                    device=self.device,
                )

                labels = deltas.named_apply(
                    lambda k, v: module.delta_tokenizers.get(k)(v),
                    nested_keys=True,
                )

                bincounts = labels.named_apply(
                    lambda k, v: torch.bincount(
                        v,
                        weights=None,
                        minlength=module.heads.get(k).out_features,
                    ),
                    nested_keys=True,
                    batch_size=[],
                )

                alpha = bincounts.apply(lambda x: 1e3 / x)

                for name, loss in losses.items():
                    loss.alpha = alpha["continuous", name]

                if isinstance(self.logger, WandbLogger):
                    for k, v in bincounts.cpu().items(True, True):
                        self.logger.log_table(
                            key="/".join([
                                "train",
                                "copycat",
                                "loss",
                                "memory_extraction",
                                *k,
                                "labels",
                            ]),
                            columns=["bin", "count"],
                            data=list(enumerate(v.tolist())),  # pyright: ignore
                            step=0,
                        )

    def on_fit_start(self) -> None:
        self._populate_focal_loss_alpha()
