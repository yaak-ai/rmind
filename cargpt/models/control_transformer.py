from enum import StrEnum, auto
from os import PathLike
from typing import Self

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_fabric.plugins.io.torch_io import pl_load
from lightning_fabric.utilities.data import AttributeDict
from pytorch_lightning.core.saving import _load_state  # noqa: PLC2701
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from tensordict import TensorDict
from torch import nn
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

        self.apply(self._init_weights)  # pyright: ignore

    @_restricted_classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | PathLike, **kwargs) -> Self:
        match kwargs:
            case {"hparams_update_fn": hparams_update_fn, **rest} if not rest:
                # relevant parts of super().load_from_checkpoint
                checkpoint = pl_load(checkpoint_path)  # pyright: ignore
                hparams = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
                hparams_updated = hparams_update_fn(hparams)
                checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams_updated

                model = _load_state(cls, checkpoint, strict=False)
                state_dict = checkpoint["state_dict"]
                device = next(
                    (t for t in state_dict.values() if isinstance(t, torch.Tensor)),
                    torch.tensor(0),
                ).device

                return model.to(device)  # pyright: ignore

            case _ if "hparams_update_fn" not in kwargs:
                return super().load_from_checkpoint(
                    checkpoint_path=checkpoint_path,  # pyright: ignore
                    **kwargs,
                )

            case _:
                msg = "`hparams_udpate_fn` cannot be combined with other kwargs"
                raise NotImplementedError(msg)

    def _step(self, batch: Batch) -> TensorDict:
        inputs = self._build_input(batch)

        # TODO: currently this does full episode construction for each objective -- optimize?
        metrics = TensorDict(
            {
                name: objective(
                    inputs, self.episode_builder, self.encoder, self.logit_bias
                )
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

    def predict_step(self, batch: Batch):
        inputs = self._build_input(batch)

        predictions = TensorDict.from_dict({
            name: objective.predict(inputs, self.episode_builder, self.encoder)
            for name, objective in self.objectives.items()
        })

        return TensorDict.from_dict(
            {
                "inputs": inputs,
                "predictions": predictions,
            },
            batch_size=batch.batch_size,  # pyright: ignore
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

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=optimizer)
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

        return result

    def _populate_logit_bias(self):
        # https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023_paper.pdf
        # Section 3.2
        from polars import col  # noqa: PLC0415

        metadata_map = {
            "VehicleMotion_gas_pedal_normalized": "gas_pedal",
            "VehicleMotion_brake_pedal_normalized": "brake_pedal",
            "VehicleMotion_steering_angle_normalized": "steering_angle",
            "VehicleMotion_speed": "speed",
            "VehicleState_turn_signal": "turn_signal",
        }

        modality_type_map = {
            "gas_pedal": "continuous",
            "brake_pedal": "continuous",
            "steering_angle": "continuous",
            "speed": "continuous",
            "turn_signal": "discrete",
        }

        dataset = self.trainer.datamodule.train_dataloader().dataset  # pyright: ignore
        ref_camera = dataset._cfg.samples.alignment.ref_camera
        metadata_df = dataset._metadata.select(
            "drive_id",
            f"{ref_camera}/ImageMetadata_frame_idx",
            *metadata_map.keys(),
        ).rename(metadata_map)

        metadata_values_df = metadata_df.select(col(metadata_map.values()))

        metadata_values = TensorDict.from_dict(
            {
                (modality_type_map[k], k): v.to_numpy(allow_copy=False, writable=False)
                for k, v in metadata_values_df.to_dict().items()
            },
            device=self.device,
        )

        labels = metadata_values.named_apply(
            lambda k, v: self.episode_builder.tokenizers.get(k)(v.reshape(-1, 1)),
            nested_keys=True,
        )

        bincounts = labels.named_apply(
            lambda k, v: torch.bincount(
                v.flatten(),
                weights=None,
                minlength=self.episode_builder.embeddings.get(k).weight.shape[0],
            ),
            nested_keys=True,
            batch_size=[],
        )

        # add +1 for empty bins
        self.logit_bias = bincounts.apply(
            lambda x: torch.log((x + 1) / x.sum()).reshape(1, -1)
        )

    # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L163C5-L172C47
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # pyright: ignore
            if module.bias is not None:  # pyright: ignore
                torch.nn.init.zeros_(module.bias)  # pyright: ignore
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # pyright: ignore
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)  # pyright: ignore
            torch.nn.init.ones_(module.weight)  # pyright: ignore

    def on_fit_start(self) -> None:
        self._populate_logit_bias()
