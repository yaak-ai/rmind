from collections import defaultdict
from os import PathLike
from typing import Self

import more_itertools as mit
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from lightning_fabric.plugins.io.torch_io import pl_load
from loguru import logger
from pytorch_lightning.core.saving import _load_state  # noqa: PLC2701
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from tensordict import TensorDict
from torch.nn import Module, ModuleDict  # noqa: TCH002
from typing_extensions import override
from wandb import Image
from yaak_datasets import Batch

from cargpt.components.episode import (
    EpisodeBuilder,
    Modality,
)
from cargpt.components.loss import ObjectiveScheduler
from cargpt.components.mask import WandbAttentionMaskLegend
from cargpt.utils._wandb import LoadableFromArtifact


class ControlTransformer(pl.LightningModule, LoadableFromArtifact):
    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.episode_builder: EpisodeBuilder = instantiate(self.hparams.episode_builder)
        self.encoder: Module = instantiate(self.hparams.encoder)
        self.objectives: ModuleDict = instantiate(self.hparams.objectives)
        self.objective_scheduler: ObjectiveScheduler | None = instantiate(
            self.hparams.get("objective_scheduler")
        )
        if self.objective_scheduler:
            self.objective_scheduler.verify(self.objectives.keys())  # pyright: ignore

    @override
    @_restricted_classmethod
    def load_from_checkpoint(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        checkpoint_path: str | PathLike[str],
        **kwargs,
    ) -> Self:
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
            name: objective.predict(inputs, self.episode_builder, self.encoder)
            for name, objective in self.objectives.items()
        })

        return TensorDict.from_dict(
            {
                "inputs": inputs,
                "predictions": predictions,
            },
            batch_size=batch.batch_size,
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
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(cfg.pop("scheduler"), optimizer=optimizer)
            result["lr_scheduler"] = {"scheduler": scheduler, **cfg}

        return result

    def _populate_logit_bias(self):
        # https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_Imbalanced_Data_With_Vision_Transformers_CVPR_2023_paper.pdf
        # Section 3.2

        logit_bias_losses: defaultdict[
            tuple[Modality, str],
            dict[str, Module],
        ] = defaultdict(dict)

        for objective_key, objective in self.objectives.items():
            # NOTE: this wouldn't work for e.g. copycat
            if hasattr(objective, "losses"):
                for loss_key, loss in objective.losses.flatten():
                    if hasattr(loss, "logit_bias") and loss.logit_bias is None:
                        logit_bias_losses[loss_key][objective_key] = loss

        if logit_bias_losses:
            col_names = {
                k: v
                for (k, v) in (
                    ("VehicleMotion_gas_pedal_normalized", "gas_pedal"),
                    ("VehicleMotion_brake_pedal_normalized", "brake_pedal"),
                    ("VehicleMotion_steering_angle_normalized", "steering_angle"),
                    ("VehicleMotion_speed", "speed"),
                    ("VehicleState_turn_signal", "turn_signal"),
                )
                if v in {name for (_, name) in logit_bias_losses.keys()}
            }

            modalities = {
                "gas_pedal": Modality.CONTINUOUS,
                "brake_pedal": Modality.CONTINUOUS,
                "steering_angle": Modality.CONTINUOUS,
                "speed": Modality.CONTINUOUS,
                "turn_signal": Modality.DISCRETE,
            }

            dataset = self.trainer.datamodule.train_dataloader().dataset
            metadata_df = dataset._metadata.select(*col_names.keys()).rename(col_names)
            values = TensorDict.from_dict(
                {
                    (modalities[k], k): v.to_numpy(zero_copy_only=False, writable=False)
                    for k, v in metadata_df.to_dict().items()
                },
                device=self.device,
                batch_size=[],
            )

            labels = values.unsqueeze(0).named_apply(
                lambda k, v: self.episode_builder.tokenizers.get(k)(v),
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

            logit_bias = bincounts.apply(lambda x: ((x + 1) / x.sum()).log())

            for loss_key, losses in logit_bias_losses.items():
                for objective_key, loss in losses.items():
                    logger.debug(
                        "setting logit bias",
                        objective=objective_key,
                        loss=(*loss_key, loss.__class__),
                    )

                    loss.logit_bias = logit_bias[loss_key]

    @override
    def on_fit_start(self) -> None:
        self._populate_logit_bias()
