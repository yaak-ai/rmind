from collections.abc import Sequence
from typing import Literal

import pytorch_lightning as pl
import rerun as rr
import torch
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict
from typing_extensions import override
from yaak_datasets import Batch

from cargpt.components.episode import Modality


class RerunPredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        **rerun_init_kwargs,
    ) -> None:
        rerun_init_kwargs.setdefault("application_id", "prediction")
        rerun_init_kwargs.setdefault("spawn", True)
        rerun_init_kwargs.setdefault("strict", True)
        self.rerun_init_kwargs = rerun_init_kwargs

        super().__init__(write_interval)

    @override
    def on_predict_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # TODO: use rerun blueprint API once available
        rr.init(**self.rerun_init_kwargs)

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        prediction = prediction.to(pl_module.device)
        inputs, predictions = prediction["inputs"], prediction["predictions"]
        inputs = inputs.update(
            inputs.select(Modality.IMAGE).apply(
                Rearrange("... c h w -> ... h w c"),  # CHW -> HWC for logging
            )
        )

        meta = batch.meta.exclude("drive_id")
        meta.batch_size = inputs.batch_size

        data = TensorDict.from_dict({
            "inputs": inputs,
            "meta": meta,
            "predictions": predictions,
        })

        for _batch in data.cpu():
            for timestep, elem in enumerate(_batch):
                for k, v in elem.items(include_nested=True, leaves_only=True):
                    match k:
                        case ("meta", k_meta):
                            entity_path = list(k)
                            match k_meta.split("/"):
                                case (_, "ImageMetadata_frame_idx"):
                                    rr.set_time_sequence("/".join(k), v.item())

                                case (_, "ImageMetadata_time_stamp"):
                                    rr.set_time_nanos("/".join(k), v.item() * 1000)

                                case _:
                                    rr.log(entity_path, rr.Scalar(v))

                        case ("inputs", modality, *_):
                            entity_path = list(k)
                            match modality:
                                case Modality.IMAGE:
                                    rr.log(entity_path, rr.Image(v))

                                case Modality.CONTINUOUS | Modality.DISCRETE:
                                    rr.log(entity_path, rr.Scalar(v))

                                case _:
                                    raise NotImplementedError

                        case (
                            "predictions",
                            objective,
                            "attention",
                            Modality.SPECIAL,
                            token_from,
                            modality_to,
                            token_to,
                        ):
                            # TODO: use blueprint API once available
                            # https://x.com/rerundotio/status/1768657557134934176

                            entity_path = ["attention", objective, token_from, token_to]
                            attn = v[timestep]
                            match modality_to:
                                case Modality.IMAGE:
                                    rr.log(entity_path, rr.Tensor(attn.view(10, 18)))

                                case _:
                                    # HACK: make rerun render it as a tensor
                                    rr.log(entity_path, rr.Tensor(torch.zeros(2, 2)))
                                    rr.log(entity_path, rr.Tensor(attn))

                        case _:
                            pass
