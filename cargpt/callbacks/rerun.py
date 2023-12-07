from typing import Literal, Sequence

import pytorch_lightning as pl
import rerun as rr
import torch
from einops import rearrange
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import LazyStackedTensorDict, TensorDict


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

    def on_predict_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
    ) -> None:
        rr.init(**self.rerun_init_kwargs)

    def write_on_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        prediction: LazyStackedTensorDict,
        _batch_indices: Sequence[int] | None,
        batch: TensorDict,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        input, pred = torch.unbind(prediction, dim=1)  # pyright: ignore

        # take last timesteps
        input = input.apply(lambda x: x[:, -1])
        meta = batch["meta"].exclude("drive_id").apply(lambda x: x[:, -1])

        # permute images for logging
        input = input.update(
            input.select("image").apply(lambda x: rearrange(x, "b c h w -> b h w c"))
        )

        for input_elem, pred_elem, meta_elem in torch.stack(
            [input, pred, meta],
            dim=1,
        ).cpu():
            for (t, n), img in input_elem.select("image").items(True, True):  # pyright: ignore
                frame_idx = meta_elem.pop(f"{n}/ImageMetadata_frame_idx")  # pyright: ignore
                timestamp_ns = meta_elem.pop(f"{n}/ImageMetadata_time_stamp") * 1000  # pyright: ignore
                rr.set_time_sequence(f"{n}/frame_idx", int(frame_idx.item()))
                rr.set_time_nanos(f"{n}/timestamp", int(timestamp_ns))

                rr.log(f"input/{t}/{n}", rr.Image(img))

            for k, v in meta_elem.items(True, True):  # pyright: ignore
                rr.log(f"meta/{k}", rr.TimeSeriesScalar(v))

            for (t, n), v in input_elem.exclude("image").items(True, True):  # pyright: ignore
                rr.log(f"input/{t}/{n}", rr.TimeSeriesScalar(v))

            for (_, n), (gt, pred) in torch.stack([input_elem, pred_elem]).items(  # pyright: ignore
                True, True
            ):
                rr.log(f"{n}/gt", rr.TimeSeriesScalar(gt))
                rr.log(f"{n}/pred", rr.TimeSeriesScalar(pred))
