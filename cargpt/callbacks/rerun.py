from typing import Literal, Sequence

import pytorch_lightning as pl
import rerun as rr
from einops.layers.torch import Rearrange
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorDict

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

    def on_predict_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
    ) -> None:
        rr.init(**self.rerun_init_kwargs)

    def write_on_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        result: TensorDict,
        _batch_indices: Sequence[int] | None,
        batch: TensorDict,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        result = result.to(pl_module.device)
        inputs, predictions = result["inputs"], result["predictions"]
        inputs = inputs.update(
            inputs.select(Modality.IMAGE).apply(
                Rearrange("... c h w -> ... h w c"),  # CHW -> HWC for logging
            )
        )

        meta = batch.meta.exclude("drive_id")  # pyright: ignore
        meta.batch_size = inputs.batch_size

        data = TensorDict.from_dict({
            "inputs": inputs[:, 1:],  # t-1 deltas -- TODO: make this more general?
            "meta": meta[:, 1:],
            "predictions": predictions,
        }).flatten(0, 1)  # b t ... -> (b t) ...

        image_keys = [("inputs", Modality.IMAGE)]

        for elem in data.cpu():
            for nested_key, tensor in elem.select(*image_keys).items(True, True):
                *_, camera = nested_key
                rr.set_time_sequence(
                    (k := f"{camera}/ImageMetadata_frame_idx"),
                    elem.pop(("meta", k)).item(),
                )
                rr.set_time_nanos(
                    (k := f"{camera}/ImageMetadata_time_stamp"),
                    elem.pop(("meta", k)).item() * 1000,  # us -> ns
                )

                rr.log(list(nested_key), rr.Image(tensor))

            for nested_key, tensor in elem.exclude(*image_keys).items(True, True):
                rr.log(list(nested_key), rr.Scalar(tensor))
