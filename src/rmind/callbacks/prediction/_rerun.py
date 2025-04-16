from collections.abc import Sequence
from typing import Literal, final, override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.viz.loggers import RerunLogger
from tensordict import TensorClass, TensorDict


@final
class RerunPredictionWriter(BasePredictionWriter):
    @validate_call
    def __init__(
        self,
        logger: InstanceOf[RerunLogger],
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        self._logger = logger

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: TensorDict | TensorClass,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            prediction.clone(recurse=False)
            .update({"batch": batch.clone(recurse=False).to_tensordict()})
            .auto_batch_size_(1)
            .lock_()
        )

        self._logger.log(data)
