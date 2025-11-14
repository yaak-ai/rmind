from collections.abc import Sequence
from typing import Any, Literal, final, override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.viz.loggers import RerunLogger
from tensordict import TensorDict


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
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            prediction.to_tensordict(retain_none=True)
            .update({"batch": TensorDict(batch)})
            .auto_batch_size_(1)
            .lock_()
        ).apply(lambda x: x.float() if x.is_floating_point() else x)

        self._logger.log(data)
