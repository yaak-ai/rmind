from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, final, override

import polars as plr  # noqa: ICN001
import pytorch_lightning as pl
from pydantic import validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorClass, TensorDict


@final
class DataFramePredictionWriter(BasePredictionWriter):
    """
    ```yaml
    _target_: cargpt.callbacks.DataFramePredictionWriter
    write_interval: batch
    path: ${hydra:run.dir}/predictions/{batch_idx}.parquet
    select:
      - [batch, data, ImageMetadata.cam_front_left.time_stamp]
      - [input, continuous]
      - [input, discrete]
      - [predictions, forward_dynamics, score_l1, continuous]
    writer:
        _target_: polars.DataFrame.write_parquet
        _partial_: true
    ```
    """

    @validate_call
    def __init__(
        self,
        *,
        path: str,
        writer: Callable[[plr.DataFrame, str], None],
        select: Sequence[str | tuple[str, ...]] | None = None,
        separator: str = "/",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval=write_interval)

        self._path = path
        self._writer = writer
        self._select = select
        self._separator = separator

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: TensorClass,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            prediction.clone(recurse=False)
            .update({"batch": batch.clone(recurse=False).to_tensordict()})
            .auto_batch_size_(1)
            .lock_()
        )

        if self._select is not None:
            data = data.select(*self._select)

        data = data.flatten_keys(self._separator).cpu()

        try:
            df = plr.from_numpy(data.to_struct_array())
        except:  # noqa: E722
            df = plr.from_dict(data.numpy())

        path = Path(
            self._path.format(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        self._writer(df, path.resolve().as_posix())
