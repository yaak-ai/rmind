from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, final, override

import polars as plr  # noqa: ICN001
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.batch import Batch
from tensordict import TensorDict
from torch import Tensor

from cargpt.components.episode import Modality
from cargpt.components.objectives.base import PredictionResultKey


@final
class DataFramePredictionWriter(BasePredictionWriter):
    """
    A PyTorch Lightning callback for writing model predictions to a file via a
    polars DataFrame. May be useful for integration with external tools (e.g.
    https://github.com/yaak-ai/sexy).

    Example:

    ```yaml
    _target_: cargpt.callbacks.DataFramePredictionWriter
    write_interval: batch
    path: ${hydra:run.dir}/predictions/{batch_idx}.parquet
    writer:
        _target_: polars.DataFrame.write_parquet
        _partial_: true
    ```
    """

    def __init__(
        self,
        *,
        writer: Callable[[plr.DataFrame, str], None],
        path: str,
        separator: str = "/",
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval=write_interval)

        self._writer = writer
        self._path = path
        self._separator = separator

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: TensorDict,
        batch_indices: Sequence[int] | None,
        batch: Batch,  # pyright: ignore[reportInvalidTypeForm]
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        data = (
            prediction.select("inputs", "predictions")
            .auto_batch_size_(1)
            .update({"batch": batch.to_tensordict().auto_batch_size_(1)})  # pyright: ignore[reportAttributeAccessIssue]
            .named_apply(self._filter, nested_keys=True)
            .flatten_keys(self._separator)
            .cpu()
        )

        try:
            df = plr.from_numpy(data.to_struct_array())
        except:  # noqa: E722
            df = plr.from_dict(data.numpy())

        path = Path(
            self._path.format(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        self._writer(df, path.resolve().as_posix())

    @staticmethod
    def _filter(key: tuple[str, ...], tensor: Tensor) -> Tensor | None:
        # TODO: configurize
        match key:
            case ("batch", "data", name) if any(
                map(name.endswith, ("idx", "time_stamp"))
            ):
                return tensor

            case ("inputs", Modality.CONTINUOUS | Modality.DISCRETE, _name):
                return tensor

            case (
                "predictions",
                _objective,
                (
                    PredictionResultKey.GROUND_TRUTH
                    | PredictionResultKey.PREDICTION
                    | PredictionResultKey.PREDICTION_STD
                    | PredictionResultKey.SCORE_LOGPROB
                    | PredictionResultKey.SCORE_L1
                ),
                (Modality.CONTINUOUS | Modality.DISCRETE),
                _name,
            ):
                return tensor

            case _:
                return None
