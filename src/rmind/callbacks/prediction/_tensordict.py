from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, final, override

import orjson
import pytorch_lightning as pl
from pydantic import validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from tensordict import TensorClass, TensorDict

from rmind.utils import monkeypatched


@final
class TensorDictPredictionWriter(BasePredictionWriter):
    """
    ```yaml
    _target_: rmind.callbacks.TensorDictPredictionWriter
    write_interval: batch
    path: ${hydra:run.dir}/predictions/{batch_idx}/
    select:
      - [batch, data, ImageMetadata.cam_front_left.time_stamp]
      - [batch, meta]
      - predictions
    writer:
      _target_: tensordict.memmap
      _partial_: true
      copy_existing: true
    ```
    """

    @validate_call
    def __init__(
        self,
        path: str,
        writer: Callable[[TensorDict, str], None],
        select: Sequence[str | tuple[str, ...]] | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)

        self._path = path
        self._writer = writer
        self._select = select

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

        if self._select is not None:
            data = data.select(*self._select)

        path = Path(
            self._path.format(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._patched_orjson_dumps():
            self._writer(data, path.resolve().as_posix())

    @classmethod
    def _patched_orjson_dumps(cls):  # noqa: ANN206
        # `TensorDict.memmap` uses `orjson.dumps` to serialize metadata,
        # which may contain StrEnum keys. `orjson.dumps` doesn't handle
        # those without the `orjson.OPT_NON_STR_KEYS` option set (see
        # https://github.com/ijl/orjson/issues/414).
        return monkeypatched(
            orjson, "dumps", partial(orjson.dumps, option=orjson.OPT_NON_STR_KEYS)
        )
