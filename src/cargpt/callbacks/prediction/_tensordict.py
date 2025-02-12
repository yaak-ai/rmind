from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, final, override

import orjson
import pytorch_lightning as pl
from pydantic import validate_call
from pytorch_lightning.callbacks import BasePredictionWriter
from rbyte.batch import Batch
from tensordict import TensorDict
from torch import Tensor

from cargpt.utils import monkeypatched


@final
class TensorDictPredictionWriter(BasePredictionWriter):
    """
    Example:

    ```yaml
    _target_: cargpt.callbacks.TensorDictPredictionWriter
    write_interval: batch
    path: ${hydra:run.dir}/predictions/{batch_idx}/
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
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)

        self._path = path
        self._writer = writer

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
        data = (
            prediction.select("input", "predictions")
            .auto_batch_size_(1)
            .update({"batch": batch.to_tensordict().auto_batch_size_(1)})
            .named_apply(self._filter, nested_keys=True)
        )

        path = Path(
            self._path.format(batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._patch_orjson_dumps():
            self._writer(data, path.resolve().as_posix())

    @staticmethod
    def _filter(key: tuple[str, ...], tensor: Tensor) -> Tensor | None:
        # TODO: configurize
        match key:
            case ("batch", "data", name) if any(
                map(name.endswith, ("idx", "time_stamp"))
            ):
                return tensor

            case ("batch", "meta", *_):
                return tensor

            case ("predictions", *_):
                return tensor

            case _:
                return None

    @classmethod
    def _patch_orjson_dumps(cls):
        """
        WARN: hacky af workaround

        `TensorDict.memmap` uses `orjson.dumps` to serialize metadata,
        which may contain StrEnum keys. `orjson.dumps` doesn't handle
        those without the `orjson.OPT_NON_STR_KEYS` option set (see
        https://github.com/ijl/orjson/issues/414).
        """
        return monkeypatched(
            orjson, "dumps", partial(orjson.dumps, option=orjson.OPT_NON_STR_KEYS)
        )
