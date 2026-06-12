from collections.abc import Sequence
from typing import Any, override

import pytorch_lightning as pl
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch.utils._pytree import MappingKey, key_get  # noqa: PLC2701

from rmind.utils.pytree import path_to_key

logger = get_logger(__name__)


class FeatureReflector(Callback):
    """Negate a single coordinate of a batch feature at predict time.

    Directional ("mirror") probe: flipping the lateral waypoint coordinate
    should flip the predicted steering sign iff the model uses waypoints in
    the right direction. Permutation only proves *sensitivity*; this proves
    the sign of the dependency.

    ```yaml
    _target_: rmind.callbacks.FeatureReflector
    features:
      - [data, waypoints/xy_normalized]
    dim: -1     # the (x, y) axis
    index: 0    # 0 = lateral (left/right), 1 = longitudinal (forward)
    ```
    """

    @validate_call
    def __init__(
        self, *, features: Sequence[Sequence[str | int]], dim: int = -1, index: int = 0
    ) -> None:
        super().__init__()
        self._paths = [tuple(map(MappingKey, p)) for p in features]
        self._dim = dim
        self._index = index

    @override
    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        logger.info(
            "reflecting features",
            features=[path_to_key(p) for p in self._paths],
            dim=self._dim,
            index=self._index,
        )

    @override
    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for path in self._paths:
            tensor = key_get(batch, path)
            idx: list[Any] = [slice(None)] * tensor.ndim
            idx[self._dim] = self._index
            tensor[tuple(idx)] = -tensor[tuple(idx)]
