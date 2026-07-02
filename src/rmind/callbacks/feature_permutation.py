from collections.abc import Sequence
from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch.utils._pytree import MappingKey, key_get  # noqa: PLC2701

from rmind.utils.pytree import path_to_key

logger = get_logger(__name__)


class FeaturePermutator(Callback):
    """```yaml
    _target_: rmind.callbacks.FeaturePermutator
    features:
      - [data, meta/VehicleMotion/speed]
      - [data, meta/VehicleMotion/steering_angle_normalized]
    seed: 42
    ```.
    """

    @validate_call
    def __init__(
        self, *, features: Sequence[Sequence[str | int]], seed: int = 42
    ) -> None:
        super().__init__()
        self._paths = [tuple(map(MappingKey, p)) for p in features]
        self._seed = seed
        self._generator: torch.Generator | None = None

    @override
    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._generator = torch.Generator(device=pl_module.device).manual_seed(
            self._seed
        )
        logger.info(
            "permuting features",
            features=[path_to_key(p) for p in self._paths],
            seed=self._seed,
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
            perm = torch.randperm(
                tensor.shape[0], generator=self._generator, device=tensor.device
            )
            *prefix, last = path
            key_get(batch, tuple(prefix))[last.key] = tensor[perm]
