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
        # raw batch data (e.g. route waypoints) is per-tick and needs slicing
        # to one reference tick before it's directly comparable/plottable
        # alongside a model's trajectory prediction -- unlike that, this is
        # read straight from `batch`, so it works regardless of whether the
        # loaded checkpoint's own episode_builder happens to expose it as a
        # model input (predict()/episode.get() can't see raw batch fields the
        # episode_builder wasn't configured to remap; this callback can).
        route_key: tuple[str, ...] | None = None,
        route_tick: int | None = None,
        route_scale: float = 1.0,
    ) -> None:
        super().__init__(write_interval)
        self._logger = logger
        self._route_key = route_key
        self._route_tick = route_tick
        self._route_scale = route_scale

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
        batch_td = TensorDict(batch)  # ty:ignore[invalid-argument-type]
        if self._route_key is not None and self._route_tick is not None:
            route = batch_td[self._route_key][:, self._route_tick] * self._route_scale
            batch_td = batch_td.update({("data", "route_local"): route})

        data = (
            prediction
            .to_tensordict(retain_none=True)
            .update({"batch": batch_td})
            .auto_batch_size_(1)
            .lock_()
        ).apply(lambda x: x.float() if x.is_floating_point() else x)

        self._logger.log(data)
