from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback

from rmind.callbacks.loggers.common import _get_wandb_loggers

_BYTES_PER_GB = 1024**3


class GpuMemoryStatsCallback(Callback):
    """Logs per-step peak CUDA memory (not a time-averaged nvidia-smi snapshot).

    `torch.cuda.max_memory_allocated`/`max_memory_reserved` are reset at the start of
    every train/validation batch and logged every `every_n_steps` steps, so short-lived
    activation/checkpoint-recompute spikes that a 15s system-metrics sample would miss
    are visible in wandb.
    """

    @validate_call
    def __init__(self, every_n_steps: int = 100) -> None:
        self._every_n_steps = every_n_steps

    @staticmethod
    def _reset() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _log(
        trainer: pl.Trainer, pl_module: pl.LightningModule, *, prefix: str
    ) -> None:
        if not torch.cuda.is_available():
            return

        metrics = {
            f"system/{prefix}/max_memory_allocated_gb": torch.cuda.max_memory_allocated()
            / _BYTES_PER_GB,
            f"system/{prefix}/max_memory_reserved_gb": torch.cuda.max_memory_reserved()
            / _BYTES_PER_GB,
        }
        for wandb_logger in _get_wandb_loggers(pl_module):
            wandb_logger.log_metrics(metrics, step=trainer.global_step)

    @override
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._reset()

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self._every_n_steps == 0:
            self._log(trainer, pl_module, prefix="train")

    @override
    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._reset()

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self._every_n_steps == 0:
            self._log(trainer, pl_module, prefix="val")
