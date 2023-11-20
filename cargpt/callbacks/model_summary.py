from typing import Any, Dict, Iterable

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelSummary as _ModelSummary
from torchinfo import summary


class ModelSummary(_ModelSummary):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        col_width: int = 16,
        depth: int = 4,
        col_names: Iterable[str] = ("trainable", "num_params"),
        row_settings: Iterable[str] = ("var_names",),
        **kwargs,
    ) -> None:
        self._kwargs: Dict[str, Any] = {
            "depth": depth,
            "col_width": col_width,
            "col_names": col_names,
            "row_settings": row_settings,
        } | kwargs

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            _summary_str = str(summary(pl_module, **self._kwargs, verbose=0))
            logger.info(f"\n{_summary_str}")
