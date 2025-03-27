from collections.abc import Iterable
from typing import Any

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import ModelSummary as _ModelSummary
from torchinfo import summary
from typing_extensions import override


class ModelSummary(_ModelSummary):
    def __init__(
        self,
        col_width: int = 16,
        depth: int = 4,
        col_names: Iterable[str] = ("trainable", "num_params"),
        row_settings: Iterable[str] = ("var_names",),
        **kwargs: dict[str, Any],
    ) -> None:
        self._kwargs: dict[str, Any] = {
            "depth": depth,
            "col_width": col_width,
            "col_names": col_names,
            "row_settings": row_settings,
        } | kwargs

    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            summary_str = str(summary(pl_module, **self._kwargs, verbose=0))
            logger.info(f"\n{summary_str}")
