from typing import Any, override

import pytorch_lightning as pl
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback

from rmind.models.control_transformer import ControlTransformer, PredictionConfig


class PredictionConfigSetter(Callback):
    @validate_call
    def __init__(self, **prediction_kwargs: Any) -> None:
        super().__init__()
        self.prediction_config = PredictionConfig.model_validate(prediction_kwargs)
        self._prev_prediction_config: PredictionConfig | None = None

    @override
    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del trainer
        if not isinstance(pl_module, ControlTransformer):
            return
        self._prev_prediction_config = pl_module.prediction_config
        pl_module.prediction_config = self.prediction_config

    @override
    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del trainer
        if not isinstance(pl_module, ControlTransformer):
            return
        if self._prev_prediction_config is not None:
            pl_module.prediction_config = self._prev_prediction_config
        self._prev_prediction_config = None
