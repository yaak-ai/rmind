from typing import Any, Tuple

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


class Features(pl.LightningModule):
    def __init__(
        self,
        inference_model: DictConfig,
    ) -> None:
        super().__init__()
        self.model = instantiate(inference_model)

    def predict_step(
        self,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        return self.model.features_step(batch, batch_idx=0)
