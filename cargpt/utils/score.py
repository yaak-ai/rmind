from typing import Any, Tuple

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor


class SafetyScore(pl.LightningModule):
    def __init__(
        self,
        inference_model: DictConfig,
    ):
        super().__init__()
        self.model = instantiate(inference_model)
        self.model.freeze()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        score = self.model.score_step(batch, batch_idx)

        return score
