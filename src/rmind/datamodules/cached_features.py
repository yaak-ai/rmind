"""Dataset over a precomputed frozen-encoder feature cache.

Produced by rmind.scripts.flow_cache_features: per frame, the condition tokens
(normal + action-history-zeroed variants) and raw target actions. Used with a
plain torch DataLoader inside GenericDataModule for decoder-only training
(FlowFeatureTrainer) — no images, no episode builder, no encoder.
"""

from pathlib import Path

import torch
from structlog import get_logger
from torch.utils.data import Dataset

logger = get_logger(__name__)


class CachedFeaturesDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.cond: torch.Tensor = payload["cond"]
        self.cond_hist0: torch.Tensor = payload["cond_hist0"]
        self.target_actions: torch.Tensor = payload["target_actions"]
        self.meta: dict = payload.get("meta", {})
        logger.info(
            "loaded feature cache",
            path=str(path),
            frames=len(self),
            model_artifact=self.meta.get("model_artifact"),
        )

    def __len__(self) -> int:
        return self.cond.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "cond": self.cond[idx],
            "cond_hist0": self.cond_hist0[idx],
            "target_actions": self.target_actions[idx],
        }
