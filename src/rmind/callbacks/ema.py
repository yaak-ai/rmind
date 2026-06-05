from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)


class EMAWeights(Callback):
    """Exponential moving average over trainable parameters.

    Flow-matching losses are noisy per batch (random flow time and noise per
    visit), so the raw weights orbit the optimum instead of settling into it;
    sampling then integrates that wobbly field over many steps. The EMA shadow
    is the time-average of the orbit — a smoother field at zero inference cost.

    Behavior:
    - tracks only ``requires_grad`` parameters (frozen modules cost nothing);
    - updates the shadow after every train batch (with warmup so early steps
      don't dominate the average);
    - swaps the shadow in for validation and restores the raw weights after,
      so val metrics measure the averaged field while training continues on
      the raw weights;
    - persists the shadow via callback state — it is stored in checkpoints
      under ``callbacks`` without replacing the canonical raw ``state_dict``.
    """

    @validate_call
    def __init__(self, decay: float = 0.999, warmup: bool = True) -> None:
        if not 0.0 < decay < 1.0:
            msg = f"decay must be in (0, 1), got {decay}"
            raise ValueError(msg)

        self.decay = decay
        self.warmup = warmup
        self._shadow: dict[str, Tensor] = {}
        self._backup: dict[str, Tensor] = {}
        self._updates: int = 0

    @staticmethod
    def _trainable(pl_module: pl.LightningModule) -> list[tuple[str, Tensor]]:
        return [
            (name, param)
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        ]

    @override
    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        trainable = self._trainable(pl_module)
        if self._shadow:  # resumed from checkpoint
            self._shadow = {
                name: tensor.to(pl_module.device)
                for name, tensor in self._shadow.items()
            }
            if set(self._shadow) != {name for name, _ in trainable}:
                msg = "EMA state from checkpoint does not match trainable params"
                raise ValueError(msg)
        else:
            self._shadow = {
                name: param.detach().clone() for name, param in trainable
            }
        logger.info(
            "ema tracking trainable params",
            count=len(self._shadow),
            decay=self.decay,
            warmup=self.warmup,
        )

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._updates += 1
        decay = (
            min(self.decay, (1.0 + self._updates) / (10.0 + self._updates))
            if self.warmup
            else self.decay
        )
        with torch.no_grad():
            for name, param in self._trainable(pl_module):
                self._shadow[name].lerp_(param.detach(), 1.0 - decay)

    @override
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._shadow:  # validate() without fit(): nothing to swap
            return
        with torch.no_grad():
            for name, param in self._trainable(pl_module):
                self._backup[name] = param.detach().clone()
                param.copy_(self._shadow[name])

    @override
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._backup:
            return
        with torch.no_grad():
            for name, param in self._trainable(pl_module):
                param.copy_(self._backup[name])
        self._backup = {}

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "shadow": self._shadow,
            "updates": self._updates,
            "decay": self.decay,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._shadow = state_dict["shadow"]
        self._updates = state_dict["updates"]
