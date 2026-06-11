"""Action-history dropout: anti-copycat training regularizer.

The action-history sensitivity probe showed the flow policy leans heavily on
past actions (perturbing them shifts predictions by 1.7-3.4x the baseline
error) — the classic behavior-cloning copycat shortcut (de Haan et al. 2019,
"Causal Confusion in Imitation Learning"; Wen et al. 2020 "Fighting Copycat
Agents"; cf. ChauffeurNet's past-motion dropout). On cruise frames,
continuation of history explains the target almost perfectly, so gradient
descent under-weights vision/route — which is exactly where held-out maneuver
onset fails (spike bias ~0.6: the model "continues" instead of initiating).

This callback zeroes the action-history batch fields for a random subset of
samples at train-batch start (before the episode is built), forcing the head
to predict from observations alone on those samples. Eval batches are
untouched. The encoder is frozen but not constant: zeroed inputs change its
summaries, so the decoder learns to operate in both regimes and the shortcut
weakens.

Usage (trainer callback):
    - _target_: rmind.callbacks.ActionHistoryDropout
      p: 0.5
"""

from typing import Any

import pytorch_lightning as pl
import torch
from structlog import get_logger

logger = get_logger(__name__)

DEFAULT_HISTORY_KEYS = (
    "meta/VehicleMotion/gas_pedal_normalized",
    "meta/VehicleMotion/brake_pedal_normalized",
    "meta/VehicleMotion/steering_angle_normalized",
)


class ActionHistoryDropout(pl.Callback):
    def __init__(
        self,
        p: float = 0.5,
        history_keys: tuple[str, ...] = DEFAULT_HISTORY_KEYS,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            msg = f"p must be in [0, 1], got {p}"
            raise ValueError(msg)
        self.p = p
        self.history_keys = history_keys

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.p <= 0.0:
            return
        data = batch.get("data") if isinstance(batch, dict) else None
        if data is None:
            return
        first = data.get(self.history_keys[0])
        if first is None:
            return
        # One mask per sample, shared across the action channels (drop the
        # whole action history jointly — partial history still leaks level).
        drop = torch.rand(first.shape[0], device=first.device) < self.p
        if not bool(drop.any()):
            return
        for key in self.history_keys:
            field = data.get(key)
            if field is not None:
                field[drop] = 0.0
