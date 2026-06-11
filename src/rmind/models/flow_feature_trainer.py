"""Decoder-only trainer over a frozen-encoder feature cache.

The encoder is frozen in every flow finetune, so the condition tokens per
frame are deterministic constants. flow_cache_features.py precomputes them;
this LightningModule trains the FlowPolicyObjective directly from the cache
via compute_metrics_from — no images, no episode builder, no encoder forward.
~An order of magnitude faster per step; intended for lever SCREENING.

Action-history dropout is exact under caching: the cache stores both condition
variants (normal / history-zeroed), and hist_dropout_p picks per sample at
train time — same semantics as the ActionHistoryDropout callback on the full
path.

Checkpoint note: state_dict keys are `objectives.policy.*`, matching
ControlTransformer, so a screened winner's decoder weights can be stitched
into a full checkpoint later. The cached path itself cannot run predict
(no encoder); confirm winners on the full pipeline.
"""

from typing import Any

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from torch.nn import ModuleDict

from rmind.components.objectives.flow_policy import FlowPolicyObjective

logger = get_logger(__name__)


class FlowFeatureTrainer(pl.LightningModule):
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        *,
        objective: InstanceOf[FlowPolicyObjective],
        optimizer: Any,
        lr_scheduler: Any | None = None,
        hist_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= hist_dropout_p <= 1.0:
            msg = f"hist_dropout_p must be in [0, 1], got {hist_dropout_p}"
            raise ValueError(msg)
        # Key layout matches ControlTransformer (objectives.policy.*) so cached
        # checkpoints can be stitched into full ones.
        self.objectives = ModuleDict({"policy": objective})
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        self.hist_dropout_p = hist_dropout_p

    @property
    def objective(self) -> FlowPolicyObjective:
        return self.objectives["policy"]

    def _condition(self, batch: dict, *, train: bool) -> torch.Tensor:
        cond = batch["cond"].float()
        if train and self.hist_dropout_p > 0.0:
            drop = (
                torch.rand(cond.shape[0], device=cond.device) < self.hist_dropout_p
            )
            cond = torch.where(
                drop[:, None, None], batch["cond_hist0"].float(), cond
            )
        return cond

    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        metrics = self.objective.compute_metrics_from(
            condition_tokens=self._condition(batch, train=True),
            target_actions=batch["target_actions"],
        )
        loss = metrics["loss"]
        self.log_dict(
            {f"train/policy/{k}": v for k, v in metrics.items()},
            batch_size=batch["cond"].shape[0],
        )
        self.log("train/loss/total", loss, batch_size=batch["cond"].shape[0])
        return loss

    def validation_step(self, batch: dict, _batch_idx: int) -> None:
        metrics = self.objective.compute_metrics_from(
            condition_tokens=self._condition(batch, train=False),
            target_actions=batch["target_actions"],
        )
        self.log_dict(
            {f"val/policy/{k}": v for k, v in metrics.items()},
            batch_size=batch["cond"].shape[0],
        )
        self.log(
            "val/loss/total", metrics["loss"], batch_size=batch["cond"].shape[0]
        )

    def configure_optimizers(self):  # noqa: ANN201
        # Robust to both forms: hydra recursive instantiation may have already
        # turned the _partial_ configs into functools.partial callables, or
        # they may still be raw DictConfigs.
        from hydra.utils import instantiate

        opt = self.optimizer_cfg
        optimizer = opt(module=self) if callable(opt) else instantiate(opt, module=self)
        if self.lr_scheduler_cfg is None:
            return optimizer
        sch = self.lr_scheduler_cfg["scheduler"]
        scheduler = sch(optimizer=optimizer) if callable(sch) else instantiate(sch, optimizer=optimizer)
        interval = self.lr_scheduler_cfg.get("interval", "step")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }
