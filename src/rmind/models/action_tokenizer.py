from collections.abc import Mapping
from typing import Any, ClassVar, Literal, override

import pytorch_lightning as pl
import torch
from pydantic import BaseModel, ConfigDict, InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._pytree import MappingKey, tree_leaves, tree_map  # noqa: PLC2701

from rmind.components import optimizers
from rmind.components.objectives.base import Targets
from rmind.components.vq import ResidualVQ
from rmind.config import HydraConfig
from rmind.utils._wandb import LoadableFromArtifact
from rmind.utils.pytree import key_get_default


class LRSchedulerHydraConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    interval: Literal["epoch", "step"]
    scheduler: HydraConfig[LRScheduler]


class ActionTokenizer(pl.LightningModule, LoadableFromArtifact):
    """Residual-VQ action tokenizer VQ-BeT
    https://arxiv.org/pdf/2403.03181.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        quantizer: HydraConfig[ResidualVQ] | InstanceOf[ResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        targets: Targets,
        commitment_weight: float = 1.0,
        vq_weight: float = 5.0,
        # Per-axis reconstruction weighting for a ZERO-INFLATED actuation.
        # `fork1` sits at exactly 0.000 for 91.7% of messages, at +/-100% for 4.5%,
        # and spreads over 804 distinct intermediate values for the remaining 3.4%
        # (measured, 8 jobs / 91,331 `linde/fork` messages). An unweighted L1 over
        # that channel is minimized by emitting zero, and the first tokenizer did
        # exactly that: holdout L1 0.0563 overall but 0.8988 on the 5.6% of samples
        # with |cmd| > 0.5, reconstructing events at mean |a_hat| = 0.0007 against a
        # target of 0.8991. Native-rate sampling does NOT fix it -- decimating fork1
        # from 90 Hz to 10 Hz preserves 72 of 74 event runs and leaves the ratio at
        # 6.5%, because fork ramps last seconds rather than milliseconds -- so the
        # imbalance has to be paid for in the loss.
        #
        # `event_weight[axis]` multiplies elements whose |target| exceeds
        # `event_threshold`. Inverse frequency (0.935/0.065 ~ 14.4) equalizes the
        # total contribution of the quiet and event populations.
        event_threshold: float = 0.05,
        event_weight: Mapping[str, float] | None = None,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        if isinstance(input_transform, HydraConfig):
            hparams["input_transform"] = input_transform.model_dump()
            input_transform = input_transform.instantiate()
        self.input_transform = input_transform

        if isinstance(encoder, HydraConfig):
            hparams["encoder"] = encoder.model_dump()
            encoder = encoder.instantiate()
        self.encoder = encoder

        if isinstance(quantizer, HydraConfig):
            hparams["quantizer"] = quantizer.model_dump()
            quantizer = quantizer.instantiate()
        self.quantizer: ResidualVQ = quantizer

        if isinstance(decoder, HydraConfig):
            hparams["decoder"] = decoder.model_dump()
            decoder = decoder.instantiate()
        self.decoder = decoder

        self.targets: Targets = targets
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        self.event_threshold = event_threshold
        self.event_weight: dict[str, float] = dict(event_weight or {})
        hparams["targets"] = targets
        hparams["commitment_weight"] = commitment_weight
        hparams["vq_weight"] = vq_weight
        hparams["event_threshold"] = event_threshold
        hparams["event_weight"] = self.event_weight

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @override
    def forward(self, action: Tensor) -> Tensor:

        action = action.flatten(-2, -1)
        *batch, action_dim = action.shape
        z = self.encoder(self._normalize(action).reshape(-1, action_dim))
        codes, _, _ = self.quantizer(z)
        return codes.reshape(*batch, self.quantizer.num_quantizers)

    def invert(self, codes: Tensor) -> Tensor:
        *batch, num_quantizers = codes.shape
        z_q = self.quantizer.lookup(codes.reshape(-1, num_quantizers))
        return self.decoder(z_q).reshape(*batch, -1)

    @property
    def _action_features(self) -> int:
        return len(tree_leaves(self.targets, is_leaf=lambda x: isinstance(x, tuple)))

    def _normalize(self, action: Tensor) -> Tensor:
        """Normalize a raw stacked action vector via the built-in per-field normalizer."""
        *batch, action_dim = action.shape
        columns = iter(action.reshape(*batch, -1, self._action_features).unbind(-1))
        structured = tree_map(
            lambda _path: next(columns),
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        normalized = self.input_transform[-1](structured)
        return torch.stack(tree_leaves(normalized), dim=-1).reshape(*batch, action_dim)

    def _gather_actions(self, inputs: Any) -> Tensor:
        gathered = tree_map(
            lambda path: key_get_default(
                inputs, tuple(MappingKey(part) for part in path), None
            ),
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        action = torch.stack(tree_leaves(gathered), dim=-1)
        return action.reshape(action.shape[0], -1)

    def _axis_order(self) -> list[str]:
        """Axis names in the order `_gather_actions` stacks them (fastest-varying)."""
        return [name for modality in self.targets.values() for name in modality]

    def _weighted_l1(self, a_hat: Tensor, a: Tensor) -> Tensor:
        """L1, with rare-event elements upweighted per axis (see `event_weight`).

        `_gather_actions` stacks axes on the LAST dim before flattening, so element
        `i` of the flat action is `(timestep, axis) = divmod(i, num_axes)` and the
        axis index is the fast one. Weights are built in that layout.
        """
        if not self.event_weight:
            return F.l1_loss(a_hat, a)

        axes = self._axis_order()
        boost = a.new_tensor([self.event_weight.get(name, 1.0) for name in axes])
        flat = a.reshape(a.shape[0], -1, len(axes))
        is_event = flat.abs() > self.event_threshold
        w = torch.where(is_event, boost.expand_as(flat), torch.ones_like(flat))
        w = w.reshape(a.shape)

        # normalized so the loss stays on the same scale as plain L1 -- otherwise
        # `vq_weight`/`commitment_weight` would need retuning alongside it
        return ((a_hat - a).abs() * w).sum() / w.sum()

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        inputs = self.input_transform(batch)
        a = self._gather_actions(inputs)

        z = self.encoder(a)
        codes, z_q, vq = self.quantizer(z)
        a_hat = self.decoder(z + (z_q - z).detach())

        recon = self._weighted_l1(a_hat, a)
        total = recon + self.vq_weight * (
            vq["codebook"] + self.commitment_weight * vq["commit"]
        )

        metrics = {
            "recon": recon,
            "codebook": vq["codebook"],
            "commit": vq["commit"],
            "total": total,
        }
        perplexity = self.quantizer.perplexity(codes)
        for q in range(self.quantizer.num_quantizers):
            metrics[f"perplexity/q{q}"] = perplexity[q]

        return total, metrics

    @override
    def training_step(self, batch: Any, _batch_idx: int) -> STEP_OUTPUT:
        total, metrics = self._step(batch)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()}, sync_dist=True)
        return {"loss": total}

    @override
    def validation_step(self, batch: Any, _batch_idx: int) -> STEP_OUTPUT:
        total, metrics = self._step(batch)
        if not self.trainer.sanity_checking:
            self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)
        return {"loss": total}

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer is None:
            msg = "optimizer not specified"
            raise ValueError(msg)

        match self.optimizer.target:
            case optimizers.SelectiveAdamW:
                optimizer = self.optimizer.instantiate(module=self)
            case _:
                optimizer = self.optimizer.instantiate(params=self.parameters())

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler.scheduler.instantiate(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler} | self.lr_scheduler.model_dump(
                exclude={"scheduler"}
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return {"optimizer": optimizer}
