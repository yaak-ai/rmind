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

from rmind.components import optimizers  # noqa: PLC0415
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
    https://arxiv.org/pdf/2403.03181
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
        hparams["targets"] = targets
        hparams["commitment_weight"] = commitment_weight
        hparams["vq_weight"] = vq_weight

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @override
    def forward(self, action: Tensor) -> Tensor:
        *batch, action_dim = action.shape
        z = self.encoder(action.reshape(-1, action_dim))
        codes, _, _ = self.quantizer(z)
        return codes.reshape(*batch, self.quantizer.num_quantizers)

    def invert(self, codes: Tensor) -> Tensor:
        *batch, num_quantizers = codes.shape
        z_q = self.quantizer.lookup(codes.reshape(-1, num_quantizers))
        return self.decoder(z_q).reshape(*batch, -1)

    def _gather_actions(self, inputs: Any) -> Tensor:
        gathered = tree_map(
            lambda path: key_get_default(
                inputs, tuple(MappingKey(part) for part in path), None
            ),
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        action = torch.stack(tree_leaves(gathered), dim=-1)  # (B, T, action_dim)
        return action.reshape(-1, action.shape[-1])

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        inputs = self.input_transform(batch)
        a = self._gather_actions(inputs)

        z = self.encoder(a)
        codes, z_q, vq = self.quantizer(z)
        a_hat = self.decoder(z + (z_q - z).detach())

        recon = F.l1_loss(a_hat, a)
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
