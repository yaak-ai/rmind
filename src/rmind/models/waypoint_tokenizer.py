from functools import reduce
from typing import Any, override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer

from rmind.components import optimizers
from rmind.components.nn import Identity
from rmind.components.vq import ResidualVQ
from rmind.config import HydraConfig
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.utils._wandb import LoadableFromArtifact

type Path = tuple[str, ...]


class WaypointTokenizer(pl.LightningModule, LoadableFromArtifact):
    """Residual-VQ waypoint tokenizer, VQ-BeT style (https://arxiv.org/pdf/2403.03181).

    Tokenizes each ego-frame future path (`num_waypoints` x `waypoint_dim`) into a
    tuple of residual-VQ codes -- the same scheme `ActionTokenizer` applies to the
    action chunk, here applied to waypoints. An encoder maps the flattened path to a
    latent, the residual quantizer discretizes it coarse-to-fine, and a decoder
    reconstructs the path; trained with an L1 reconstruction + VQ commitment loss.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        quantizer: HydraConfig[ResidualVQ] | InstanceOf[ResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        waypoints: Path,
        num_waypoints: int = 10,
        waypoint_dim: int = 2,
        normalize: HydraConfig[Module] | InstanceOf[Module] | None = None,
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

        if isinstance(normalize, HydraConfig):
            hparams["normalize"] = normalize.model_dump()
            normalize = normalize.instantiate()
        self.normalize: Module = normalize if normalize is not None else Identity()

        self.waypoints: Path = waypoints
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        hparams |= {
            "waypoints": waypoints,
            "num_waypoints": num_waypoints,
            "waypoint_dim": waypoint_dim,
            "commitment_weight": commitment_weight,
            "vq_weight": vq_weight,
        }

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @property
    def _input_dim(self) -> int:
        return self.num_waypoints * self.waypoint_dim

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        *batch, _num_waypoints, _waypoint_dim = waypoints.shape
        w = self.normalize(waypoints.reshape(-1, self._input_dim))
        codes, _, _ = self.quantizer(self.encoder(w))
        return codes.reshape(*batch, self.quantizer.num_quantizers)

    def invert(self, codes: Tensor) -> Tensor:
        *batch, num_quantizers = codes.shape
        z_q = self.quantizer.lookup(codes.reshape(-1, num_quantizers))
        w = self.decoder(z_q)
        return w.reshape(*batch, self.num_waypoints, self.waypoint_dim)

    def _gather_waypoints(self, batch: Any) -> Tensor:
        inputs = self.input_transform(batch)
        waypoints = reduce(lambda acc, key: acc[key], self.waypoints, inputs)
        return waypoints.reshape(-1, self._input_dim)

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        w = self.normalize(self._gather_waypoints(batch))

        z = self.encoder(w)
        codes, z_q, vq = self.quantizer(z)
        w_hat = self.decoder(z + (z_q - z).detach())  # straight-through

        recon = F.l1_loss(w_hat, w)
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
