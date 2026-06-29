from functools import reduce
from typing import Any, Self, final, override

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pydantic import ConfigDict, InstanceOf, validate_call
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer

from rmind.components import optimizers
from rmind.components.lr_schedulers import LRSchedulerHydraConfig
from rmind.components.vq import ResidualVQ
from rmind.config import HydraConfig, init_hydra_param
from rmind.utils._wandb import LoadableFromArtifact

type Path = tuple[str, ...]


class WaypointsTokenizer(pl.LightningModule, LoadableFromArtifact):
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
        vq_weight: float = 5.0,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        **_legacy_hparams: Any,
    ) -> None:
        # HOTFIX: `**_legacy_hparams` swallows (and ignores) `__init__` kwargs saved
        # by older tokenizer checkpoints but absent from this model -- e.g.
        # `normalize` (a normalization submodule) and `commitment_weight`. Their
        # leftover state_dict entries are dropped by `strict=False` on load.
        super().__init__()

        hparams: dict[str, Any] = {}

        self.input_transform = init_hydra_param(
            hparams, "input_transform", input_transform
        )
        self.encoder = init_hydra_param(hparams, "encoder", encoder)
        self.quantizer: ResidualVQ = init_hydra_param(hparams, "quantizer", quantizer)
        self.decoder = init_hydra_param(hparams, "decoder", decoder)

        self.waypoints: Path = waypoints
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.vq_weight = vq_weight
        hparams |= {
            "waypoints": waypoints,
            "num_waypoints": num_waypoints,
            "waypoint_dim": waypoint_dim,
            "vq_weight": vq_weight,
        }

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @override
    @_restricted_classmethod
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def load_from_checkpoint(
        cls,  # noqa: N805
        checkpoint_path: _PATH,
        *,
        map_location: _MAP_LOCATION_TYPE = None,
        strict: bool | None = False,
        weights_only: bool | None = False,
        **kwargs: Any,
    ) -> Self:  # ty:ignore[invalid-method-override]
        # `weights_only` defaults to False because the checkpoint's saved hparams
        # contain non-tensor objects.
        # HOTFIX: `strict` defaults to False so artifacts whose state_dict carries
        # extra keys absent from this model (e.g. a `normalization` submodule added
        # in a newer tokenizer) still load instead of raising on unexpected keys.
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    @property
    def _input_dim(self) -> int:
        return self.num_waypoints * self.waypoint_dim

    def encode(self, waypoints: Tensor) -> Tensor:
        *batch, _num_waypoints, _waypoint_dim = waypoints.shape
        w = waypoints.reshape(-1, self._input_dim)
        _codes, z_q, _ = self.quantizer(self.encoder(w))
        return z_q.reshape(*batch, z_q.shape[-1])

    def _gather_waypoints(self, batch: Any) -> Tensor:
        inputs = self.input_transform(batch)
        waypoints = reduce(lambda acc, key: acc[key], self.waypoints, inputs)
        return waypoints.reshape(-1, self._input_dim)

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        w = self._gather_waypoints(batch)

        z = self.encoder(w)
        codes, z_q, commit = self.quantizer(z)
        w_hat = self.decoder(z + (z_q - z).detach())  # straight-through

        recon = F.l1_loss(w_hat, w)
        total = recon + self.vq_weight * commit

        metrics = {"recon": recon, "commit": commit, "total": total}
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


@final
class WaypointsLatentTokenizer(Module):
    """Encode a per-frame waypoint path into ONE residual-VQ latent token.

    Wraps a pretrained `WaypointsTokenizer`. `forward` maps waypoints
    `(*batch, num_waypoints, waypoint_dim)` -> `(*batch, 1, dim)`: the quantized
    latent `z_q` (the sum of the per-level codebook vectors) with a singleton
    token axis, so the policy episode sees a single waypoint token per timestep
    instead of one token per waypoint.

    Freezing the wrapped tokenizer (so its residual-VQ codebook EMA never updates
    during downstream training) is the job of the `ModuleFreezer` callback, not
    this module -- see `config/trainer/callbacks`.
    """

    def __init__(self, *, tokenizer: WaypointsTokenizer, **_legacy_kwargs: Any) -> None:
        # HOTFIX: `**_legacy_kwargs` swallows kwargs saved by older checkpoints but
        # no longer used here -- e.g. `freeze`, now handled by the `ModuleFreezer`
        # callback (see the class docstring) rather than this module.
        super().__init__()
        self.tokenizer = tokenizer

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        z_q = self.tokenizer.encode(waypoints)  # (*batch, dim)
        return z_q.unsqueeze(-2)  # (*batch, 1, dim) -- one latent token per frame
