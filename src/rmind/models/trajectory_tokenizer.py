from __future__ import annotations

from functools import reduce
from typing import Any, Self, final, override

import pytorch_lightning as pl
import torch
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
from rmind.components.vq import ResidualVQ
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.utils._wandb import LoadableFromArtifact
from rmind.utils.pytree import key_get_default

type Path = tuple[str, ...]


class TrajectoryTokenizer(pl.LightningModule, LoadableFromArtifact):
    """Residual-VQ autoencoder over a per-frame future trajectory.

    Target shape is `(*batch, num_poses, 3)` for ego-centric `(x, y, heading)`
    poses. The model flattens each trajectory into a single vector and learns a
    residual-VQ latent for it, then decodes back to continuous poses.

    This mirrors the route tokenizer pattern in `WaypointsTokenizer`, but is
    specialized for a dead-reckoned future trajectory target.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        quantizer: HydraConfig[ResidualVQ] | InstanceOf[ResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        trajectory: Path,
        num_poses: int = 5,
        trajectory_dim: int = 3,
        commitment_weight: float = 1.0,
        vq_weight: float = 5.0,
        heading_weight: float = 0.1,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        **_legacy_hparams: Any,
    ) -> None:
        # HOTFIX: older checkpoints can carry extra hparams we no longer use.
        super().__init__()

        hparams: dict[str, Any] = {}

        self.input_transform = init_hydra_param(
            hparams, "input_transform", input_transform
        )
        self.encoder = init_hydra_param(hparams, "encoder", encoder)
        self.quantizer: ResidualVQ = init_hydra_param(hparams, "quantizer", quantizer)
        self.decoder = init_hydra_param(hparams, "decoder", decoder)

        self.trajectory: Path = trajectory
        self.num_poses = num_poses
        self.trajectory_dim = trajectory_dim
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        self.heading_weight = heading_weight
        hparams |= {
            "trajectory": trajectory,
            "num_poses": num_poses,
            "trajectory_dim": trajectory_dim,
            "commitment_weight": commitment_weight,
            "vq_weight": vq_weight,
            "heading_weight": heading_weight,
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
        return super().load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            strict=strict,
            weights_only=weights_only,
            **kwargs,
        )

    @property
    def _input_dim(self) -> int:
        return self.num_poses * self.trajectory_dim

    @staticmethod
    def _heading_error(pred: Tensor, target: Tensor) -> Tensor:
        """Wrapped heading difference in radians, `(*batch, num_poses)`."""
        return torch.atan2(
            torch.sin(pred[..., 2] - target[..., 2]),
            torch.cos(pred[..., 2] - target[..., 2]),
        ).abs()

    def _recon_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Trajectory reconstruction loss with angle-aware heading term."""
        xy = F.l1_loss(pred[..., :2], target[..., :2], reduction="mean")
        heading = self._heading_error(pred, target).mean()
        return xy + self.heading_weight * heading

    def encode(self, trajectory: Tensor) -> Tensor:
        """Quantized latent `z_q` for a future trajectory `(*batch, P, 3)` -> `(*batch, dim)`."""
        *batch, num_poses, _trajectory_dim = trajectory.shape
        if num_poses != self.num_poses or _trajectory_dim != self.trajectory_dim:
            msg = (
                "trajectory shape mismatch: expected "
                f"(*batch, {self.num_poses}, {self.trajectory_dim}), got {trajectory.shape}"
            )
            raise ValueError(msg)

        flat = trajectory.reshape(-1, self._input_dim)
        _codes, z_q, _ = self.quantizer(self.encoder(flat))
        return z_q.reshape(*batch, z_q.shape[-1])

    def _gather_trajectory(self, batch: Any) -> Tensor:
        inputs = self.input_transform(batch)
        trajectory = reduce(lambda acc, key: acc[key], self.trajectory, inputs)
        return trajectory.reshape(-1, self._input_dim)

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        traj = self._gather_trajectory(batch)
        traj = traj.reshape(-1, self.num_poses, self.trajectory_dim)

        z = self.encoder(traj.reshape(-1, self._input_dim))
        codes, z_q, vq = self.quantizer(z)
        traj_hat = self.decoder(z + (z_q - z).detach())

        # decode back to the same flattened trajectory layout the target uses
        traj_hat = traj_hat.reshape(-1, self.num_poses, self.trajectory_dim)
        recon = self._recon_loss(traj_hat, traj)
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


@final
class TrajectoryLatentTokenizer(Module):
    """Encode a per-frame future trajectory into one residual-VQ latent token.

    Wraps a pretrained `TrajectoryTokenizer`. It maps trajectories
    `(*batch, num_poses, 3)` -> `(*batch, 1, dim)` so the policy can consume a
    single discrete trajectory token per frame the same way it consumes the
    waypoint latent token.
    """

    def __init__(self, *, tokenizer: TrajectoryTokenizer, **_legacy_kwargs: Any) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    @override
    def forward(self, trajectory: Tensor) -> Tensor:
        z_q = self.tokenizer.encode(trajectory)
        return z_q.unsqueeze(-2)
