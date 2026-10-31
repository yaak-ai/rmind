from functools import reduce
from typing import Any, Literal, override

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pydantic import InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from rmind.components import optimizers
from rmind.components.vq import ResidualVQ
from rmind.config import HydraConfig, init_hydra_param
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.utils._wandb import LoadableFromArtifact

type Path = tuple[str, ...]

#: `(x, y, theta)` per pose -- `rmind.components.nn.TrajectoryTarget`'s layout.
POSE_DIM = 3


def _wrap(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class TrajectoryTokenizer(pl.LightningModule, LoadableFromArtifact):
    """Residual-VQ autoencoder over a dead-reckoned future ego trajectory, the
    VQ-BeT (https://arxiv.org/pdf/2403.03181) tokenizer for a trajectory head.

    Input is `rmind.components.nn.TrajectoryTarget`'s output, `(*batch,
    num_poses, 3)` anchor-frame `(x, y, theta)` (meters, radians). The encoder
    sees one of two `representation`s of it, standardized per feature:

    - `"absolute"`: the poses as is.
    - `"delta"`: per-step increments `p_k - p_{k-1}` (with `p_0` the anchor
      origin), still in the anchor frame so a yaw rotation of the target
      (`rmind.components.augment.YawShiftCrop`) rotates them too; the
      `theta` increments are wrapped to `[-pi, pi]`. Increments have a
      roughly horizon-independent scale, where absolute poses spread out
      linearly with `k`.

    Either way the decoder output is mapped back to absolute poses (a cumsum
    for `"delta"`) and the reconstruction loss is taken THERE -- otherwise
    per-step errors that drift under the cumsum would go unpenalized.
    `delta_weight` optionally adds an L1 on the standardized increments.

    The standardization stats are accumulated over the first
    `norm_warmup_steps` training batches and frozen afterwards; they are
    persistent buffers, so a loaded checkpoint carries them.
    """

    norm_count: Tensor
    norm_sum: Tensor
    norm_sum_sq: Tensor

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        input_transform: HydraConfig[Module] | InstanceOf[Module],
        encoder: HydraConfig[Module] | InstanceOf[Module],
        quantizer: HydraConfig[ResidualVQ] | InstanceOf[ResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        trajectory: Path = ("context", "trajectory_target"),
        num_poses: int = 5,
        representation: Literal["absolute", "delta"] = "absolute",
        heading_weight: float = 10.0,
        delta_weight: float = 0.0,
        norm_warmup_steps: int = 100,
        commitment_weight: float = 1.0,
        vq_weight: float = 5.0,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
    ) -> None:
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
        self.representation: Literal["absolute", "delta"] = representation
        self.heading_weight = heading_weight
        self.delta_weight = delta_weight
        self.norm_warmup_steps = norm_warmup_steps
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        hparams |= {
            "trajectory": trajectory,
            "num_poses": num_poses,
            "representation": representation,
            "heading_weight": heading_weight,
            "delta_weight": delta_weight,
            "norm_warmup_steps": norm_warmup_steps,
            "commitment_weight": commitment_weight,
            "vq_weight": vq_weight,
        }

        # float64 running sums: thousands of batches of 2k samples in float32
        # would lose the variance to cancellation.
        self.register_buffer("norm_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "norm_sum", torch.zeros(self._input_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "norm_sum_sq", torch.zeros(self._input_dim, dtype=torch.float64)
        )

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.save_hyperparameters(hparams)

    @property
    def _input_dim(self) -> int:
        return self.num_poses * POSE_DIM

    # --- representation ---------------------------------------------------

    def to_representation(self, poses: Tensor) -> Tensor:
        """`(*b, P, 3)` absolute poses -> `(*b, P * 3)` encoder-space vector
        (before standardization)."""
        if self.representation == "delta":
            prev = torch.cat(
                [torch.zeros_like(poses[..., :1, :]), poses[..., :-1, :]], -2
            )
            delta = poses - prev
            poses = torch.cat([delta[..., :2], _wrap(delta[..., 2:])], dim=-1)
        return poses.flatten(-2, -1)

    def from_representation(self, x: Tensor) -> Tensor:
        """Inverse of `to_representation`: `(*b, P * 3)` -> `(*b, P, 3)`
        absolute poses, `theta` wrapped to `[-pi, pi]`."""
        poses = x.unflatten(-1, (self.num_poses, POSE_DIM))
        if self.representation == "delta":
            poses = poses.cumsum(dim=-2)
        return torch.cat([poses[..., :2], _wrap(poses[..., 2:])], dim=-1)

    # --- standardization --------------------------------------------------

    @property
    def _norm_stats(self) -> tuple[Tensor, Tensor]:
        count = self.norm_count.clamp_min(1.0)
        mean = self.norm_sum / count
        var = (self.norm_sum_sq / count - mean.square()).clamp_min(0.0)
        # before any stats exist (count == 0) this is the identity transform
        std = torch.where(self.norm_count > 0, var.sqrt().clamp_min(1e-6), 1.0)
        return mean.float(), std.float()

    def _normalize(self, x: Tensor) -> Tensor:
        mean, std = self._norm_stats
        return (x - mean) / std

    def _denormalize(self, x: Tensor) -> Tensor:
        mean, std = self._norm_stats
        return x * std + mean

    @torch.no_grad()
    def _update_norm_stats(self, x: Tensor) -> None:
        x = x.double()
        stats = torch.cat([x.new_tensor([x.shape[0]]), x.sum(0), x.square().sum(0)])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
        self.norm_count += stats[0]
        self.norm_sum += stats[1 : 1 + self._input_dim]
        self.norm_sum_sq += stats[1 + self._input_dim :]

    # --- public API (used by a downstream VQ-BeT head) ----------------------

    @override
    def forward(self, poses: Tensor) -> Tensor:
        """`(*batch, P, 3)` poses -> `(*batch, num_quantizers)` codes."""
        *batch, _, _ = poses.shape
        x = self._normalize(self.to_representation(poses)).reshape(-1, self._input_dim)
        codes, _, _ = self.quantizer(self.encoder(x))
        return codes.reshape(*batch, self.quantizer.num_quantizers)

    def decode(self, z_q: Tensor) -> Tensor:
        """Latent `(*batch, dim)` (sum of codebook vectors, optionally plus a
        predicted offset) -> `(*batch, P, 3)` absolute poses."""
        return self.from_representation(self._denormalize(self.decoder(z_q)))

    @torch.no_grad()
    def reconstruct(self, poses: Tensor) -> Tensor:
        """`(*batch, P, 3)` poses -> encode, quantize, decode -> `(*batch, P, 3)` poses."""
        *batch, _, _ = poses.shape
        x = self._normalize(self.to_representation(poses)).reshape(-1, self._input_dim)
        _, z_q, _ = self.quantizer(self.encoder(x))
        return self.decode(z_q).reshape(*batch, self.num_poses, POSE_DIM)

    def invert(self, codes: Tensor) -> Tensor:
        """`(*batch, num_quantizers)` codes -> `(*batch, P, 3)` absolute poses."""
        *batch, num_quantizers = codes.shape
        z_q = self.quantizer.lookup(codes.reshape(-1, num_quantizers))
        return self.decode(z_q).reshape(*batch, self.num_poses, POSE_DIM)

    # --- training ---------------------------------------------------------

    def _gather_poses(self, batch: Any) -> Tensor:
        inputs = self.input_transform(batch)
        poses = reduce(lambda acc, key: acc[key], self.trajectory, inputs)
        return poses.reshape(-1, self.num_poses, POSE_DIM).float()

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:  # noqa: PLR0914
        poses = self._gather_poses(batch)
        x_raw = self.to_representation(poses)

        if self.training and self.global_step < self.norm_warmup_steps:
            self._update_norm_stats(x_raw)

        x = self._normalize(x_raw)
        z = self.encoder(x)
        codes, z_q, vq = self.quantizer(z)
        x_hat = self.decoder(z + (z_q - z).detach())  # straight-through
        poses_hat = self.from_representation(self._denormalize(x_hat))

        xy_err = (poses_hat[..., :2] - poses[..., :2]).norm(dim=-1)  # (n, P) meters
        heading_err = _wrap(poses_hat[..., 2] - poses[..., 2]).abs()  # (n, P) radians

        recon_xy = (poses_hat[..., :2] - poses[..., :2]).abs().mean()
        recon_heading = heading_err.mean()
        recon = recon_xy + self.heading_weight * recon_heading
        if self.delta_weight > 0:
            recon += self.delta_weight * (x_hat - x).abs().mean()

        total = recon + self.vq_weight * (
            vq["codebook"] + self.commitment_weight * vq["commit"]
        )

        metrics = {
            "recon": recon,
            "recon_xy": recon_xy,
            "recon_heading": recon_heading,
            "ade": xy_err.mean(),
            "fde": xy_err[..., -1].mean(),
            "heading_err_deg": torch.rad2deg(heading_err).mean(),
            "final_heading_err_deg": torch.rad2deg(heading_err[..., -1]).mean(),
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
