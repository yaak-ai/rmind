"""Residual-VQ tokenizer for 100-step robot-native nero chunks (P3), per the playbook.

The chunk is one side's `(H = 100, A = 13)` robot-native action at 30 Hz (7 arm
joints + 6 Revo2 finger commands). 1300 values per chunk is far outside what the
team's tokenizers were validated on (D12: 0.43 bits/value, the car: 0.67; 64 bits
over 1300 values is 0.05), so the recipe follows the "Action tokenizer: a
debugging playbook" instead of the inherited VQ-BeT defaults:

* **Tokenize 10 Hz KEYFRAMES** (`k = 0, 3, ..., 99` -> 34 steps) and decode
  through a FIXED in-graph linear interpolation back to 30 Hz
  (`KeyframeInterpolation`). Loss and metrics are taken at 30 Hz, so the
  interpolation floor is part of what is measured; `nero_tokenizer_report`
  reports that floor on ground truth separately.
* **Dilated temporal conv encoder/decoder** (`ChunkConvEncoder/Decoder`, k=7,
  residual units at dilations 1/3/9, ELU, no norm) -- the playbook's step 4 win.
* **Rate by DEPTH, not width**: codebook 16, depth 16/24/32 (64/96/128 bits).
* **Loss**: `smooth_l1` with `beta` BELOW the typical error (0.2), plus
  inverse-frequency **event weighting** on the finger axes (weight 5 on elements
  away from the axis's atom). Finger commands are an open atom plus a closed
  plateau -- the D12 fork shape, whose channel plain L1 killed (L1's minimiser is
  the per-code median). `event_weight: null` disables it (NOT `{}`: OmegaConf
  merges a mapping). The atom is NOT estimated from a batch: it is the
  deterministic train-split `event_reference_<mode>.json` from `nero_fit_stats`
  (`EventReference`), pinned to this tokenizer's standardizer SHA256, and
  training refuses to start without it.
* lr 3e-4, wd 0.01, cosine with `lr_total_steps` = the real step count, vq
  weight 1 (configs).

The tokenizer OWNS the action standardizer of its relative mode (train-split,
per axis, nutron_standardizer JSON) and applies the relative transform itself, so
the policy cannot pair it with a mismatched one: the policy reads both from here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence  # noqa: TC003 (pydantic validate_call)
from typing import Any, Literal, Self, override

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import (  # noqa: TC002 (pydantic validate_call)
    _MAP_LOCATION_TYPE,
    _PATH,
)
from pydantic import ConfigDict, InstanceOf, validate_call
from pytorch_lightning.utilities.model_helpers import (
    _restricted_classmethod,  # noqa: PLC2701
)
from pytorch_lightning.utilities.types import (  # noqa: TC002
    STEP_OUTPUT,
    OptimizerLRScheduler,
)
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer  # noqa: TC002 (pydantic validate_call)

from rmind.components import optimizers
from rmind.components.nn import KeyframeInterpolation
from rmind.components.vq import (  # noqa: TC001 (pydantic validate_call)
    GroupedResidualVQ,
    ResidualVQ,
)
from rmind.config import HydraConfig, init_hydra_param
from rmind.data.nero_robot import (
    ARM_AXES,
    FINGER_AXES,
    NUM_AXES,
    RELATIVE_MODES,
    AxisStandardizer,
    EventReference,
    to_relative,
)
from rmind.models.action_tokenizer import (  # noqa: TC001 (pydantic validate_call)
    LRSchedulerHydraConfig,
)
from rmind.utils._wandb import LoadableFromArtifact

__all__ = ["NeroChunkTokenizer", "explained_variance", "total_variation"]

type Path = tuple[str, ...]


def explained_variance(
    pred: Tensor, target: Tensor, weight: Tensor | None = None
) -> Tensor:
    """Per-axis EV = 1 - MSE / Var over `(n, H, A)`, optionally `(n, H)`-weighted."""
    w = (
        torch.ones(target.shape[:-1], device=target.device)
        if weight is None
        else weight
    )
    w = w.unsqueeze(-1).to(target.dtype)
    total = w.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (target * w).sum(dim=(0, 1)) / total
    var = (((target - mean) ** 2) * w).sum(dim=(0, 1)) / total
    mse = (((pred - target) ** 2) * w).sum(dim=(0, 1)) / total
    return 1.0 - mse / var.clamp_min(1e-12)


def total_variation(x: Tensor, weight: Tensor | None = None) -> Tensor:
    """Per-axis mean |x[t+1] - x[t]| over `(n, H, A)` (pairs where both are real)."""
    d = (x[:, 1:] - x[:, :-1]).abs()
    if weight is None:
        return d.mean(dim=(0, 1))
    w = (weight[:, 1:] * weight[:, :-1]).unsqueeze(-1).to(x.dtype)
    return (d * w).sum(dim=(0, 1)) / w.sum(dim=(0, 1)).clamp_min(1.0)


class NeroChunkTokenizer(pl.LightningModule, LoadableFromArtifact):
    """RVQ autoencoder over one side's standardized `(H, A)` robot chunk."""

    event_reference: Tensor  # buffer, (A,)

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        encoder: HydraConfig[Module] | InstanceOf[Module],
        #: `GroupedResidualVQ` (with `AxisGroupChunkMLPEncoder/Decoder`) gives
        #: each axis group, e.g. arm and fingers, its own codes (opt-in)
        quantizer: HydraConfig[ResidualVQ]
        | HydraConfig[GroupedResidualVQ]
        | InstanceOf[ResidualVQ]
        | InstanceOf[GroupedResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        standardizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        action_horizon: int = 100,
        action_features: int = NUM_AXES,
        keyframe_stride: int = 3,
        relative_mode: Literal["none", "hand", "all"] = "none",
        loss: Literal["l1", "smooth_l1", "l2"] = "smooth_l1",
        smooth_l1_beta: float = 0.2,
        event_weight: float | None = 5.0,
        event_axes: Sequence[int] = FINGER_AXES,
        event_threshold: float = 0.5,
        #: `event_reference_<mode>.json` (nero_fit_stats). Required to TRAIN with
        #: `event_weight`; a checkpoint restores the buffer without it.
        event_reference: str | None = None,
        commitment_weight: float = 1.0,
        vq_weight: float = 1.0,
        chunk: Path = ("action.chunk",),
        state: Path = ("state",),
        side_valid: Path = ("side_valid",),
        is_pad: Path | None = ("action.is_pad",),
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
    ) -> None:
        super().__init__()
        if relative_mode not in RELATIVE_MODES:
            msg = f"relative_mode {relative_mode!r} not in {RELATIVE_MODES}"
            raise ValueError(msg)

        hparams: dict[str, Any] = {}
        self.encoder = init_hydra_param(hparams, "encoder", encoder)
        self.quantizer: ResidualVQ | GroupedResidualVQ = init_hydra_param(
            hparams, "quantizer", quantizer
        )
        self.decoder = init_hydra_param(hparams, "decoder", decoder)
        std = init_hydra_param(hparams, "standardizer", standardizer)
        self.standardizer: AxisStandardizer = AxisStandardizer() if std is None else std
        self.interpolate = KeyframeInterpolation(
            num_steps=action_horizon, num_axes=action_features, stride=keyframe_stride
        )
        self.action_horizon = action_horizon
        self.action_features = action_features
        self.keyframe_stride = keyframe_stride
        self.relative_mode = relative_mode
        self.loss = loss
        self.smooth_l1_beta = smooth_l1_beta
        self.event_weight = event_weight
        self.event_axes = tuple(event_axes)
        self.event_threshold = event_threshold
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        self.chunk_path, self.state_path = chunk, state
        self.side_valid_path, self.is_pad_path = side_valid, is_pad
        # the atom each event axis sits at, in standardized units: the train-split
        # EventReference file (or restored from a checkpoint). NaN = not set, and
        # training with event weighting refuses that (no first-batch fallback).
        self.register_buffer(
            "event_reference", torch.full((action_features,), float("nan"))
        )
        if event_reference is not None:
            self.set_event_reference(EventReference.load(event_reference))
        hparams |= {
            "action_horizon": action_horizon,
            "action_features": action_features,
            "keyframe_stride": keyframe_stride,
            "relative_mode": relative_mode,
            "loss": loss,
            "smooth_l1_beta": smooth_l1_beta,
            "event_weight": event_weight,
            "event_axes": self.event_axes,
            "event_threshold": event_threshold,
            "event_reference": event_reference,
            "commitment_weight": commitment_weight,
            "vq_weight": vq_weight,
            "chunk": chunk,
            "state": state,
            "side_valid": side_valid,
            "is_pad": is_pad,
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
        strict: bool | None = True,
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

    # ------------------------------------------------------------ geometry

    @property
    def num_keyframes(self) -> int:
        return self.interpolate.num_keyframes

    @property
    def latent_dim(self) -> int:
        return int(self.quantizer.dim)

    @property
    def bits(self) -> float:
        return self.quantizer.num_quantizers * math.log2(self.quantizer.codebook_size)

    @property
    def standardizer_digest(self) -> str:
        return self.standardizer.digest

    # --------------------------------------------------------------- codec

    def keyframes(self, chunk: Tensor) -> Tensor:
        """`(n, H, A)` -> flat `(n, K * A)` 10 Hz keyframes."""
        return chunk[:, :: self.keyframe_stride].flatten(1)

    def encode_latent(self, chunk: Tensor) -> Tensor:
        """STANDARDIZED `(n, H, A)` -> unquantized latent `(n, L)`."""
        return self.encoder(self.keyframes(chunk))

    def encode(self, chunk: Tensor) -> Tensor:
        """STANDARDIZED `(n, H, A)` -> codes `(n, num_quantizers)`."""
        codes, _, _ = self.quantizer(self.encode_latent(chunk))
        return codes

    def lookup(self, codes: Tensor) -> Tensor:
        """`(n, Q)` codes -> quantized latent `(n, L)`."""
        return self.quantizer.lookup(codes)

    def decode_latent(self, z: Tensor) -> Tensor:
        """Latent `(n, L)` -> STANDARDIZED `(n, H, A)` at 30 Hz (decoder + interpolation)."""
        dense = self.interpolate(self.decoder(z))
        return dense.reshape(*z.shape[:-1], self.action_horizon, self.action_features)

    def invert(self, codes: Tensor) -> Tensor:
        """`(*b, Q)` -> flat STANDARDIZED `(*b, H * A)` (the table-offset API)."""
        *batch, q = codes.shape
        return self.decode_latent(self.lookup(codes.reshape(-1, q))).reshape(
            *batch, self.action_horizon * self.action_features
        )

    @override
    def forward(self, chunk: Tensor) -> Tensor:
        """STANDARDIZED `(*b, H, A)` -> codes `(*b, Q)`."""
        *batch, h, a = chunk.shape
        return self.encode(chunk.reshape(-1, h, a)).reshape(*batch, -1)

    # --------------------------------------------------------------- data

    def prepare(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Batch -> STANDARDIZED per-side rows `(n, H, A)` and real-step mask `(n, H)`.

        `action.chunk (b, T, H, S, A)` is made relative to each FRAME's own state
        (`relative_mode`), standardized with this tokenizer's standardizer, and the
        valid `(batch, frame, side)` rows are kept.
        """

        def get(path: Path) -> Any:
            value = batch
            for key in path:
                value = value[key]
            return value

        chunk = get(self.chunk_path).float()
        b, t, h, s, a = chunk.shape
        if self.relative_mode != "none":
            chunk = to_relative(chunk, get(self.state_path).float(), self.relative_mode)
        chunk = self.standardizer(chunk)
        valid = get(self.side_valid_path).bool()  # (b, S)
        rows = valid[:, None, :].expand(b, t, s).reshape(-1)
        per_side = chunk.permute(0, 1, 3, 2, 4).reshape(-1, h, a)
        real = (
            ~get(self.is_pad_path).bool()
            if self.is_pad_path is not None
            else torch.ones(b, t, h, dtype=torch.bool, device=chunk.device)
        )
        real = real[:, :, None, :].expand(b, t, s, h).reshape(-1, h)
        return per_side[rows], real[rows]

    # --------------------------------------------------------------- loss

    def _elementwise(self, pred: Tensor, target: Tensor) -> Tensor:
        match self.loss:
            case "l1":
                return (pred - target).abs()
            case "smooth_l1":
                return F.smooth_l1_loss(
                    pred, target, beta=self.smooth_l1_beta, reduction="none"
                )
            case "l2":
                return (pred - target) ** 2

    @torch.no_grad()
    def set_event_reference(self, reference: EventReference) -> None:
        """Install a train-split `EventReference`.

        Raises:
            ValueError: if it was fitted for another relative mode or in the units
                of another action standardizer.
        """
        if reference.relative_mode != self.relative_mode:
            msg = (
                f"event reference is for relative_mode {reference.relative_mode!r}, "
                f"tokenizer is {self.relative_mode!r}"
            )
            raise ValueError(msg)
        if reference.standardizer_sha256 != self.standardizer_digest:
            msg = (
                f"event reference was fitted against action standardizer "
                f"{reference.standardizer_sha256}, tokenizer has "
                f"{self.standardizer_digest}: re-run nero_fit_stats"
            )
            raise ValueError(msg)
        self.event_reference.copy_(torch.tensor(reference.reference))

    def _require_event_reference(self) -> None:
        if self.event_weight is None or not self.event_axes:
            return
        if bool(torch.isnan(self.event_reference[list(self.event_axes)]).any()):
            msg = (
                "event weighting is on but no event reference is set: pass "
                "`event_reference: <stats>/event_reference_<mode>.json` (nero_fit_stats) "
                "or `event_weight: null`"
            )
            raise ValueError(msg)

    def weights(self, target: Tensor, real: Tensor) -> Tensor:
        """`(n, H, A)` loss weights: real steps only, events on `event_axes` boosted."""
        w = real.unsqueeze(-1).to(target.dtype).expand_as(target).clone()
        if self.event_weight is None or not self.event_axes:
            return w
        reference = self.event_reference.to(target.dtype)
        event = (target - reference).abs() > self.event_threshold
        boost = torch.ones(
            self.action_features, device=target.device, dtype=target.dtype
        )
        boost[list(self.event_axes)] = self.event_weight
        return w * torch.where(event, boost, torch.ones_like(boost))

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        target, real = self.prepare(batch)
        if target.numel() == 0:
            msg = "no valid sides in batch -- side_valid is all False"
            raise ValueError(msg)
        self._require_event_reference()

        z = self.encode_latent(target)
        codes, z_q, vq = self.quantizer(z)
        recon = self.decode_latent(z + (z_q - z).detach())  # straight-through

        w = self.weights(target, real)
        recon_loss = (self._elementwise(recon, target) * w).sum() / w.sum().clamp_min(
            1.0
        )
        total = recon_loss + self.vq_weight * (
            vq["codebook"] + self.commitment_weight * vq["commit"]
        )
        metrics: dict[str, Tensor] = {
            "recon": recon_loss,
            "codebook": vq["codebook"],
            "commit": vq["commit"],
            "total": total,
        }
        perplexity = self.quantizer.perplexity(codes)
        for q in range(self.quantizer.num_quantizers):
            metrics[f"perplexity/q{q}"] = perplexity[q]
        with torch.no_grad():
            ev = explained_variance(recon, target, real)
            metrics["ev/mean"] = ev.mean()
            metrics["ev/arm"] = ev[list(ARM_AXES)].mean()
            metrics["ev/finger"] = ev[list(FINGER_AXES)].mean()
            ratio = recon.std(dim=(0, 1)) / target.std(dim=(0, 1)).clamp_min(1e-8)
            metrics["dead_channels"] = (ratio < 0.1).sum().float()  # noqa: PLR2004
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
