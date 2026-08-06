"""VQ-BeT residual-VQ tokenizer for the nero-arms SE(3) action space.

Same recipe as `rmind.models.waypoints_tokenizer.WaypointsTokenizer` and
`rmind.models.action_tokenizer.ActionTokenizer` (encoder -> `ResidualVQ` ->
decoder, straight-through, codebook + commitment losses), adapted to the
contract's action space:

* the token is **one side's action chunk**: `(action_horizon, action_features)`
  in the contract's 9D form (default `6 x 60`). Because `action(t)` is *future
  state* (§6.2), state and action live in the same space -- so **this one
  tokenizer serves both** current-state encoding and action prediction;
* the tokenizer is **per side, weight-shared**. A right-only dataset (the dummy)
  therefore trains a tokenizer that is immediately valid for the left hand, and
  the `side_valid` mask simply drops invalid rows from the fit rather than
  teaching a codebook that "left == zeros";
* inputs are **per-channel standardised** with train-split statistics
  (`PoseStandardizer`, §5.4). Translations are ~0.06-0.7 m while 6D rotation
  components are ~1; unstandardised, rotation error swamps translation error in
  any L1/L2 objective;
* reconstruction error is reported **separately for translation (mm) and
  rotation (degrees, geodesic)** (§5.5). A single scalar recon loss is the single
  most likely silent failure in this task.

Configuration seam (contract §11)
---------------------------------
`action_features` and `action_horizon` are constructor arguments, not constants.
Swapping to the BrainCo Revo2 joint-angle action space of §11 option (B) --
~12 dims per side instead of 60 -- is `action_features: 12` plus a new
checkpoint; no code change. When `action_features != 60` the pose-layout
physical metrics are skipped automatically (there is no translation/rotation
split in a joint-angle vector) and only the standardised recon loss is reported.
"""

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
from rmind.data.nero import SIDE_DIM, PoseStandardizer, pose_error_metrics
from rmind.models.action_tokenizer import LRSchedulerHydraConfig
from rmind.utils._wandb import LoadableFromArtifact

__all__ = ["NeroPoseTokenizer"]


class NeroPoseTokenizer(pl.LightningModule, LoadableFromArtifact):
    """Residual-VQ autoencoder over one side's `(horizon, features)` pose chunk."""

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        encoder: HydraConfig[Module] | InstanceOf[Module],
        quantizer: HydraConfig[ResidualVQ] | InstanceOf[ResidualVQ],
        decoder: HydraConfig[Module] | InstanceOf[Module],
        standardizer: HydraConfig[Module] | InstanceOf[Module] | None = None,
        action_features: int = SIDE_DIM,
        action_horizon: int = 6,
        commitment_weight: float = 1.0,
        vq_weight: float = 5.0,
        action: tuple[str, ...] = ("action", "future_state"),
        side_valid: tuple[str, ...] = ("side_valid",),
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        **_legacy_hparams: Any,
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        self.encoder = init_hydra_param(hparams, "encoder", encoder)
        self.quantizer: ResidualVQ = init_hydra_param(hparams, "quantizer", quantizer)
        self.decoder = init_hydra_param(hparams, "decoder", decoder)
        standardizer_module = init_hydra_param(hparams, "standardizer", standardizer)
        self.standardizer: Module = (
            PoseStandardizer(dim=action_features)
            if standardizer_module is None
            else standardizer_module
        )

        self.action_features = action_features
        self.action_horizon = action_horizon
        self.commitment_weight = commitment_weight
        self.vq_weight = vq_weight
        self.action = action
        self.side_valid = side_valid
        hparams |= {
            "action_features": action_features,
            "action_horizon": action_horizon,
            "commitment_weight": commitment_weight,
            "vq_weight": vq_weight,
            "action": action,
            "side_valid": side_valid,
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

    # ------------------------------------------------------------- geometry

    @property
    def _action_features(self) -> int:
        """Name kept for parity with `ActionTokenizer` (the policy reads it)."""
        return self.action_features

    @property
    def action_dim(self) -> int:
        return self.action_horizon * self.action_features

    @property
    def has_pose_layout(self) -> bool:
        """Whether the per-side vector is the contract §6.1 60-dim pose layout.

        False for the §11 option-(B) Revo2 joint-angle variant, where a
        translation/rotation split is meaningless.
        """
        return self.action_features == SIDE_DIM

    def _normalize(self, action: Tensor) -> Tensor:
        """Standardise a FLAT `(*b, horizon * features)` chunk. Mirrors `ActionTokenizer`."""
        *batch, _ = action.shape
        chunk = action.reshape(*batch, self.action_horizon, self.action_features)
        return self.standardizer(chunk).reshape(*batch, self.action_dim)

    def _denormalize(self, action: Tensor) -> Tensor:
        *batch, _ = action.shape
        chunk = action.reshape(*batch, self.action_horizon, self.action_features)
        return self.standardizer.unstandardize(chunk).reshape(  # ty:ignore[call-non-callable]
            *batch, self.action_dim
        )

    # ---------------------------------------------------------------- codec

    @override
    def forward(self, action: Tensor) -> Tensor:
        """`(*b, horizon, features)` raw chunk -> `(*b, num_quantizers)` codes."""
        action = action.flatten(-2, -1)
        *batch, action_dim = action.shape
        z = self.encoder(self._normalize(action).reshape(-1, action_dim))
        codes, _, _ = self.quantizer(z)
        return codes.reshape(*batch, self.quantizer.num_quantizers)

    def invert(self, codes: Tensor) -> Tensor:
        """`(*b, num_quantizers)` -> flat STANDARDISED `(*b, horizon * features)`."""
        *batch, num_quantizers = codes.shape
        z_q = self.quantizer.lookup(codes.reshape(-1, num_quantizers))
        return self.decoder(z_q).reshape(*batch, self.action_dim)

    # ---------------------------------------------------------------- train

    def _gather(self, batch: Any) -> tuple[Tensor, Tensor]:
        """`(n, horizon, features)` chunks and their `(n,)` validity mask.

        Contract §8 emits `action.future_state` as `(b, T, H, 2, 60)` and
        `side_valid` as `(b, 2)`. Every (batch, frame, side) triple is one
        training example; invalid sides are dropped entirely rather than fed as
        zeros -- see the class docstring.
        """
        action = batch
        for key in self.action:
            action = action[key]
        valid = batch
        for key in self.side_valid:
            valid = valid[key]

        b, t, h, s, f = action.shape
        chunks = action.permute(0, 1, 3, 2, 4).reshape(b * t * s, h, f)
        mask = valid[:, None, :].expand(b, t, s).reshape(-1)
        return chunks, mask

    def _step(self, batch: Any) -> tuple[Tensor, dict[str, Tensor]]:
        chunks, mask = self._gather(batch)
        chunks = chunks[mask]
        if chunks.numel() == 0:
            msg = "no valid sides in batch -- side_valid is all False"
            raise ValueError(msg)

        raw = chunks.flatten(-2, -1)  # (n, horizon * features)
        a = self._normalize(raw)

        z = self.encoder(a)
        codes, z_q, vq = self.quantizer(z)
        a_hat = self.decoder(z + (z_q - z).detach())  # straight-through

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

        # ⚠️ contract §5.5: NEVER report a single scalar recon for this space.
        with torch.no_grad():
            metrics |= self.reconstruction_metrics(a_hat, raw)

        return total, metrics

    def reconstruction_metrics(
        self, predicted: Tensor, target_raw: Tensor
    ) -> dict[str, Tensor]:
        """Split reconstruction error, in physical units where the layout allows it.

        Args:
            predicted: STANDARDISED flat reconstruction `(n, horizon * features)`.
            target_raw: RAW (unstandardised) flat target, same shape.
        """
        target = self._normalize(target_raw)
        out = {
            "recon_std_l1": F.l1_loss(predicted, target),
            "recon_std_l2": F.mse_loss(predicted, target).sqrt(),
        }
        if not self.has_pose_layout:
            return out

        n = predicted.shape[0]
        shape = (n, self.action_horizon, self.action_features)
        pred_raw = self._denormalize(predicted).reshape(shape)
        gt = target_raw.reshape(shape)
        out |= pose_error_metrics(pred_raw, gt)

        # the same split in standardised units, so the training objective itself
        # can be audited channel-group-wise
        from rmind.data.nero import translation_rotation_split  # noqa: PLC0415

        p_t, p_r = translation_rotation_split(predicted.reshape(shape))
        g_t, g_r = translation_rotation_split(target.reshape(shape))
        out |= {
            "recon_std_l1/translation": F.l1_loss(p_t, g_t),
            "recon_std_l1/rotation": F.l1_loss(p_r, g_r),
        }
        return out

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
class NeroGoalXYZTokenizer(NeroPoseTokenizer):
    """Optional second RVQ over the normalised goal xyz 3-vector (contract §9).

    Same recipe, defaulted to a single 3-vector instead of a pose chunk, so
    `has_pose_layout` is False and only the standardised metrics are reported.
    This is the config seam for "predict the goal rather than merely condition on
    it" (§9). Scaffolding: NOT trained or evaluated in this work.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**{
            "action_horizon": 1,
            "action_features": 3,
            "action": ("goal", "xyz"),
            **kwargs,
        })
