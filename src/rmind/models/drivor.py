import math
from collections.abc import Iterator
from typing import Any, Literal, cast, override

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from rmind.components.drivor.loss import (
    winner_takes_all_pose_l1,
    winner_takes_all_pose_l1_components,
)
from rmind.components.drivor.trajectory_target import (
    dead_reckon_future_trajectory,
    gnss_anchor_drift_m,
)
from rmind.config import HydraConfig
from rmind.models.control_transformer import LRSchedulerHydraConfig
from rmind.utils._wandb import LoadableFromArtifact

CONTINUOUS_FIELDS = (
    "speed",
    "gas_pedal_normalized",
    "brake_pedal_normalized",
    "steering_angle_normalized",
)


class DrivoR(pl.LightningModule, LoadableFromArtifact):
    """Trajectory-only, single-camera adaptation of DrivoR (arXiv:2601.05083)
    on rmind/Yaak data.

    Bypasses `EpisodeBuilder`/tokenizer machinery entirely -- unlike
    `rmind.models.control_transformer.ControlTransformer`, this model reads a
    handful of raw scalar/tensor fields directly off the batch dict, at a
    single reference timestep (`reference_timestep`, default the FIRST step
    of the window: `0`).

    Ground truth trajectory: NOT `waypoints/xy_normalized` -- that field is a
    reference *route* (matched to the drive, but not equal to the ego's
    realized future path; see `dataset/yaak/*.yaml`'s `ST_Contains` sanity
    filter and the plan history for how this was established). Instead, the
    target is dead-reckoned forward from `reference_timestep` using CAN speed
    (a genuinely dense ~50Hz signal, smoothly interpolated onto every window
    step) and EKF/RTS-denoised heading (`dead_reckon_future_trajectory`).
    Verified against real drive data: heading is noise-*reduced*, not
    upsampled -- it's still sourced from the ~1Hz GNSS stream (one denoised
    value per raw GNSS fix), so it repeats across 2-3 consecutive window
    steps the same way raw GNSS position does; only speed is genuinely dense.
    The resulting trajectory is still a meaningfully better target than raw
    GNSS position directly (no per-fix GPS jitter, real per-step distance
    from dense speed), but isn't as smooth as originally assumed -- see
    `gnss_anchor_drift_m` and the plan's verification notes for how this was
    checked. `waypoints/xy_normalized` IS still used, but only as the model's
    driving-command substitute (see `EgoStateEncoder`/`route_tokenizer`
    below), since rmind has no NAVSIM-style discrete routing command.
    """

    @validate_call
    def __init__(  # noqa: C901, PLR0913
        self,
        *,
        image_preprocess: HydraConfig[Module] | InstanceOf[Module],
        backbone: HydraConfig[Module] | InstanceOf[Module],
        register_projection: HydraConfig[Module] | InstanceOf[Module],
        route_tokenizer: HydraConfig[Module] | InstanceOf[Module],
        trajectory_head: HydraConfig[Module] | InstanceOf[Module],
        loss: HydraConfig[Module] | InstanceOf[Module],
        score_head: HydraConfig[Module] | InstanceOf[Module] | None = None,
        score_tau: float = 0.02,
        score_tau_relative: bool = False,
        score_weight: float = 1.0,
        freeze_base: bool = False,
        optimizer: HydraConfig[Optimizer] | None = None,
        lr_scheduler: LRSchedulerHydraConfig | None = None,
        reference_timestep: int = 0,
    ) -> None:
        super().__init__()

        hparams: dict[str, Any] = {}

        if isinstance(image_preprocess, HydraConfig):
            hparams["image_preprocess"] = image_preprocess.model_dump()
            image_preprocess = image_preprocess.instantiate()
        self.image_preprocess = image_preprocess

        if isinstance(backbone, HydraConfig):
            hparams["backbone"] = backbone.model_dump()
            backbone = backbone.instantiate()
        self.backbone = backbone

        if isinstance(register_projection, HydraConfig):
            hparams["register_projection"] = register_projection.model_dump()
            register_projection = register_projection.instantiate()
        self.register_projection = register_projection

        if isinstance(route_tokenizer, HydraConfig):
            hparams["route_tokenizer"] = route_tokenizer.model_dump()
            route_tokenizer = route_tokenizer.instantiate()
        route_tokenizer.requires_grad_(False)  # noqa: FBT003
        self.route_tokenizer = route_tokenizer

        if isinstance(trajectory_head, HydraConfig):
            hparams["trajectory_head"] = trajectory_head.model_dump()
            trajectory_head = trajectory_head.instantiate()
        self.trajectory_head = trajectory_head

        if isinstance(loss, HydraConfig):
            hparams["loss"] = loss.model_dump()
            loss = loss.instantiate()
        self.loss = loss

        if isinstance(score_head, HydraConfig):
            hparams["score_head"] = score_head.model_dump()
            score_head = score_head.instantiate()
        self.score_head: Module | None = score_head

        self.score_tau = score_tau
        self.score_tau_relative = score_tau_relative
        self.score_weight = score_weight
        self.freeze_base = freeze_base
        hparams |= {
            "score_tau": score_tau,
            "score_tau_relative": score_tau_relative,
            "score_weight": score_weight,
            "freeze_base": freeze_base,
        }

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.reference_timestep = reference_timestep
        hparams["reference_timestep"] = reference_timestep

        self.save_hyperparameters(hparams)

        if freeze_base:
            for module in self._base_modules():
                module.requires_grad_(False).eval()  # noqa: FBT003

    def _base_modules(self) -> Iterator[Module]:
        """Every child `freeze_base` freezes: everything except `score_head`.
        Iterates `named_children()` rather than an allow-list so a submodule added
        later is frozen by default instead of silently trained (`loss` -- a Module,
        see `__init__` -- is easy to miss in a hand-written list).
        `route_tokenizer` (already frozen in `__init__`) and `backbone` (whose own
        `train()` force-evals the pretrained ViT) are idempotent under this.
        """
        return (m for name, m in self.named_children() if name != "score_head")

    @override
    def train(self, mode: bool = True) -> "DrivoR":
        super().train(mode)
        self.route_tokenizer.eval()  # frozen, see __init__
        if self.freeze_base:
            for module in self._base_modules():
                module.eval()
        return self

    def _inputs(
        self, batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        t0 = self.reference_timestep
        data = batch["data"]

        image = data["cam_front_left"][:, t0]
        continuous = torch.stack(
            [data[f"meta/VehicleMotion/{k}"][:, t0] for k in CONTINUOUS_FIELDS], dim=-1
        )
        turn_signal = data["meta/VehicleState/turn_signal"][:, t0].long()
        # driving-command substitute -- NOT the trajectory target, see class docstring
        route = data["waypoints/xy_normalized"][:, t0]

        # raw int64 microseconds-since-epoch (polars Datetime[us] -> torch int64). Epoch
        # values (~1.67e15) lose all sub-second precision if cast to float32 (~7 significant
        # digits) BEFORE differencing -- subtract the reference timestamp while still in
        # exact int64 arithmetic first, so the values handed to float32 are small (a few
        # seconds' worth of microseconds) and safe.
        time_stamp_us = data["meta/ImageMetadata.cam_front_left/time_stamp"]
        time_stamp_s = (time_stamp_us - time_stamp_us[:, t0 : t0 + 1]).float() / 1e6

        target_xy, target_heading = dead_reckon_future_trajectory(
            speed_kmh=data["meta/VehicleMotion/speed"],
            heading_deg=data["headings_denoised/heading"],
            time_stamp_s=time_stamp_s,
            reference_index=t0,
        )

        return image, continuous, turn_signal, route, target_xy, target_heading

    def _loss_hparams(self) -> tuple[float, Literal["mean", "sum"]]:
        """`self.loss` is typed generically (`InstanceOf[Module]`, see
        `__init__`) since any Hydra-configured module could be plugged in --
        but `winner_takes_all_pose_l1_components` (used below to get logging
        breakdowns `self.loss(...)` alone doesn't expose) needs its own
        `heading_weight`/`reduction` to compute a `loss` that actually matches
        `self.loss(...)`'s. Read them off `self.loss` when present (i.e. it's
        a `WinnerTakesAllPoseLoss`), else fall back to the same defaults the
        loss functions themselves use.
        """
        heading_weight = getattr(self.loss, "heading_weight", 0.1)
        reduction = cast(
            "Literal['mean', 'sum']", getattr(self.loss, "reduction", "mean")
        )
        return heading_weight, reduction

    def _pose_metrics(
        self,
        pred: torch.Tensor,
        target_xy: torch.Tensor,
        target_heading: torch.Tensor,
        *,
        prefix: Literal["train", "val"],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]
    ]:
        """Shared by `training_step`/`validation_step`: the scalar loss plus
        the logging breakdown (`winner_takes_all_pose_l1_components` gives us
        both from one call, so both steps see identical diagnostics instead
        of just `train/loss` with everything else gated on validation).

        Returns `(loss, best_index, per_candidate, metrics)` -- `best_index`
        and `per_candidate` are also needed by the score head (see
        `_score_metrics`), which distills the latter's ranking.
        """
        heading_weight, reduction = self._loss_hparams()
        loss, best_index, per_candidate, winner_xy_loss, winner_heading_loss = (
            winner_takes_all_pose_l1_components(
                pred,
                target_xy,
                target_heading,
                heading_weight=heading_weight,
                reduction=reduction,
            )
        )
        best_index_unique_frac: float = best_index.unique().numel() / best_index.numel()
        metrics: dict[str, torch.Tensor | float] = {
            f"{prefix}/loss": loss,
            # unweighted, unlike f"{prefix}/loss" -- see winner_takes_all_pose_l1_components
            f"{prefix}/loss_xy": winner_xy_loss.mean(),
            f"{prefix}/loss_heading": winner_heading_loss.mean(),
            # winner-collapse/candidate-diversity diagnostics
            f"{prefix}/best_index_unique_frac": best_index_unique_frac,
            f"{prefix}/per_candidate_loss_std": per_candidate.std(dim=-1).mean(),
        }
        return loss, best_index, per_candidate, metrics

    def _predict_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Unnormalized per-candidate scores, `(b, num_queries)`. Trained to distill
        the winner-takes-all oracle's per-candidate loss ranking (see
        `_score_metrics`) so a trajectory can be picked with no ground truth at
        deployment. Only call when `self.score_head is not None`.
        """
        return self.score_head(features).squeeze(-1)  # ty:ignore[call-non-callable]

    def _score_metrics(
        self,
        scores: torch.Tensor,  # (b, Q), raw, requires_grad
        per_candidate: torch.Tensor,  # (b, Q), from _pose_metrics
        *,
        prefix: Literal["train", "val"],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        eps = torch.finfo(torch.float32).tiny
        # DETACHED and float32: see the gradient + numerics notes below.
        pc = per_candidate.detach().float()
        tau = (
            self.score_tau * pc.std(dim=-1, keepdim=True).clamp_min(1e-8)
            if self.score_tau_relative
            else self.score_tau
        )
        # min-shifted: mathematically identical to softmax(-pc / tau), far better
        # conditioned under bf16-mixed.
        target = torch.softmax((pc.min(dim=-1, keepdim=True).values - pc) / tau, dim=-1)
        log_p = torch.log_softmax(scores.float(), dim=-1)
        loss = -(target * log_p).sum(dim=-1).mean()

        with torch.no_grad():
            entropy = -(target * (target + eps).log()).sum(dim=-1)  # (b,) nats
            oracle = pc.argmin(dim=-1)  # (b,)
            picked = scores.argmax(dim=-1)  # (b,)
            top5 = pc.topk(5, dim=-1, largest=False).indices  # (b, 5)
            picked_cost = pc.gather(-1, picked[:, None]).squeeze(-1)  # (b,)
            best_cost = pc.min(dim=-1).values  # (b,)
            regret = picked_cost - best_cost  # (b,) >= 0
            chance_regret = pc.mean(dim=-1) - best_cost  # (b,) random-pick baseline
            metrics: dict[str, torch.Tensor | float] = {
                f"{prefix}/score_loss": loss,
                # 0 => one-hot target (tau too small); 1 => uniform (tau too large)
                f"{prefix}/score_target_entropy_norm": entropy.mean()
                / math.log(scores.shape[-1]),
                # effective number of candidates the target spreads over, 1..Q; aim ~5-15
                f"{prefix}/score_target_perplexity": entropy.exp().mean(),
                f"{prefix}/score_top1_agreement": (picked == oracle).float().mean(),
                f"{prefix}/score_top5_agreement": (top5 == picked[:, None])
                .any(dim=-1)
                .float()
                .mean(),
                f"{prefix}/score_regret": regret.mean(),
                f"{prefix}/score_regret_chance": chance_regret.mean(),
                # THE number to watch: 0 = oracle, 1 = no better than random
                f"{prefix}/score_regret_ratio": regret.sum()
                / chance_regret.sum().clamp_min(1e-12),
                # scores collapsing to a constant => head learned nothing
                f"{prefix}/score_std": scores.detach().float().std(dim=-1).mean(),
            }
        return loss, metrics

    def _forward(
        self,
        *,
        image: torch.Tensor,
        continuous: torch.Tensor,
        turn_signal: torch.Tensor,
        route: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        registers = self.backbone(self.image_preprocess(image))
        context = self.register_projection(registers)

        with torch.no_grad():
            route_embedding = self.route_tokenizer(route).squeeze(-2)

        # `self.trajectory_head` is typed generically (`InstanceOf[Module]`, see
        # __init__) since any Hydra-configured module could be plugged in --
        # `forward_features` is specific to `TrajectoryDecoderHead`, not the
        # generic `Module` interface ty checks against here.
        pred, features = self.trajectory_head.forward_features(  # ty:ignore[call-non-callable]
            context=context,
            ego_continuous=continuous,
            ego_turn_signal=turn_signal,
            ego_route_embedding=route_embedding,
        )
        return pred, features  # (b, Q, P, 3), (b, Q, dim_model)

    @override
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        image, continuous, turn_signal, route, *_ = self._inputs(batch)
        pred, _ = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        return pred

    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred, features = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        loss, _, per_candidate, metrics = self._pose_metrics(
            pred, target_xy, target_heading, prefix="train"
        )
        total = loss
        if self.score_head is not None:
            score_loss, score_metrics = self._score_metrics(
                self._predict_scores(features), per_candidate, prefix="train"
            )
            total = loss + self.score_weight * score_loss
            metrics |= score_metrics | {"train/loss_total": total}
        self.log_dict(metrics, sync_dist=True)
        return {"loss": total}

    @override
    def validation_step(  # noqa: PLR0914
        self, batch: dict[str, Any], _batch_idx: int
    ) -> STEP_OUTPUT:
        data = batch["data"]
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred, features = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        loss, _, per_candidate, metrics = self._pose_metrics(
            pred, target_xy, target_heading, prefix="val"
        )
        total = loss
        if self.score_head is not None:
            score_loss, score_metrics = self._score_metrics(
                self._predict_scores(features), per_candidate, prefix="val"
            )
            total = loss + self.score_weight * score_loss
            metrics |= score_metrics | {"val/loss_total": total}

        if not self.trainer.sanity_checking:
            # QA check on the dead-reckoned target itself (see
            # trajectory_target.gnss_anchor_drift_m / class docstring) --
            # large drift flags wheel slip, GPS multipath, or a heading-filter
            # failure for that batch, independent of model quality.
            drift_m = gnss_anchor_drift_m(
                dead_reckoned_position_normalized=target_xy,
                gnss_xy=data["meta/Gnss/xy"],
                heading_deg=data["headings_denoised/heading"],
                reference_index=self.reference_timestep,
            )
            self.log_dict(
                metrics
                | {
                    "val/gnss_anchor_drift_m_median": drift_m.median(),
                    "val/gnss_anchor_drift_m_p90": drift_m.quantile(0.9),
                },
                sync_dist=True,
            )

        return {"loss": total}

    @override
    def predict_step(  # noqa: PLR0914
        self, batch: dict[str, Any], batch_idx: int = 0
    ) -> TensorDict:
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred, features = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        heading_weight, reduction = self._loss_hparams()
        _, best_index, per_candidate = winner_takes_all_pose_l1(
            pred,
            target_xy,
            target_heading,
            heading_weight=heading_weight,
            reduction=reduction,
        )
        best_pred = pred.gather(
            1, best_index[:, None, None, None].expand(-1, 1, *pred.shape[-2:])
        ).squeeze(1)

        trajectory = {
            "prediction": pred,
            "best_prediction": best_pred,
            "best_index": best_index,
            "per_candidate_loss": per_candidate,
            "ground_truth_xy": target_xy,
            "ground_truth_heading": target_heading,
        }
        if self.score_head is not None:
            scores = self._predict_scores(features)  # (b, Q)
            predicted_index = scores.argmax(dim=-1)  # (b,)
            trajectory |= {
                "scores": scores,
                "predicted_index": predicted_index,
                "selected_prediction": pred.gather(
                    1,
                    predicted_index[:, None, None, None].expand(
                        -1, 1, *pred.shape[-2:]
                    ),
                ).squeeze(1),  # (b, P, 3)
            }

        prediction = {"trajectory": trajectory}
        return TensorDict(prediction, batch_size=[pred.shape[0]])  # ty:ignore[invalid-argument-type]

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            msg = "no trainable parameters (freeze_base=True and no score_head?)"
            raise ValueError(msg)

        if self.optimizer is not None:
            optimizer = self.optimizer.instantiate(params=params)
        else:
            optimizer = torch.optim.Adam(params, lr=2e-4)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler.scheduler.instantiate(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler} | self.lr_scheduler.model_dump(
                exclude={"scheduler"}
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return {"optimizer": optimizer}

    @classmethod
    def load_for_score_head_training(  # noqa: PLR0913
        cls,
        artifact: str,
        *,
        score_head: Any,
        score_tau: float = 0.02,
        score_tau_relative: bool = False,
        score_weight: float = 1.0,
        optimizer: Any,
        lr_scheduler: Any | None = None,
        **_ignored: Any,
    ) -> "DrivoR":
        """Attach and train a `score_head` on top of an otherwise-FROZEN
        checkpoint (see the module docstring / plan history for why: no
        oracle at deployment to pick a trajectory with `best_index`).

        Loads weights AND saved hparams from the artifact, then:

        - sets `freeze_base=True`, which freezes (`requires_grad_(False)` +
          permanent `.eval()`, see `_base_modules`/`train`) every module
          except the new `score_head`;
        - instantiates `score_head` and attaches it, plus the scalar
          `score_tau`/`score_tau_relative`/`score_weight` knobs;
        - replaces the optimizer/lr_scheduler (fresh Adam moments).

        `**_ignored` swallows the architecture keys (`image_preprocess`,
        `backbone`, ...) the parent experiment inlines under `model.*` -- on
        this path the architecture comes from the checkpoint, not the config.
        """
        if not isinstance(optimizer, HydraConfig):
            optimizer = HydraConfig[Optimizer].model_validate(optimizer)
        if lr_scheduler is not None and not isinstance(
            lr_scheduler, LRSchedulerHydraConfig
        ):
            lr_scheduler = LRSchedulerHydraConfig.model_validate(lr_scheduler)
        if not isinstance(score_head, HydraConfig):
            score_head = HydraConfig[Module].model_validate(score_head)

        model = cls.load_from_wandb_artifact(
            artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        )

        model.freeze_base = True
        for module in model._base_modules():  # noqa: SLF001
            module.requires_grad_(False).eval()  # noqa: FBT003

        model.score_head = score_head.instantiate()
        model.score_tau = score_tau
        model.score_tau_relative = score_tau_relative
        model.score_weight = score_weight

        model.optimizer = optimizer
        model.lr_scheduler = lr_scheduler

        # dump_checkpoint reads `model.hparams` (the mutable dict), not the
        # `_hparams_initial` snapshot from `__init__` -- so every mutation
        # above must be written back here, or the resulting checkpoint trains
        # fine but nothing (not even plain `DrivoR.__init__`) can load it back.
        model.hparams["score_head"] = score_head.model_dump()
        model.hparams["freeze_base"] = True
        model.hparams["score_tau"] = score_tau
        model.hparams["score_tau_relative"] = score_tau_relative
        model.hparams["score_weight"] = score_weight
        model.hparams["optimizer"] = optimizer.model_dump()
        model.hparams["lr_scheduler"] = (
            lr_scheduler.model_dump() if lr_scheduler is not None else None
        )
        return model
