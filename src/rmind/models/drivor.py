from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from tensordict import TensorDict
from torch.nn import Module
from torch.optim import Optimizer

from rmind.components.drivor.loss import winner_takes_all_pose_l1
from rmind.components.drivor.trajectory_target import dead_reckon_future_trajectory
from rmind.config import HydraConfig
from rmind.models.control_transformer import LRSchedulerHydraConfig

CONTINUOUS_FIELDS = (
    "speed",
    "gas_pedal_normalized",
    "brake_pedal_normalized",
    "steering_angle_normalized",
)


class DrivoR(pl.LightningModule):
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
    and EKF+RTS-denoised heading (`dead_reckon_future_trajectory`), which are
    both densely, smoothly available at every window step -- unlike raw GNSS
    (~1Hz). `waypoints/xy_normalized` IS still used, but only as the model's
    driving-command substitute (see `EgoStateEncoder`/`route_tokenizer`
    below), since rmind has no NAVSIM-style discrete routing command.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        image_preprocess: HydraConfig[Module] | InstanceOf[Module],
        backbone: HydraConfig[Module] | InstanceOf[Module],
        register_projection: HydraConfig[Module] | InstanceOf[Module],
        route_tokenizer: HydraConfig[Module] | InstanceOf[Module],
        trajectory_head: HydraConfig[Module] | InstanceOf[Module],
        loss: HydraConfig[Module] | InstanceOf[Module],
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

        if optimizer is not None:
            hparams["optimizer"] = optimizer.model_dump()
        self.optimizer: HydraConfig[Optimizer] | None = optimizer

        if lr_scheduler is not None:
            hparams["lr_scheduler"] = lr_scheduler.model_dump()
        self.lr_scheduler: LRSchedulerHydraConfig | None = lr_scheduler

        self.reference_timestep = reference_timestep
        hparams["reference_timestep"] = reference_timestep

        self.save_hyperparameters(hparams)

    @override
    def train(self, mode: bool = True) -> "DrivoR":
        super().train(mode)
        self.route_tokenizer.eval()  # frozen, see __init__
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

        target_xy, target_heading = dead_reckon_future_trajectory(
            speed_kmh=data["meta/VehicleMotion/speed"],
            heading_deg=data["headings_denoised/heading"],
            time_stamp_s=data["meta/ImageMetadata.cam_front_left/time_stamp"],
            reference_index=t0,
        )

        return image, continuous, turn_signal, route, target_xy, target_heading

    def _forward(
        self,
        *,
        image: torch.Tensor,
        continuous: torch.Tensor,
        turn_signal: torch.Tensor,
        route: torch.Tensor,
    ) -> torch.Tensor:
        registers = self.backbone(self.image_preprocess(image))
        context = self.register_projection(registers)

        with torch.no_grad():
            route_embedding = self.route_tokenizer(route).squeeze(-2)

        return self.trajectory_head(
            context=context,
            ego_continuous=continuous,
            ego_turn_signal=turn_signal,
            ego_route_embedding=route_embedding,
        )

    @override
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        image, continuous, turn_signal, route, *_ = self._inputs(batch)
        return self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )

    @override
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        loss = self.loss(pred, target_xy, target_heading)
        self.log("train/loss", loss, sync_dist=True)
        return {"loss": loss}

    @override
    def validation_step(self, batch: dict[str, Any], _batch_idx: int) -> STEP_OUTPUT:
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        loss, best_index, per_candidate = winner_takes_all_pose_l1(
            pred, target_xy, target_heading
        )

        if not self.trainer.sanity_checking:
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/best_index_unique_frac": (
                        best_index.unique().numel() / best_index.numel()
                    ),
                    "val/per_candidate_loss_std": per_candidate.std(dim=-1).mean(),
                },
                sync_dist=True,
            )

        return {"loss": loss}

    @override
    def predict_step(self, batch: dict[str, Any], batch_idx: int = 0) -> TensorDict:
        image, continuous, turn_signal, route, target_xy, target_heading = self._inputs(
            batch
        )
        pred = self._forward(
            image=image, continuous=continuous, turn_signal=turn_signal, route=route
        )
        _, best_index, per_candidate = winner_takes_all_pose_l1(
            pred, target_xy, target_heading
        )
        best_pred = pred.gather(
            1, best_index[:, None, None, None].expand(-1, 1, *pred.shape[-2:])
        ).squeeze(1)

        prediction = {
            "trajectory": {
                "prediction": pred,
                "best_prediction": best_pred,
                "best_index": best_index,
                "per_candidate_loss": per_candidate,
                "ground_truth_xy": target_xy,
                "ground_truth_heading": target_heading,
            }
        }
        return TensorDict(prediction, batch_size=[pred.shape[0]])  # ty:ignore[invalid-argument-type]

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer is not None:
            optimizer = self.optimizer.instantiate(params=self.parameters())
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler.scheduler.instantiate(optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler} | self.lr_scheduler.model_dump(
                exclude={"scheduler"}
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        return {"optimizer": optimizer}
