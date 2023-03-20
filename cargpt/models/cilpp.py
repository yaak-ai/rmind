from functools import lru_cache
from typing import Annotated, Optional, Sequence, cast

import more_itertools as mit
import pytorch_lightning as pl
import torch
from deephouse.tools.camera import Camera
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from jaxtyping import Float, Shaped
from torch import Tensor
from torchvision.transforms.functional import resize

from cargpt.utils.wandb import LoadableFromArtifact


class DepthFrameEncoder(torch.nn.Module):
    def __init__(
        self,
        disp_net: pl.LightningModule,
        point_positional_encoder: torch.nn.Module,
        target_shape: Optional[Annotated[Sequence[int], 2]] = None,
    ):
        super().__init__()

        self.disp_net = disp_net
        self.disp_net.freeze()

        self.point_positional_encoder = point_positional_encoder
        self.target_shape = tuple(target_shape) if target_shape is not None else None

    def forward(
        self,
        *,
        frames: Float[Tensor, "b c1 h1 w1"],
        camera: Camera,
    ) -> Float[Tensor, "b h2 w2 c2"]:
        # compute depth
        disp = self.disp_net(frames)[0]
        depth = rearrange(1 / disp, "b 1 h w -> b h w 1")

        # compute 3d points (equivalent to Eq 6 [https://arxiv.org/abs/2211.14710])
        B, _, H_frame, W_frame = frames.shape
        grid_2d = self._generate_2d_grid(height=H_frame, width=W_frame).to(depth)
        pts_2d = repeat(grid_2d, "h w t -> b h w t", b=B)
        pts = camera.unproject(pts_2d, depth)

        # normalize 3d points (Eq 7 [https://arxiv.org/abs/2211.14710])
        pts_min = reduce(pts, "b h w c -> b 1 1 c", "min")
        pts_max = reduce(pts, "b h w c -> b 1 1 c", "max")
        pts_norm = (pts - pts_min) / (pts_max - pts_min)

        if self.target_shape is not None and self.target_shape != (H_frame, W_frame):
            # sample x,y,z coordinates to match target shape
            # NOTE: using `resize` this way _should_ be equivalent to `grid_sample`
            pts_norm = rearrange(pts_norm, "b h w c -> b c h w")
            pts_norm = resize(pts_norm, list(self.target_shape))
            pts_norm = rearrange(pts_norm, "b c h w -> b h w c")

        pt_pos_enc = self.point_positional_encoder(pts_norm)

        return pt_pos_enc

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_2d_grid(cls, *, height: int, width: int) -> Shaped[Tensor, "h w 2"]:
        x_mesh, y_mesh = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )

        grid = rearrange([x_mesh, y_mesh], "t h w -> h w t")  # type: ignore

        return grid


class CILpp(pl.LightningModule, LoadableFromArtifact):
    """CIL++

    src: Scaling Self-Supervised End-to-End Driving with Multi-View Attention Learning
    (https://arxiv.org/abs/2302.03198)
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.state_embedding = instantiate(self.hparams.state_embedding)
        self.transformer_encoder = instantiate(self.hparams.transformer_encoder)
        self.action_prediction = instantiate(self.hparams.action_prediction)
        self.loss = instantiate(self.hparams.loss)

    def forward(
        self,
        *,
        frames: Float[Tensor, "b c h w"],
        speed: Float[Tensor, "b 1"],
        camera: Optional[Camera] = None,
    ) -> Float[Tensor, "b 1 2"]:
        state = self._embed_state(frames=frames, speed=speed, camera=camera)
        encoded = self.transformer_encoder(state)
        pred = self.action_prediction(encoded)

        return pred

    def training_step(self, *args, **kwargs):
        return self._step("train", *args, **kwargs)

    def _step(self, stage, batch, batch_idx):
        clips = mit.one(batch["clips"].values())
        frames = rearrange(clips["frames"], "b 1 c h w -> b c h w")

        camera_params = clips["camera_params"]
        camera_model = mit.one(set(camera_params.pop("model")))
        camera = Camera.from_params(model=camera_model, params=camera_params).to(frames)

        meta = clips["meta"]
        speed = meta["VehicleMotion_speed"].to(torch.float32)

        pred = self.forward(frames=frames, speed=speed, camera=camera)
        accel_pred, steering_pred = rearrange(pred, "b 1 c -> c b 1")

        gas = meta["VehicleMotion_gas_pedal_normalized"].to(pred)
        brake = meta["VehicleMotion_brake_pedal_normalized"].to(pred)
        # NOTE: assuming (gas > 0) xor (brake > 0)
        accel_lbl = gas - brake
        steering_lbl = meta["VehicleMotion_steering_angle_normalized"].to(pred)

        losses = {
            "acceleration": self.loss.acceleration(accel_pred, accel_lbl),
            "steering_angle": self.loss.steering_angle(steering_pred, steering_lbl),
        }

        losses["total"] = sum(self.loss.weights[k] * v for k, v in losses.items())

        self.log_dict(
            {f"{stage}/loss/{k}": v for k, v in losses.items()},
            sync_dist=True,
            rank_zero_only=True,
        )

        return losses["total"]

    def _embed_state(
        self,
        *,
        frames: Float[Tensor, "b 3 h w"],
        speed: Float[Tensor, "b 1"],
        camera: Camera,
    ) -> Float[Tensor, "b s c"]:
        feats = []

        if encoder := getattr(self.state_embedding.frame, "backbone", None):
            feat = encoder(frames)
            feat = rearrange(feat, "b c h w -> b (h w) c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding.frame, "depth", None):
            feat = encoder(frames=frames, camera=camera)
            feat = rearrange(feat, "b h w c -> b (h w) c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding, "speed"):
            feat = encoder(speed)
            feats.append(feat)

        if encoder := getattr(self.state_embedding, "position"):
            feat = encoder(feats[0].shape)
            feats.append(feat)

        state = cast(Float[Tensor, "b s c"], sum(feats))

        return state

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result
