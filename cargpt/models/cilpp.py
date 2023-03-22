from functools import lru_cache
from typing import Annotated, Dict, Optional, Sequence, Tuple, cast

import more_itertools as mit
import pytorch_lightning as pl
import torch
from deephouse.tools.camera import Camera
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from jaxtyping import Float, Shaped
from omegaconf import DictConfig
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor
from torch.nn import Module, ModuleDict, TransformerEncoder
from torchvision.transforms.functional import resize

import wandb
from cargpt.utils.wandb import LoadableFromArtifact


class DepthFrameEncoder(Module):
    def __init__(
        self,
        disp_net: pl.LightningModule,
        point_positional_encoder: Module,
        target_shape: Optional[Annotated[Sequence[int], 2]] = None,
    ) -> None:
        super().__init__()

        self.disp_net = disp_net
        self.disp_net.freeze()

        self.point_positional_encoder = point_positional_encoder
        self.target_shape: Optional[Tuple[int, ...]] = (
            tuple(target_shape) if target_shape is not None else None
        )

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

    hparams: AttributeDict

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.state_embedding: ModuleDict = instantiate(self.hparams.state_embedding)
        self.transformer_encoder: TransformerEncoder = instantiate(
            self.hparams.transformer_encoder
        )
        self.action_prediction: Module = instantiate(self.hparams.action_prediction)
        self.loss: DictConfig = instantiate(self.hparams.loss)

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

    def _compute_predictions(self, batch) -> Dict[str, Float[Tensor, "b 1"]]:
        clips = mit.one(batch["clips"].values())
        frames = rearrange(clips["frames"], "b 1 c h w -> b c h w")

        meta = clips["meta"]
        speed = meta["VehicleMotion_speed"].to(torch.float32)

        if isinstance(_camera_params := clips.get("camera_params"), dict):
            camera_params = _camera_params.copy()
            camera_model = mit.one(set(camera_params.pop("model")))
            camera = Camera.from_params(model=camera_model, params=camera_params)
            camera = camera.to(frames)
        else:
            camera = None

        pred = self.forward(frames=frames, speed=speed, camera=camera)
        accel_pred, steering_pred = rearrange(pred, "b 1 c -> c b 1")

        return {"acceleration": accel_pred, "steering_angle": steering_pred}

    def _compute_labels(self, batch) -> Dict[str, Float[Tensor, "b 1"]]:
        clips = mit.one(batch["clips"].values())
        meta = clips["meta"]

        gas = meta["VehicleMotion_gas_pedal_normalized"]
        brake = meta["VehicleMotion_brake_pedal_normalized"]
        # NOTE: assuming (gas > 0) xor (brake > 0)
        accel_lbl = gas - brake
        steering_lbl = meta["VehicleMotion_steering_angle_normalized"]

        return {
            "acceleration": accel_lbl.to(torch.float32),
            "steering_angle": steering_lbl.to(torch.float32),
        }

    def training_step(self, batch, batch_idx):
        predictions = self._compute_predictions(batch)
        labels = self._compute_labels(batch)

        losses = {
            k: self.loss[k](predictions[k], labels[k])
            for k in ("acceleration", "steering_angle")
        }

        losses["total"] = sum(self.loss.weights[k] * v for k, v in losses.items())

        self.log_dict(
            {f"train/loss/{k}": v for k, v in losses.items()},
            sync_dist=True,
            rank_zero_only=True,
        )

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        preds = self._compute_predictions(batch)
        labels = self._compute_labels(batch)

        losses = {
            k: self.loss[k](preds[k], labels[k])
            for k in ("acceleration", "steering_angle")
        }

        losses["total"] = sum(self.loss.weights[k] * v for k, v in losses.items())

        self.log_dict(
            {f"val/loss/{k}": v for k, v in losses.items()},
            sync_dist=True,
            rank_zero_only=True,
        )

        if hasattr(self, "_tables"):
            if table := self._tables.get("frames"):
                clips = batch["clips"]
                frames = [clips[camera]["frames"] for camera in table.columns]
                for row in zip(*frames):
                    images = [wandb.Image(frame) for frame in row]
                    table.add_data(*images)

            if table := self._tables.get("outputs"):
                data: Float[Tensor, "b 4"] = rearrange(
                    [  # type: ignore
                        preds["acceleration"],
                        labels["acceleration"],
                        preds["steering_angle"],
                        labels["steering_angle"],
                    ],
                    "t b 1 -> b t",
                )

                for row in data.tolist():
                    table.add_data(*row)

        return losses["total"]

    def on_validation_epoch_start(self):
        if isinstance(self.logger, pl.loggers.WandbLogger) and (
            log_cfg := self.hparams.get("log", {}).get("validation", {})
        ):
            self._tables = {}

            if log_cfg.get(name := "frames") and self.current_epoch == 0:
                dataset = self.trainer.val_dataloaders[0].dataset  # type: ignore[index]
                cameras = [x.value for x in dataset.config.data.metadata.cameras]  # type: ignore[union-attr]
                self._tables[name] = wandb.Table(columns=cameras)

            if log_cfg.get(name := "outputs"):
                self._tables[name] = wandb.Table(
                    columns=[
                        "acceleration_pred",
                        "acceleration_label",
                        "steering_angle_pred",
                        "steering_angle_label",
                    ]
                )

    def on_validation_epoch_end(self):
        if hasattr(self, "_tables") and self.trainer.state.stage != "sanity_check":
            run = self.logger.experiment  # type: ignore[union-attr]

            if table := self._tables.get(name := "frames"):
                table.add_column("_step", list(map(int, table.get_index())))
                artifact = wandb.Artifact(f"run-{run.id}-val_{name}", "run_table")
                artifact.add(table, name)
                frame_artifact = run.log_artifact(artifact)
            else:
                frame_artifact = None

            if table := self._tables.get(name := "outputs"):
                table.add_column("_step", list(map(int, table.get_index())))

                if frame_artifact is not None:
                    frame_artifact.wait()
                else:
                    try:
                        frame_artifact = run.use_artifact(f"run-{run.id}-val_frames:v0")
                    except wandb.CommError:
                        pass

                artifact = wandb.Artifact(f"run-{run.id}-val_{name}", "run_table")

                if frame_artifact is not None:
                    frame_table = frame_artifact.get_path("frames.table.json")
                    joined_table = wandb.JoinedTable(frame_table, table, "_step")
                    artifact.add(joined_table, name)
                else:
                    artifact.add(table, name)

                run.log_artifact(artifact)

    def _embed_state(
        self,
        *,
        frames: Float[Tensor, "b 3 h w"],
        speed: Float[Tensor, "b 1"],
        camera: Optional[Camera] = None,
    ) -> Float[Tensor, "b s c"]:
        feats = []

        if encoder := getattr(self.state_embedding.frame, "backbone", None):
            feat = encoder(frames)
            feat = rearrange(feat, "b c h w -> b (h w) c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding.frame, "depth", None):
            if camera is None:
                raise ValueError("camera required for depth encoder")

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
