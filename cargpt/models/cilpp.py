from functools import lru_cache
from typing import Annotated, Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import more_itertools as mit
import pytorch_lightning as pl
import torch
from deephouse.tools.camera import Camera
from einops import rearrange, reduce, repeat
from hydra.utils import instantiate
from jaxtyping import Float, Int, Shaped
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor
from torch.nn import Module, ModuleDict, TransformerEncoder
from torch.nn import functional as F
from torchvision.transforms.functional import resize

import wandb
from cargpt.utils._wandb import LoadableFromArtifact
from wandb.data_types import JoinedTable
from wandb.errors import CommError
from wandb.sdk.interface.artifacts import Artifact, ArtifactManifestEntry
from wandb.wandb_run import Run


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
        frames: Float[Tensor, "*b c1 h1 w1"],
        camera: Camera,
    ) -> Float[Tensor, "*b h2 w2 c2"]:
        *b, _, _, _ = frames.shape
        frames = rearrange(frames, "... c h w -> (...) c h w")

        # compute depth
        disp = self.disp_net(frames)[0]
        depth = rearrange(1 / disp, "b 1 h w -> b h w 1")

        # compute 3d points (equivalent to Eq 6 [https://arxiv.org/abs/2211.14710])
        B_frame, _, H_frame, W_frame = frames.shape
        grid_2d = self._generate_2d_grid(height=H_frame, width=W_frame).to(depth)
        pts_2d = repeat(grid_2d, "h w t -> b h w t", b=B_frame)
        pts = camera.unproject(pts_2d, depth)  # type: ignore

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
        pt_pos_enc = pt_pos_enc.view(*b, *pt_pos_enc.shape[-3:])

        return pt_pos_enc

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_2d_grid(cls, *, height: int, width: int) -> Shaped[Tensor, "h w 2"]:
        x_mesh, y_mesh = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )

        grid = rearrange([x_mesh, y_mesh], "t h w -> h w t")

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
        frames: Float[Tensor, "b t v c h w"],
        speed: Float[Tensor, "b t"],
        turn_signal: Int[Tensor, "b t"],
        camera: Optional[Camera] = None,
    ) -> Float[Tensor, "b 1 2"]:
        state = self._embed_state(
            frames=frames,
            speed=speed,
            turn_signal=turn_signal,
            camera=camera,
        )
        encoded = self.transformer_encoder(state)
        pred = self.action_prediction(encoded)

        return pred

    def unpack_batch_for_predictions(
        self, batch: Mapping[str, Any]
    ) -> Tuple[
        Float[Tensor, "b t v c h w"],
        Float[Tensor, "b t"],
        Int[Tensor, "b t"],
        Optional[Camera],
    ]:
        clips = batch["clips"]
        camera_names = sorted(clips.keys())
        frames = cast(
            Float[Tensor, "b t v c h w"],
            rearrange(
                [clips[k]["frames"] for k in camera_names],
                "v b t c h w -> b t v c h w",
            ),
        )

        meta = clips[camera_names[0]]["meta"]

        speed = meta["VehicleMotion_speed"].to(torch.float32)
        turn_signal = meta["VehicleState_turn_signal"]

        if isinstance(_camera_params := clips.get("camera_params"), dict):
            camera_params = _camera_params.copy()
            camera_model = mit.one(set(camera_params.pop("model")))
            _, t, *_ = frames.shape
            # need one camera per frame
            camera_params = {
                k: repeat(v, "b -> (b t)", t=t) for k, v in camera_params.items()
            }
            camera = Camera.from_params(model=camera_model, params=camera_params)
            camera = camera.to(frames)
        else:
            camera = None

        return frames, speed, turn_signal, camera

    def _compute_predictions(self, batch) -> Dict[str, Float[Tensor, "b 1"]]:
        frames, speed, turn_signal, camera = self.unpack_batch_for_predictions(batch)
        pred = self.forward(
            frames=frames,
            speed=speed,
            turn_signal=turn_signal,
            camera=camera,
        )
        accel_pred, steering_pred = rearrange(pred, "b 1 c -> c b 1")

        return {"acceleration": accel_pred, "steering_angle": steering_pred}

    def _compute_labels(self, batch) -> Dict[str, Float[Tensor, "b 1"]]:
        clips = batch["clips"]
        camera_names = sorted(clips.keys())
        meta = clips[camera_names[0]]["meta"]

        gas = meta["VehicleMotion_gas_pedal_normalized"]
        brake = meta["VehicleMotion_brake_pedal_normalized"]
        steering_angle = meta["VehicleMotion_steering_angle_normalized"]

        # NOTE: taking last timestep as label
        # NOTE: assuming (gas > 0) xor (brake > 0)
        accel_lbl = gas[..., [-1]] - brake[..., [-1]]
        steering_angle_lbl = steering_angle[..., [-1]]

        return {
            "acceleration": accel_lbl.to(torch.float32),
            "steering_angle": steering_angle_lbl.to(torch.float32),
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
                    [
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
        if (
            isinstance(logger := self.logger, WandbLogger)
            and isinstance(run := logger.experiment, Run)
            and (log_cfg := self.hparams.get("log", {}).get("validation", {}))
        ):
            self._tables = {}

            if log_cfg.get(name := "frames"):
                try:
                    _ = run.use_artifact(f"run-{run.id}-val_frames:latest")
                except CommError:
                    dataset = self.trainer.val_dataloaders[0].dataset  # type: ignore
                    cameras = [x.value for x in dataset.config.data.metadata.cameras]  # type: ignore
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
            run: Run = self.logger.experiment  # type: ignore

            frame_artifact: Optional[Artifact] = None

            if table := self._tables.get(name := "frames"):
                table.add_column("_step", list(map(int, table.get_index())))
                artifact = wandb.Artifact(f"run-{run.id}-val_{name}", "run_table")
                artifact.add(table, name)
                frame_artifact = run.log_artifact(artifact)

            if table := self._tables.get(name := "outputs"):
                table.add_column("_step", list(map(int, table.get_index())))

                if frame_artifact is not None:
                    frame_artifact.wait()
                else:
                    try:
                        frame_artifact = run.use_artifact(
                            f"run-{run.id}-val_frames:latest"
                        )
                    except CommError:
                        # may have never been logged in the first place
                        pass

                frame_table: Optional[ArtifactManifestEntry] = None
                if frame_artifact is not None:
                    with logger.catch(message="failed to get frame table"):
                        frame_table = frame_artifact.get_path("frames.table.json")

                _table = (
                    table
                    if frame_table is None
                    else JoinedTable(frame_table, table, "_step")
                )
                artifact = wandb.Artifact(f"run-{run.id}-val_{name}", "run_table")
                artifact.add(_table, name)
                run.log_artifact(artifact)

    def _embed_state(
        self,
        *,
        frames: Float[Tensor, "b t v c h w"],
        speed: Float[Tensor, "b t"],
        turn_signal: Int[Tensor, "b t"],
        camera: Optional[Camera] = None,
    ) -> Float[Tensor, "b s c"]:
        feats: List[Float[Tensor, "b t s c"]] = []

        if encoder := getattr(self.state_embedding.frame, "backbone", None):
            feat = encoder(frames)
            feat = rearrange(feat, "b t v c h w -> b t (v w h) c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding.frame, "depth", None):
            if camera is None:
                raise ValueError("camera required for depth encoder")

            feat = encoder(frames=frames, camera=camera)
            feat = rearrange(feat, "b t h w c -> b t (w h) c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding, "speed"):
            speed = rearrange(speed, "b t -> b t 1")
            feat = encoder(speed)
            feat = rearrange(feat, "b t c -> b t 1 c")
            feats.append(feat)

        if encoder := getattr(self.state_embedding, "turn_signal"):
            turn_signal = F.one_hot(turn_signal, encoder.in_features).float()
            feat = encoder(turn_signal)
            feat = rearrange(feat, "b t c -> b t 1 c")
            feats.append(feat)

        state = cast(
            Float[Tensor, "b s c"],
            rearrange(sum(feats), "b t s c -> b (t s) c"),
        )

        if encoder := getattr(self.state_embedding, "position"):
            state += encoder(state.shape)

        return state

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.parameters())
        result = {"optimizer": optimizer}

        if (lr_scheduler_cfg := self.hparams.get("lr_scheduler")) is not None:
            scheduler = instantiate(lr_scheduler_cfg.scheduler, optimizer=optimizer)
            result["lr_scheduler"] = {**lr_scheduler_cfg, **{"scheduler": scheduler}}

        return result
