import math
from typing import Any, Dict, List, NamedTuple, Tuple

import cv2
import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from deephouse.tools.camera import Camera  # pyright: ignore
from einops import rearrange
from hydra.utils import instantiate
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from cargpt.visualization.utils import get_images


class TrajectoryPoint(NamedTuple):
    x: float
    y: float
    z: float
    phi: float
    t: float
    v: float


class Trajectory(pl.LightningModule):
    def __init__(
        self,
        inference_model: DictConfig,
        images_transform: DictConfig,
        ground_truth_trajectory: DictConfig,
        car: DictConfig,
        logging: DictConfig,
    ) -> None:
        super().__init__()
        self.model = instantiate(inference_model)
        self.images_transform = instantiate(images_transform)

        self.wheelbase: float = car.wheelbase
        self.max_beta: float = math.radians(car.max_beta)
        self.turn_radius = self.wheelbase / math.sin(self.max_beta)

        self.gt_steps: int = ground_truth_trajectory.steps
        self.gt_time_interval: float = ground_truth_trajectory.time_interval
        self.logging = logging

        self.in_to_out = {
            "VehicleMotion_speed": "VehicleMotion_speed",  # km / h
            "VehicleMotion_steering_angle_normalized": "VehicleMotion_steering_angle_normalized",
            "VehicleMotion_gas_pedal_normalized": "VehicleMotion_gas_pedal_normalized",
            "VehicleMotion_brake_pedal_normalized": "VehicleMotion_brake_pedal_normalized",
            "VehicleMotion_acceleration_x": "VehicleMotion_acceleration_x",  # m / s2
            "VehicleMotion_acceleration_y": "VehicleMotion_acceleration_y",  # m / s2
            "VehicleMotion_acceleration_z": "VehicleMotion_acceleration_z",
        }

    def get_model_trajactories(self, batch):
        preds_last = self.model.predict_step(
            batch,
            batch_idx=0,
            start_timestep=-1,
            verbose=self.logging.log_to_terminal,
        )

        prediction_actions_ = []
        for _batch_key, batch_dict in sorted(preds_last.items()):
            ts_key = next(
                iter(batch_dict),
            )  # there is only one key for the last timestep
            prediction_actions_.append(
                [batch_dict[ts_key][key]["pred"] for key in self.model.action_keys],
            )
        prediction_actions_ = torch.tensor(prediction_actions_, device=self.device)

        return {
            self.in_to_out[in_key]: prediction_actions_[:, idx]
            for idx, in_key in enumerate(self.model.action_keys)
        }

    def predict_step(
        self,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        images = rearrange(
            get_images(batch, self.images_transform),
            "b f c h w -> b f h w c",
        )
        batch_size, clip_len, *_ = images.shape
        # assume batch_size is 1, at least for now
        if batch_size > 1:
            msg = (
                "Assumed batch size equals 1, increase clip length to process more data"
            )

            raise NotImplementedError(msg)

        images = images.squeeze(0)  # kick out batch dimension
        images = images[[clip_len - 1]]
        camera = Camera.from_params(mit.one(batch["camera_params"].values()))
        metadata = self.prepare_metadata(batch, clip_len - 1)

        # Ground-truth trajactories
        gt_points_3d: Float[Tensor, "f n 3"] = self.get_trajectory_3d_points(
            steps=self.gt_steps,
            time_interval=self.gt_time_interval,
            **metadata,
        )

        gt_points_2d: Float[Tensor, "f n 2"] = rearrange(
            camera.project(rearrange(gt_points_3d, "f n d -> (f n) 1 1 d")),
            "(f n) 1 1 d -> f n (1 1 d)",
            f=1,
        )

        visualizations: np.ndarray = np.ascontiguousarray(
            (images * 255).int().cpu().numpy().astype(np.uint8),
        )
        draw_trajectory(visualizations, gt_points_2d)

        # Model prediction trajactories
        model_actions = self.get_model_trajactories(batch)
        metadata["VehicleMotion_gas_pedal_normalized"] = model_actions[
            "VehicleMotion_gas_pedal_normalized"
        ]
        metadata["VehicleMotion_brake_pedal_normalized"] = model_actions[
            "VehicleMotion_brake_pedal_normalized"
        ]
        metadata["VehicleMotion_steering_angle_normalized"] = model_actions[
            "VehicleMotion_steering_angle_normalized"
        ]

        pred_points_3d: Float[Tensor, "f n 3"] = self.get_trajectory_3d_points(
            steps=self.gt_steps,
            time_interval=self.gt_time_interval,
            **metadata,
        )

        pred_points_2d: Float[Tensor, "f n 2"] = rearrange(
            camera.project(rearrange(pred_points_3d, "f n d -> (f n) 1 1 d")),
            "(f n) 1 1 d -> f n (1 1 d)",
            f=1,
        )

        if not self.logging.smooth_predictions:
            draw_trajectory(
                visualizations,
                pred_points_2d,
                point_color=(0, 255, 0),
                line_color=(0, 255, 0),
            )

            draw_preds(visualizations, metadata, line_color=(0, 255, 0))

        return (visualizations, metadata)

    def prepare_metadata(self, batch, clip_index):
        meta = batch["meta"]

        out = {
            out_key: meta[in_key][:, [clip_index]].to(self.device)
            for in_key, out_key in self.in_to_out.items()
        }
        # units change
        out["VehicleMotion_speed"] *= 10 / 36  # change to m / s

        # Assumption: batch size = 1
        # if the assumption changes, kick out the flattening
        return {k: v.squeeze(0) for k, v in out.items()}

    def get_trajectory_3d_points(
        self,
        *,
        steps,
        time_interval,
        VehicleMotion_speed,
        VehicleMotion_acceleration_y,
        VehicleMotion_steering_angle_normalized,
        VehicleMotion_gas_pedal_normalized,
        VehicleMotion_brake_pedal_normalized,
    ) -> Float[Tensor, "f n 3"]:
        clips, *_ = VehicleMotion_speed.shape
        points_3d: List[List[Tuple[float, float, float]]] = []

        for i in range(clips):
            start_point = TrajectoryPoint(
                x=0.0,
                y=1.5,
                z=1e-6,
                phi=math.radians(0.0),
                v=VehicleMotion_speed[i].item(),
                t=0.0,
            )
            curr_points = self.calculate_trajectory(
                last_elem=start_point,
                steps=steps,
                time_interval=time_interval,
                acceleration_y=VehicleMotion_acceleration_y[i].item(),
                gas_norm=VehicleMotion_gas_pedal_normalized[i].item(),
                brake_norm=VehicleMotion_brake_pedal_normalized[i].item(),
                steering_wheel_norm=VehicleMotion_steering_angle_normalized[i].item(),
            )
            points_3d.append([(p.x, p.y, p.z) for p in [start_point, *curr_points]])

        return torch.tensor(points_3d, device=self.device)

    def calculate_trajectory(
        self,
        last_elem: TrajectoryPoint,
        steps: int,
        time_interval: float,
        acceleration_y: float,
        gas_norm: float,
        brake_norm: float,
        steering_wheel_norm: float,
    ) -> List[TrajectoryPoint]:
        # https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
        # Fig. 26.
        # Their y-axis is z-axis, their time_interval is 1., their delta is our beta
        elems: List[TrajectoryPoint] = []
        # https://yaakai.slack.com/archives/CQKL412BC/p1685447717475339
        a = math.copysign(acceleration_y, gas_norm - brake_norm)
        beta = steering_wheel_norm * self.max_beta
        # norm: minus means left, plus means right
        for _step in range(steps):
            p = last_elem
            x = p.x + p.v * math.sin(p.phi) * time_interval
            z = p.z + p.v * math.cos(p.phi) * time_interval
            phi = p.phi + (p.v * math.tan(beta) / self.wheelbase) * time_interval
            v = p.v + a * time_interval
            p_next = TrajectoryPoint(
                x=x,
                y=p.y,
                z=z,
                phi=phi,
                v=v,
                t=p.t + time_interval,
            )
            elems.append(p_next)
            last_elem = p_next
        return elems


def draw_preds(
    visualizations: np.ndarray,
    metadata: Dict,
    line_color: Tuple[int, int, int] = (255, 255, 255),
):
    image = visualizations[0]
    w = image.shape[1]
    brake = metadata["VehicleMotion_brake_pedal_normalized"].item()
    text = f"[Brake:{brake:.3f}]"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_w, text_h = text_size
    pos = (0, w // 2)
    x, y = pos
    _ = cv2.rectangle(image, pos, (x + text_w, y + text_h), (0, 0, 0), -1)
    image = cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        line_color,
        2,
        cv2.LINE_AA,
    )


def draw_trajectory(
    visualizations: np.ndarray,
    points_2d: Float[Tensor, "f n 2"],
    line_color: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 1,
    point_color: Tuple[int, int, int] = (255, 0, 0),
    point_thickness: int = 2,
    point_radius: int = 1,
) -> None:
    points_2d = points_2d.cpu().numpy().astype(np.int32)
    close_polygon = False
    for i, (_vis, points) in enumerate(zip(visualizations, points_2d)):
        vis = cv2.polylines(
            _vis,
            [points],
            close_polygon,
            line_color,
            line_thickness,
        )
        for point in points:
            vis = cv2.circle(vis, point, point_radius, point_color, point_thickness)  # type: ignore
        visualizations[i] = vis


def smooth_predictions(metadatas, window_size: int = 6):
    gas = (
        torch.cat([m["VehicleMotion_gas_pedal_normalized"] for m in metadatas])
        .cpu()
        .numpy()
    )
    brake = (
        torch.cat([m["VehicleMotion_brake_pedal_normalized"] for m in metadatas])
        .cpu()
        .numpy()
    )
    steer = (
        torch.cat([m["VehicleMotion_steering_angle_normalized"] for m in metadatas])
        .cpu()
        .numpy()
    )

    window = np.ones(window_size) / window_size
    gas = np.convolve(gas, window, mode="same")
    brake = np.convolve(brake, window, mode="same")
    steer = np.convolve(steer, window, mode="same")

    for g, b, s, metadata in zip(gas, brake, steer, metadatas):
        metadata["VehicleMotion_gas_pedal_normalized"][:] = g
        metadata["VehicleMotion_brake_pedal_normalized"][:] = b
        metadata["VehicleMotion_steering_angle_normalized"][:] = s

    return metadatas
