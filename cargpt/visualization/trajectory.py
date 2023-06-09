import math
from collections import namedtuple
from typing import Any, List, Tuple

import cv2
import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
from deephouse.tools.camera import Camera
from einops import rearrange
from hydra.utils import instantiate
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from cargpt.visualization.utils import get_images

TrajectoryPoint = namedtuple("TrajectoryPoint", "x,y,z,phi,t,v")


class Trajectory(pl.LightningModule):
    def __init__(
        self,
        inference_model: DictConfig,
        images_transform: DictConfig,
        ground_truth_trajectory: DictConfig,
        car: DictConfig,
    ):
        super().__init__()
        self.model = instantiate(inference_model)
        self.images_transform = instantiate(images_transform)

        self.wheelbase: float = car.wheelbase
        self.max_beta: float = math.radians(car.max_beta)
        self.turn_radius = self.wheelbase / math.sin(self.max_beta)

        self.gt_steps: int = ground_truth_trajectory.steps
        self.gt_time_interval: float = ground_truth_trajectory.time_interval

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> np.ndarray:
        images = rearrange(
            get_images(batch, self.images_transform), "b f c h w -> b f h w c"
        )
        batch_size, clip_len, *_ = images.shape
        # assume batch_size is 1, at least for now
        if batch_size > 1:
            raise NotImplementedError(
                "Assumed batch size equals 1, increase clip length to process more data"
            )

        images = images.squeeze(0)  # kick out batch dimension
        camera = self.get_camera(batch)
        metadata = self.prepare_metadata(batch)
        gt_points_3d: Float[Tensor, "f n 3"] = self.get_trajectory_3d_points(
            steps=self.gt_steps,
            time_interval=self.gt_time_interval,
            **metadata,
        )

        gt_points_2d: Float[Tensor, "f n 2"] = rearrange(
            camera.project(rearrange(gt_points_3d, "f n d -> (f n) 1 1 d")),  # type: ignore
            "(f n) 1 1 d -> f n (1 1 d)",
            f=clip_len,
        )

        visualizations: np.ndarray = np.ascontiguousarray(
            (images * 255).int().cpu().numpy().astype(np.uint8)
        )
        draw_trajectory(visualizations, gt_points_2d)
        # TODO: add debug drawings: straigh line + 1/4 circle
        return visualizations

    def prepare_metadata(self, batch):
        clips = mit.one(batch["clips"].values())
        meta = clips["meta"]
        in_to_out = {
            "VehicleMotion_speed": "speed",  # km / h
            "VehicleMotion_steering_angle_normalized": "steering_norm",
            "VehicleMotion_acceleration_x": "acceleration_x",  # m / s
            "VehicleMotion_acceleration_z": "acceleration_z",
        }
        out = {
            out_key: meta[in_key].to(self.device)
            for in_key, out_key in in_to_out.items()
        }
        # units change
        out["speed"] *= 10 / 36  # change to m / s

        # Assumption: batch size = 1
        # if the assumption changes, kick out the flattening
        out = {k: v.squeeze(0) for k, v in out.items()}
        return out

    def get_trajectory_3d_points(
        self,
        *,
        steps,
        time_interval,
        speed,
        acceleration_x,
        acceleration_z,
        steering_norm,
    ) -> Float[Tensor, "f n 3"]:
        clips, *_ = speed.shape
        points_3d: List[List[Tuple[float, float, float]]] = []

        for i in range(clips):
            start_point = TrajectoryPoint(
                x=0.0,
                y=1.5,
                z=1e-6,
                phi=math.radians(90.0),
                v=speed[i].item(),
                t=0.0,
            )
            curr_points = self.calculate_trajectory(
                steps=steps,
                time_interval=time_interval,
                acceleration_x=acceleration_x[i].item(),
                acceleration_z=acceleration_z[i].item(),
                steering_wheel_norm=steering_norm[i].item(),
                last_elem=start_point,
            )
            points_3d.append([(p.x, p.y, p.z) for p in [start_point] + curr_points])

        return torch.tensor(points_3d, device=self.device)

    def get_camera(self, batch) -> Camera:
        clips = mit.one(batch["clips"].values())
        camera_params = dict(clips["camera_params"])
        model = mit.one(camera_params.pop("model"))
        camera = Camera.from_params(model=model, params=camera_params)
        return camera

    def calculate_trajectory(
        self,
        last_elem: TrajectoryPoint,
        steps: int,
        time_interval: float,
        acceleration_x: float,
        acceleration_z: float,
        steering_wheel_norm: float = 0.0,
    ) -> List[TrajectoryPoint]:
        # https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
        # Fig. 26.
        # Their y-axis is z-axis, their time_interval is 1., their delta is our beta
        elems: List[TrajectoryPoint] = []
        a = math.sqrt(acceleration_x**2 + acceleration_z**2)
        beta = math.radians(
            -steering_wheel_norm * self.max_beta
        )  # norm: minus means left, plus means right
        for step in range(steps):
            p = last_elem
            v = p.v + a * time_interval
            p_next = TrajectoryPoint(
                x=p.x + v * math.cos(p.phi + beta) * time_interval,
                y=p.y,
                z=p.z + v * math.sin(p.phi + beta) * time_interval,
                phi=p.phi + time_interval * math.tan(beta) * v / self.wheelbase,
                v=v + a * time_interval,
                t=p.t + time_interval,
            )
            elems.append(p_next)
            last_elem = p_next
        return elems


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
    for i, (vis, points) in enumerate(zip(visualizations, points_2d)):
        vis = cv2.polylines(
            vis,
            [points],
            close_polygon,
            line_color,
            line_thickness,
        )
        for point in points:
            vis = cv2.circle(
                vis, point, point_radius, point_color, point_thickness  # type: ignore
            )
        visualizations[i] = vis
