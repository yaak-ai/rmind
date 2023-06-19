import math
from collections import namedtuple
from typing import Any, List, Tuple

import cv2
import more_itertools as mit
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deephouse.tools.camera import Camera
from einops import rearrange
from hydra.utils import instantiate
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from cargpt.visualization.utils import get_images

TrajectoryPoint = namedtuple("TrajectoryPoint", "x,y,z,phi,t,v")
np.set_printoptions(suppress=True)


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
        self.model.eval()
        self.images_transform = instantiate(images_transform)

        self.wheelbase: float = car.wheelbase
        self.max_beta: float = math.radians(car.max_beta)
        self.turn_radius = self.wheelbase / math.sin(self.max_beta)

        self.gt_steps: int = ground_truth_trajectory.steps
        self.gt_time_interval: float = ground_truth_trajectory.time_interval

    def get_model_trajactories(self, batch):
        sample = self.model.prepare_batch(batch)

        b, clips, c, h, w = sample["frames"].shape
        episode, labels, _, episode_values = self.model._make_episode(sample)
        b, seqlen, d = episode.shape
        num_actions = len(self.model.action_keys)
        # Take out action we don't use it for predictions
        observations, _ = torch.split(
            episode, [seqlen - num_actions, num_actions], dim=1
        )

        action_values = []
        action_tokens = []
        for idx in range(num_actions):
            action_key = self.model.action_keys[idx]
            # we only pass observations to the model and let it predict action 1 at a time
            pred, _ = self.model.forward(episode=observations)
            # can be argmax of multinomial sampling
            action_class = torch.argmax(F.softmax(pred, dim=2), dim=2)
            # get last action predicted
            action_token = action_class[:, [-1]]
            action_tokens.append(action_token.clone())
            action_embedding = self.model.sensor_embedding(action_token)
            # add action position embedding
            action_embedding += self.model.action_position(
                torch.tensor([0], device=action_embedding.device)
            ).view(b, 1, d)
            global_positions = torch.tensor([clips - 1]).to(action_embedding.device)
            global_positions_encoded = self.model.global_position(
                global_positions
            ).view(1, 1, d)
            action_embedding += global_positions_encoded
            # next observation takes the past action
            observations = torch.cat([observations, action_embedding], dim=1)
            # unshift and detokenize
            action_token -= self.model.hparams.tokens_shift[action_key]
            detokenizer = self.model.sensor_detokenization[action_key]
            action_value = detokenizer(action_token)
            action_values.append(action_value)

        # computed_actions.append(detokenized_actions)
        prediction_actions = torch.cat(action_values)
        # predected_tokens = torch.cat(action_tokens).flatten().cpu().numpy().tolist()
        # print(f"pred {np.array2string(prediction_actions, precision=3, floatmode='fixed')}")
        # print(
        #     f"gt: {np.array2string(episode_values[:, -num_actions:].cpu().numpy(), precision=3,floatmode='fixed')}"
        # )
        # print(f"pred {predected_tokens}")
        # print(f"gt: {labels[:, -num_actions:].cpu().numpy()}")

        return prediction_actions

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> np.ndarray:
        model_actions = self.get_model_trajactories(batch)

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
        images = images[[clip_len - 1]]
        camera = self.get_camera(batch)
        metadata = self.prepare_metadata(batch, clip_len - 1)

        # Ground-truth trajactories

        gt_points_3d: Float[Tensor, "f n 3"] = self.get_trajectory_3d_points(
            steps=self.gt_steps,
            time_interval=self.gt_time_interval,
            **metadata,
        )

        gt_points_2d: Float[Tensor, "f n 2"] = rearrange(
            camera.project(rearrange(gt_points_3d, "f n d -> (f n) 1 1 d")),  # type: ignore
            "(f n) 1 1 d -> f n (1 1 d)",
            f=1,
        )

        visualizations: np.ndarray = np.ascontiguousarray(
            (images * 255).int().cpu().numpy().astype(np.uint8)
        )
        draw_trajectory(visualizations, gt_points_2d)

        # Model prediction trajactories
        metadata["gas_pedal_norm"][:] = model_actions[0]
        metadata["brake_pedal_norm"][:] = model_actions[1]
        metadata["steering_norm"][:] = model_actions[2]

        pred_points_3d: Float[Tensor, "f n 3"] = self.get_trajectory_3d_points(
            steps=self.gt_steps,
            time_interval=self.gt_time_interval,
            **metadata,
        )

        pred_points_2d: Float[Tensor, "f n 2"] = rearrange(
            camera.project(rearrange(pred_points_3d, "f n d -> (f n) 1 1 d")),  # type: ignore
            "(f n) 1 1 d -> f n (1 1 d)",
            f=1,
        )

        draw_trajectory(visualizations, pred_points_2d, point_color=[0, 255, 0])

        return visualizations

    def prepare_metadata(self, batch, clip_index):
        clips = mit.one(batch["clips"].values())
        meta = clips["meta"]
        in_to_out = {
            "VehicleMotion_speed": "speed",  # km / h
            "VehicleMotion_steering_angle_normalized": "steering_norm",
            "VehicleMotion_gas_pedal_normalized": "gas_pedal_norm",
            "VehicleMotion_brake_pedal_normalized": "brake_pedal_norm",
            "VehicleMotion_acceleration_x": "acceleration_x",  # m / s2
            "VehicleMotion_acceleration_y": "acceleration_y",  # m / s2
            "VehicleMotion_acceleration_z": "acceleration_z",
        }
        out = {
            out_key: meta[in_key][:, [clip_index]].to(self.device)
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
        acceleration_y,
        acceleration_z,
        steering_norm,
        gas_pedal_norm,
        brake_pedal_norm,
    ) -> Float[Tensor, "f n 3"]:
        clips, *_ = speed.shape
        points_3d: List[List[Tuple[float, float, float]]] = []

        for i in range(clips):
            start_point = TrajectoryPoint(
                x=0.0,
                y=1.5,
                z=1e-6,
                phi=math.radians(0.0),
                v=speed[i].item(),
                t=0.0,
            )
            curr_points = self.calculate_trajectory(
                last_elem=start_point,
                steps=steps,
                time_interval=time_interval,
                acceleration_x=acceleration_x[i].item(),
                acceleration_y=acceleration_y[i].item(),
                acceleration_z=acceleration_z[i].item(),
                gas_norm=gas_pedal_norm[i].item(),
                brake_norm=brake_pedal_norm[i].item(),
                steering_wheel_norm=steering_norm[i].item(),
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
        acceleration_y: float,
        acceleration_z: float,
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
        for step in range(steps):
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
            vis = cv2.circle(vis, point, point_radius, point_color, point_thickness)  # type: ignore
        visualizations[i] = vis
        visualizations[i] = vis
