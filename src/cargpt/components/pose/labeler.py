from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

# import utm
from torch import Tensor

from .types import Pose


class PoseLabeler(ABC):
    @abstractmethod
    def __call__(self, meta: Mapping[str, Tensor]) -> Pose: ...


class SpeedPoseLabeler(PoseLabeler):
    def __call__(self, episode) -> Pose:
        speed = episode.inputs["continuous", "speed"][:, :-1, ...]
        dt = episode.inputs["meta", "timestamp"].diff(dim=-2)
        z = ((speed * dt).to(torch.float32) * 1e-3 / 3600).to(
            torch.float32
        )  # time is in microseconds, speed in kph
        x = y = theta_x = theta_y = theta_z = torch.full_like(z, torch.nan)
        return torch.cat([x, y, z, theta_x, theta_y, theta_z], axis=-1)
