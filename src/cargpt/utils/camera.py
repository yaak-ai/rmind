import json
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, final

import torch
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict, tensorclass
from torch import Tensor
import os

type Grid = Float[Tensor, "h w 2"]
type GridBatch = Float[Tensor, "*b h w 2"]
type Depth = Float[Tensor, "*b h w 1"]
type Points = Float[Tensor, "*b h w 3"]


class Camera(ABC, torch.nn.Module):
    @abstractmethod
    def project(self, points: Points) -> GridBatch: ...

    @abstractmethod
    def unproject(self, grid: Grid, depth: Depth) -> Points: ...

    @final
    @classmethod
    def from_params(cls, params: Any) -> "Camera":
        if camera_cls := {
            CameraParametersOpenCVFisheye: PinholeCamera,
            CameraParametersEnhancedUnified: EUCMCamera,
        }.get(type(params)):
            return camera_cls(**params.to_dict())

        raise NotImplementedError(f"params not supported: {params}")

    @final
    @classmethod
    def from_camera_config(cls, config: str):
        params = CameraParameters.from_file(config)
        return cls.from_params(params)


class PinholeCamera(Camera):
    """Pinhole Camera Model"""

    def __init__(
        self,
        *,
        fx: Float[Tensor, "b"],
        fy: Float[Tensor, "b"],
        cx: Float[Tensor, "b"],
        cy: Float[Tensor, "b"],
    ):
        super().__init__()

        self.f_x = torch.nn.Parameter(fx)
        self.f_y = torch.nn.Parameter(fy)
        self.c_x = torch.nn.Parameter(cx)
        self.c_y = torch.nn.Parameter(cy)

        self.requires_grad_(False)
        self.train(False)

    @property
    def batch_size(self):
        return self.f_x.shape[0]

    @cached_property
    def f_xy(self) -> Float[Tensor, "b 2"]:
        return torch.stack([self.f_x, self.f_y], dim=-1).contiguous()

    @cached_property
    def c_xy(self) -> Float[Tensor, "b 2"]:
        return torch.stack([self.c_x, self.c_y], dim=-1).contiguous()

    def project(self, points: Points) -> GridBatch:
        param_shape = (self.batch_size, *(1 for _ in range(points.dim() - 2)), -1)
        f_xy = self.f_xy.view(*param_shape)
        c_xy = self.c_xy.view(*param_shape)

        xy, z = points.split([2, 1], dim=-1)
        z = z.clamp(min=1e-6)
        grid = f_xy * (xy / z) + c_xy

        return grid

    def unproject(self, grid: Grid, depth: Depth) -> Points:
        xyz = self._unproject(grid)
        z = xyz[..., 2].unsqueeze(-1).clamp_min(1e-6)
        broadcast_shape = (
            xyz.shape[0],
            *(1 for _ in range(depth.dim() - xyz.dim())),
            *xyz.shape[1:],
        )
        points = (xyz / z).view(broadcast_shape) * depth

        return points

    def _unproject(self, grid: Grid) -> Points:
        param_shape = (self.batch_size, 1, 1, -1)
        f_xy = self.f_xy.view(*param_shape)
        c_xy = self.c_xy.view(*param_shape)

        m_xy = (grid - c_xy) / f_xy
        m_z = torch.ones(*m_xy.shape[:-1], 1, device=m_xy.device)
        m_xyz = torch.cat([m_xy, m_z], dim=-1)
        xyz = F.normalize(m_xyz, dim=-1, p=2, eps=1e-6)

        return xyz


class EUCMCamera(Camera):
    """Enhanced Unified Camera Model

    see https://arxiv.org/pdf/1807.08957.pdf for (un-)projection equations
    """

    def __init__(
        self,
        *,
        fx: Float[Tensor, "b"],
        fy: Float[Tensor, "b"],
        cx: Float[Tensor, "b"],
        cy: Float[Tensor, "b"],
        alpha: Float[Tensor, "b"],
        beta: Float[Tensor, "b"],
    ):
        super().__init__()

        self.f_x = torch.nn.Parameter(fx)
        self.f_y = torch.nn.Parameter(fy)
        self.c_x = torch.nn.Parameter(cx)
        self.c_y = torch.nn.Parameter(cy)
        self.alpha = torch.nn.Parameter(alpha)
        self.beta = torch.nn.Parameter(beta)

        self.requires_grad_(False)
        self.train(False)

    @property
    def batch_size(self):
        return self.f_x.shape[0]

    @cached_property
    def f_xy(self) -> Float[Tensor, "b 2"]:
        return torch.stack([self.f_x, self.f_y], dim=-1).contiguous()

    @cached_property
    def c_xy(self) -> Float[Tensor, "b 2"]:
        return torch.stack([self.c_x, self.c_y], dim=-1).contiguous()

    def project(self, points: Points) -> GridBatch:
        param_shape = (self.batch_size, *(1 for _ in range(points.dim() - 2)), -1)
        f_xy = self.f_xy.view(*param_shape)
        c_xy = self.c_xy.view(*param_shape)
        alpha = self.alpha.view(*param_shape)
        beta = self.beta.view(*param_shape)

        xy, z = points.split([2, 1], dim=-1)
        d = (beta * xy.square().sum(dim=-1, keepdim=True) + z.square()).sqrt()
        denom = (alpha * d + (1 - alpha) * z).clamp_min(1e-6)
        grid = f_xy * (xy / denom) + c_xy

        return grid

    def unproject(self, grid: Grid, depth: Depth) -> Points:
        xyz = self._unproject(grid)
        z = xyz[..., 2].unsqueeze(-1).clamp_min(1e-6)
        broadcast_shape = (
            xyz.shape[0],
            *(1 for _ in range(depth.dim() - xyz.dim())),
            *xyz.shape[1:],
        )
        points = (xyz / z).view(broadcast_shape) * depth

        return points

    @lru_cache
    def _unproject(self, grid: Grid) -> Points:
        param_shape = (self.batch_size, 1, 1, -1)
        f_xy = self.f_xy.view(*param_shape)
        c_xy = self.c_xy.view(*param_shape)
        alpha = self.alpha.view(*param_shape)
        beta = self.beta.view(*param_shape)

        m_xy = (grid - c_xy) / f_xy
        r_sq = m_xy.square().sum(dim=-1, keepdim=True)
        m_z = (1 - beta * alpha * alpha * r_sq) / (
            alpha * (1 - (2 * alpha - 1) * beta * r_sq).sqrt() + (1 - alpha)
        )

        m_xyz = torch.cat([m_xy, m_z], dim=-1)
        xyz = F.normalize(m_xyz, dim=-1, p=2, eps=1e-6)

        return xyz


class CameraName(Enum):
    CAM_FRONT_CENTER = auto()
    CAM_FRONT_LEFT = auto()
    CAM_FRONT_RIGHT = auto()
    CAM_LEFT_FORWARD = auto()
    CAM_LEFT_BACKWARD = auto()
    CAM_RIGHT_FORWARD = auto()
    CAM_RIGHT_BACKWARD = auto()
    CAM_REAR = auto()


@tensorclass  # pyright: ignore[reportArgumentType]
class CameraIntrinsics:
    fx: Tensor
    fy: Tensor
    cx: Tensor
    cy: Tensor


class CameraParameters(ABC):
    @abstractmethod
    def __init__(self) -> None: ...

    @final
    @classmethod
    def from_file(cls, path: str | Path) -> "CameraParameters":
        with Path(path).open("rb") as f:
            data = json.load(f)
            model_data = data["Calibration"]["cameras"][0]["model"]
            model_name = model_data["polymorphic_name"].split("::")[-1]
            params = model_data["ptr_wrapper"]["data"]["parameters"]
            intrinsics = {
                "fx": params["f"]["val"],
                "fy": params["f"]["val"],
                "cx": params["cx"]["val"],
                "cy": params["cy"]["val"],
            }

            if model_name == "CameraModelOpenCVFisheye":
                params = TensorDict(intrinsics, batch_size=[])
                return CameraParametersOpenCVFisheye.(params)

            if model_name == "CameraModelEnhancedUnified":
                params = TensorDict(
                    intrinsics
                    | {"alpha": params["alpha"]["val"], "beta": params["beta"]["val"]},
                    batch_size=[],
                )

                return CameraParametersEnhancedUnified.from_tensordict(params)

            msg = f"unsupported camera model: {model_name}"
            raise NotImplementedError(msg)


@tensorclass  # pyright: ignore[reportUntypedClassDecorator]
class CameraParametersOpenCVFisheye(CameraParameters, CameraIntrinsics): ...  # pyright: ignore[reportGeneralTypeIssues, reportUntypedBaseClass]


@tensorclass  # pyright: ignore[reportUntypedClassDecorator]
class CameraParametersEnhancedUnified(CameraParameters, CameraIntrinsics):  # pyright: ignore[reportGeneralTypeIssues, reportUntypedBaseClass]
    alpha: Tensor
    beta: Tensor
