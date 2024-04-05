from functools import lru_cache
from math import prod
from typing import cast

import torch
from einops import rearrange, reduce
from jaxtyping import Float, Shaped
from torch import Tensor, nn
from typing_extensions import override


class DepthFeatureEncoder(nn.Module):
    def __init__(self, *, disp_net: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        encoder = cast(nn.Module, disp_net.encoder)
        self.encoder = encoder.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(
        self,
        frames: Float[Tensor, "*b c1 h1 w1"],
    ) -> Float[Tensor, "*b c2 h2 w2"]:
        *b, c, h, w = frames.shape
        frames = frames.view(prod(b), c, h, w)

        disp = self.encoder(frames)[2]

        *_, c, h, w = disp.shape
        return disp.view(*b, c, h, w)


class DepthEncoder(nn.Module):
    def __init__(self, *, disp_net: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        self.disp_net = disp_net.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(
        self,
        frames: Float[Tensor, "*b c _h _w"],
        *,
        camera,
    ) -> Float[Tensor, "*b h w 3"]:
        B, *_, H, W = frames.shape

        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        disp = self.disp_net(frames)[0]
        depth = 1 / disp
        depth = rearrange(depth, "(b t) 1 h w -> b t h w 1", b=B)

        # compute 3d points (equivalent to Eq 6 [https://arxiv.org/abs/2211.14710])
        grid = self._generate_2d_grid(height=H, width=W).to(depth)
        pts = camera.unproject(grid, depth)

        # normalize 3d points (Eq 7 [https://arxiv.org/abs/2211.14710])
        pts_min = reduce(pts, "b t h w c -> b t 1 1 c", "min")
        pts_max = reduce(pts, "b t h w c -> b t 1 1 c", "max")
        return (pts - pts_min) / (pts_max - pts_min)

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_2d_grid(cls, *, height: int, width: int) -> Shaped[Tensor, "h w 2"]:
        x_mesh, y_mesh = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )

        return rearrange([x_mesh, y_mesh], "t h w -> h w t")
