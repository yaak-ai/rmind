"""Action-conditioned latent world-model predictor for `PatchPolicy`'s auxiliary
WM loss (docs/wm_aux_loss_plan.md, after AD-E2E-JEPA arXiv:2609.34085).

`(pooled trunk patch tokens of frame t, ego relative poses P_1..P_k) ->
pooled frozen-encoder patch features of frame t+k`. Training-time only: it
reads the trunk's outputs and never feeds anything back into the policy's
forward/serving path.
"""

import math
from typing import override

import torch
from einops import rearrange
from pydantic import validate_call
from torch import Tensor, nn
from torch.nn import functional as F


def pool_patch_grid(x: Tensor, *, num_cameras: int, pool_size: int) -> Tensor:
    """Average-pool per-camera square patch grids.

    `x` is `(..., cam * p, d)` with `p = side**2` patches per camera in
    row-major order (the ViT `(h w)` flattening); returns `(..., cam *
    pool_size**2, d)`, cameras kept in order. ONE helper for the trunk side and
    the target side, so the two are pooled onto identical cells.

    Raises:
        ValueError: if the patch axis is not `num_cameras` square grids.
    """
    *batch, num_tokens, dim = x.shape
    p, rem = divmod(num_tokens, num_cameras)
    side = math.isqrt(p)
    if rem or side * side != p:
        msg = (
            f"cannot split {num_tokens} patch tokens into {num_cameras} square "
            "per-camera grids"
        )
        raise ValueError(msg)

    grid = rearrange(
        x.reshape(-1, num_cameras, side, side, dim), "n cam h w d -> (n cam) d h w"
    )
    pooled = F.adaptive_avg_pool2d(grid, pool_size)
    return rearrange(pooled, "(n cam) d h w -> n (cam h w) d", cam=num_cameras).reshape(
        *batch, num_cameras * pool_size * pool_size, dim
    )


class WorldModelPredictor(nn.Module):
    """Small bidirectional transformer over

        [ spatial_0 .. spatial_{S-1}, horizon, pose_1 .. pose_k ]

    with `S = num_cameras * pool_size**2` pooled trunk tokens (learned
    positional embedding per cell, which also carries camera identity), one
    learned horizon token for `k`, and one token per relative pose (MLP on
    `(x, y) * position_scale, sin theta, cos theta` plus a learned pose-index embedding). The
    spatial positions are read out through `Linear(dim_model, image_dim)`.

    Bidirectional is fine: the whole pose sequence is a given input and no
    future image is visible to it. No dropout or any other stochastic op --
    `PatchPolicy`'s `wm_weight=0` parity gate needs the forward RNG stream
    unchanged by this module.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int,
        image_dim: int,
        num_cameras: int = 1,
        pool_size: int = 4,
        max_horizon: int = 5,
        position_scale: float = 0.1,
        num_layers: int = 2,
        num_heads: int = 8,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()

        self.num_cameras = num_cameras
        self.pool_size = pool_size
        self.max_horizon = max_horizon
        # dead-reckoned (x, y) are meters (~0..40 at k<=5 clip steps); bring
        # them to O(1) before the pose MLP
        self.position_scale = position_scale
        self.num_spatial_tokens = num_cameras * pool_size * pool_size

        self.spatial_embedding = nn.Parameter(
            torch.empty(self.num_spatial_tokens, dim_model)
        )
        nn.init.trunc_normal_(
            self.spatial_embedding, mean=0.0, std=0.02, a=-0.04, b=0.04
        )
        self.horizon_embedding = nn.Embedding(max_horizon + 1, dim_model)
        self.pose_index_embedding = nn.Embedding(max_horizon, dim_model)
        self.pose_mlp = nn.Sequential(
            nn.Linear(4, dim_model), nn.GELU(), nn.Linear(dim_model, dim_model)
        )
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=hidden_layer_multiplier * dim_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.head = nn.Linear(dim_model, image_dim)

    @override
    def forward(self, tokens: Tensor, relative_poses: Tensor) -> Tensor:
        """`tokens` `(N, S, dim_model)`, `relative_poses` `(N, k, 3)` cumulative
        ego-centric `(x, y, theta)` at `t+1 .. t+k` relative to the anchor
        `t` -> predicted pooled target features `(N, S, image_dim)`.

        Raises:
            ValueError: on a token count other than `num_spatial_tokens` or a
                horizon outside `1 .. max_horizon`.
        """
        n, s, _ = tokens.shape
        k = relative_poses.shape[-2]
        if s != self.num_spatial_tokens:
            msg = f"expected {self.num_spatial_tokens} spatial tokens, got {s}"
            raise ValueError(msg)
        if not 1 <= k <= self.max_horizon:
            msg = f"horizon {k} outside 1..{self.max_horizon}"
            raise ValueError(msg)

        theta = relative_poses[..., 2]
        pose_features = torch.stack(
            [
                relative_poses[..., 0] * self.position_scale,
                relative_poses[..., 1] * self.position_scale,
                theta.sin(),
                theta.cos(),
            ],
            dim=-1,
        ).to(tokens.dtype)
        index = torch.arange(k, device=tokens.device)
        pose_tokens = self.pose_mlp(pose_features) + self.pose_index_embedding(index)
        horizon_token = self.horizon_embedding(
            torch.full((n, 1), k, dtype=torch.long, device=tokens.device)
        )

        x = torch.cat(
            [tokens + self.spatial_embedding, horizon_token, pose_tokens], dim=1
        )
        x = self.layers(x)
        return self.head(self.norm(x[:, :s]))
