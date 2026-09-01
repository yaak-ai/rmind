"""A learned MLP standing in for the frozen waypoints tokenizer's goal latent."""

from typing import Final, final, override

import torch
from pydantic import validate_call
from torch import Tensor, nn
from torch.nn import Module


@final
class _GoalLatentSpec:
    """The slice of `ResidualVQ`'s surface that `PatchPolicy` actually reads.

    `PatchPolicy` sizes the patch projection from `goal_encoder.quantizer.dim`
    (`patch_dim = patch_projection.in_features - goal_dim`), so an MLP substitute has
    to answer that. `codebook_size`/`num_quantizers` are deliberately absent rather
    than faked: they are read ONLY by `_init_fusion_norm`'s Monte-Carlo fallback, which
    estimates the goal stream's RMS by sampling random codes. That is meaningless for an
    MLP, so hitting it should raise rather than silently calibrate against noise --
    pass `fusion_goal_rms` explicitly instead (see `MlpGoalEncoder` below).
    """

    def __init__(self, dim: int) -> None:
        self.dim: Final[int] = dim


@final
class MlpGoalEncoder(Module):
    """Learned goal embedding with the `WaypointsTokenizer.encode` contract.

    Drop-in for `PatchPolicy.goal_encoder`: `encode((*batch, N, 2)) -> (*batch, dim)`,
    same reshape, and a `.quantizer.dim` for the patch-projection arithmetic.

    WHY THE OUTPUT IS NORMALIZED. The goal is not a token -- `_frame_tokens`
    concatenates it onto EVERY patch before `patch_projection`, so with three cameras
    one goal vector is broadcast across 3 x 144 = 432 patch tokens per frame. Its scale
    therefore lands as a shared additive component on the whole frame block after
    projection, and if it is loud it swamps the part of each patch that distinguishes it
    from its neighbours. `fusion_norm` balances the two streams by scaling the goal by
    `1 / RMS(z_q)`, which for a real tokenizer is a bounded, measurable quantity (the
    car measures 0.077, and notes the uniform-random-code estimate overstates it 1.86x
    because codebook usage is non-uniform). An unconstrained MLP has no such bound and
    its output scale is free to GROW during training, so a run could start balanced and
    slowly drown the cameras -- a failure that looks like the model ignoring vision.

    The final `LayerNorm` fixes element-RMS at 1 by construction, so the arm can declare
    `fusion_goal_rms: 1.0` and the goal gain becomes exact rather than estimated. It
    also makes an MLP arm and a tokenizer arm directly comparable: both goal streams
    enter at a known matched scale, so a difference between them is about the
    representation and not about fusion gain.

    Watch `quality/token_norm/*/out/patch/<camera>` (logged only when
    `len(cameras) > 1`) to confirm empirically: if the goal is too loud the per-camera
    patch norms converge on each other and lose spread.
    """

    @validate_call
    def __init__(
        self,
        *,
        dim: int,
        num_waypoints: int = 1,
        waypoint_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self._input_dim: Final[int] = num_waypoints * waypoint_dim
        self.quantizer: Final[_GoalLatentSpec] = _GoalLatentSpec(dim)
        self.encoder: Module = nn.Sequential(
            nn.Linear(self._input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            # element-RMS 1 by construction -- see the class docstring
            nn.LayerNorm(dim),
        )

    def encode(self, waypoints: Tensor) -> Tensor:
        """Goal latent for `(*batch, N, waypoint_dim)` -> `(*batch, dim)`.

        Mirrors `WaypointsTokenizer.encode`'s reshape exactly so the two are
        interchangeable under `PatchPolicy`.
        """
        *batch, _num_waypoints, _waypoint_dim = waypoints.shape
        w = waypoints.reshape(-1, self._input_dim)
        z = self.encoder(w)

        return z.reshape(*batch, z.shape[-1])

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        return self.encode(waypoints)
