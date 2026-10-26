import math
from collections.abc import Callable
from typing import Annotated, override

import torch
from pydantic import Field, validate_call
from torch import Tensor, nn
from torch.nn import Module
from torchvision.ops import MLP


def sinusoidal_time_embedding(time: Tensor, dim: int) -> Tensor:
    """`(*b,)` flow time in `[0, 1]` -> `(*b, dim)` sin/cos features. `time`
    is scaled by 1000 first so `[0, 1]` spans the frequency range a diffusion
    timestep embedding expects."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / half
    )
    angles = time.float().unsqueeze(-1) * 1000.0 * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FlowMatchingTrajectoryHead(Module):
    """Rectified-flow (linear-interpolation conditional flow matching, pi0
    style) trajectory head: a drop-in replacement for the multi-hypothesis
    `trajectory_head` MLP in `PatchPolicy`.

    Instead of regressing a fixed number of hypotheses under a
    winner-takes-all loss, a velocity MLP `v(x_t, t, features)` is trained
    on the whole `(num_poses, 3)` trajectory with
    `x_t = (1 - t) x_0 + t x_1`, `x_0 ~ N(0, I)`, `t ~ U(0, 1)` and loss
    `MSE(v, x_1 - x_0)`. Sampling integrates Euler from `t = 0` to `1` in
    `num_inference_steps`, so any number of trajectories can be drawn from
    one readout feature; `noise=False` starts from `x_0 = 0` for a single
    deterministic trajectory.

    The flow runs in a normalized space: per-(pose, channel) mean/std of the
    `(x, y, theta)` target, accumulated over the first
    `normalizer_warmup_steps` training batches and then frozen. They are
    buffers, so they ship with the checkpoint. Meters and radians differ by
    orders of magnitude and `x` grows with the pose index, so an unnormalized
    flow would spend its capacity almost entirely on far-pose `x`.
    """

    stats_steps: Tensor
    stats_count: Tensor
    stats_sum: Tensor
    stats_sum_sq: Tensor

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        in_channels: int,
        num_poses: int,
        hidden_channels: tuple[int, ...] = (1024, 1024),
        # torchvision.ops.MLP's hidden-layer norm/activation, same knobs as
        # the regression head (LayerNorm + GELU there since the R2 fix)
        norm_layer: Callable[..., Module] | None = nn.LayerNorm,
        activation_layer: Callable[..., Module] = nn.GELU,
        time_embedding_dim: Annotated[int, Field(gt=0, multiple_of=2)] = 64,
        num_inference_steps: Annotated[int, Field(gt=0)] = 10,
        normalizer_warmup_steps: Annotated[int, Field(ge=0)] = 1000,
        min_std: Annotated[float, Field(gt=0.0)] = 1e-3,
    ) -> None:
        super().__init__()

        self.num_poses = num_poses
        self.time_embedding_dim = time_embedding_dim
        self.num_inference_steps = num_inference_steps
        self.normalizer_warmup_steps = normalizer_warmup_steps
        self.min_std = min_std

        out_channels = num_poses * 3
        self.velocity_net = MLP(
            in_channels + out_channels + time_embedding_dim,
            [*hidden_channels, out_channels],
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # float64 running sums for the target normalizer (see class docstring)
        self.register_buffer("stats_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("stats_count", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "stats_sum", torch.zeros(num_poses, 3, dtype=torch.float64)
        )
        self.register_buffer(
            "stats_sum_sq", torch.zeros(num_poses, 3, dtype=torch.float64)
        )

    def _normalizer(self) -> tuple[Tensor, Tensor]:
        """`(num_poses, 3)` mean and std; identity until the first update."""
        if self.stats_count == 0:
            return (
                torch.zeros_like(self.stats_sum, dtype=torch.float32),
                torch.ones_like(self.stats_sum, dtype=torch.float32),
            )
        mean = self.stats_sum / self.stats_count
        var = (self.stats_sum_sq / self.stats_count - mean.square()).clamp_min(0.0)
        return mean.float(), var.sqrt().float().clamp_min(self.min_std)

    @torch.no_grad()
    def _update_normalizer(self, target: Tensor) -> None:
        flat = target.detach().reshape(-1, self.num_poses, 3).double()
        self.stats_count += flat.shape[0]
        self.stats_sum += flat.sum(dim=0)
        self.stats_sum_sq += flat.square().sum(dim=0)
        self.stats_steps += 1

    def velocity(self, features: Tensor, x: Tensor, time: Tensor) -> Tensor:
        """`features (*b, d)`, normalized `x (*b, num_poses, 3)`, `time (*b,)`
        -> velocity `(*b, num_poses, 3)`."""
        inputs = torch.cat(
            [
                features.float(),
                x.float().flatten(-2),
                sinusoidal_time_embedding(time, self.time_embedding_dim),
            ],
            dim=-1,
        )
        return self.velocity_net(inputs).unflatten(-1, (self.num_poses, 3)).float()

    def loss(self, features: Tensor, target: Tensor) -> Tensor:
        """Flow-matching MSE for `features (*b, d)` against the
        `(x, y, theta)` trajectory `target (*b, num_poses, 3)`."""
        if self.training and self.stats_steps < self.normalizer_warmup_steps:
            self._update_normalizer(target)

        mean, std = self._normalizer()
        x1 = (target.float() - mean) / std
        x0 = torch.randn_like(x1)
        time = torch.rand(x1.shape[:-2], device=x1.device)
        t = time[..., None, None]
        xt = (1.0 - t) * x0 + t * x1
        return (self.velocity(features, xt, time) - (x1 - x0)).square().mean()

    @torch.no_grad()
    def sample(
        self, features: Tensor, num_samples: int, *, noise: bool = True
    ) -> Tensor:
        """`features (*b, d)` -> `(*b, num_samples, num_poses, 3)` trajectories
        in target units. `noise=False` integrates from `x_0 = 0`, so every
        sample is the same deterministic trajectory."""
        features = features.unsqueeze(-2).expand(
            *features.shape[:-1], num_samples, features.shape[-1]
        )
        shape = (*features.shape[:-1], self.num_poses, 3)
        x = (
            torch.randn(shape, device=features.device)
            if noise
            else torch.zeros(shape, device=features.device)
        )
        steps = self.num_inference_steps
        for i in range(steps):
            time = torch.full(features.shape[:-1], i / steps, device=features.device)
            x += self.velocity(features, x, time) / steps

        mean, std = self._normalizer()
        return x * std + mean

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Deterministic (`x_0 = 0`) trajectory `(*b, num_poses, 3)`."""
        return self.sample(features, 1, noise=False).squeeze(-3)
