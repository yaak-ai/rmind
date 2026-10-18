"""Native, horizon-independent sinusoidal time embedding construction."""

import math

import torch
from torch import Tensor, nn


def native_time_frequencies(
    channel_dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Tensor:
    """Evaluate the analytical frequencies in CPU FP64, then round once to FP32.

    Construction is independent of the active accelerator, action horizon,
    training dataset and external model artifacts. The single-frequency case
    uses the minimum period, matching the first endpoint of the general case.
    """
    if isinstance(channel_dim, bool) or not isinstance(channel_dim, int):
        raise ValueError("channel_dim must be a positive even integer")
    if channel_dim < 2 or channel_dim % 2:
        raise ValueError("channel_dim must be a positive even integer")
    if not (
        math.isfinite(min_period)
        and math.isfinite(max_period)
        and 0 < min_period <= max_period
    ):
        raise ValueError("periods must be finite and 0 < min_period <= max_period")
    count = channel_dim // 2
    fraction = torch.arange(count, dtype=torch.float64, device="cpu") / max(
        count - 1, 1
    )
    period = min_period * (max_period / min_period) ** fraction
    frequency = (2 * math.pi / period).to(torch.float32)
    if not (torch.isfinite(frequency).all() and (frequency > 0).all()):
        raise ValueError("frequencies must be representable as finite positive FP32")
    return frequency


class NativeTimeEmbedding(nn.Module):
    """Sin/cos embedding for arbitrary leading time axes, with derived state."""

    def __init__(
        self, channel_dim: int, min_period: float = 4e-3, max_period: float = 4.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "frequency", native_time_frequencies(channel_dim, min_period, max_period)
        )

    def forward(self, time: Tensor) -> Tensor:
        angle = time.to(torch.float32)[..., None] * self.frequency
        return torch.cat((angle.sin(), angle.cos()), dim=-1)
