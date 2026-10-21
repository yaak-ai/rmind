"""Native, horizon-independent sinusoidal time embedding construction."""

import math

import torch
from torch import Tensor, nn

MIN_CHANNEL_DIM = 2


def native_time_frequencies(
    channel_dim: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Tensor:
    """Evaluate the analytical frequencies in CPU FP64, then round once to FP32.

    Construction is independent of the active accelerator, action horizon,
    training dataset and external model artifacts. The single-frequency case
    uses the minimum period, matching the first endpoint of the general case.

    Raises:
        TypeError: The channel dimension is not an integer.
        ValueError: Dimensions, periods or resulting frequencies are invalid.
    """
    if isinstance(channel_dim, bool) or not isinstance(channel_dim, int):
        msg = "channel_dim must be a positive even integer"
        raise TypeError(msg)
    if channel_dim < MIN_CHANNEL_DIM or channel_dim % 2:
        msg = "channel_dim must be a positive even integer"
        raise ValueError(msg)
    if not (
        math.isfinite(min_period)
        and math.isfinite(max_period)
        and 0 < min_period <= max_period
    ):
        msg = "periods must be finite and 0 < min_period <= max_period"
        raise ValueError(msg)
    count = channel_dim // 2
    fraction = torch.arange(count, dtype=torch.float64, device="cpu") / max(
        count - 1, 1
    )
    period = min_period * (max_period / min_period) ** fraction
    frequency = (2 * math.pi / period).to(torch.float32)
    if not (torch.isfinite(frequency).all() and (frequency > 0).all()):
        msg = "frequencies must be representable as finite positive FP32"
        raise ValueError(msg)
    return frequency


class NativeTimeEmbedding(nn.Module):
    """Sin/cos embedding for arbitrary leading time axes, with derived state."""

    frequency: Tensor

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
