"""Map-derived context conditioning components (Traffic rules & environment
awareness, Arm M).

Shared max-speed vocabulary (must match the GT-sidecar contract in
`caches/map_gt` and any serving-side tokenization):

- id 0: UNKNOWN -- NaN / missing input
- id 1: UNLIMITED -- explicitly no legal limit (German autobahn), encoded as
  a negative sentinel (-1.0) on the data side
- ids 2..15: nearest of [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110,
  120, 130] km/h
"""

from typing import final, override

import torch
from torch import Tensor, nn

MAX_SPEED_VOCAB_KMH: tuple[float, ...] = (
    5.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    130.0,
)
MAX_SPEED_UNKNOWN_ID = 0
MAX_SPEED_UNLIMITED_ID = 1
MAX_SPEED_VOCAB_SIZE = len(MAX_SPEED_VOCAB_KMH) + 2  # 16


@final
class MaxSpeedTokenizer(nn.Module):
    """Float km/h -> token ids per the shared 16-token max-speed vocabulary.

    NaN -> UNKNOWN (0); any negative value (the -1.0 "explicitly unlimited"
    sentinel) -> UNLIMITED (1); everything else -> 2 + index of the nearest
    vocabulary speed. Shape-preserving; ONNX-traceable (isnan/where/argmin).
    """

    values: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "values", torch.tensor(MAX_SPEED_VOCAB_KMH, dtype=torch.float32)
        )

    @property
    def vocab_size(self) -> int:
        return MAX_SPEED_VOCAB_SIZE

    @override
    def forward(self, input: Tensor) -> Tensor:
        x = input.float()
        nearest = (x.unsqueeze(-1) - self.values).abs().argmin(dim=-1) + 2
        ids = torch.where(
            x < 0.0, torch.full_like(nearest, MAX_SPEED_UNLIMITED_ID), nearest
        )
        return torch.where(
            torch.isnan(x), torch.full_like(nearest, MAX_SPEED_UNKNOWN_ID), ids
        )
