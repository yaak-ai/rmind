"""Map-derived context conditioning components (Traffic rules & environment
awareness, Arm M).

Shared max-speed vocabulary -- SEMANTIC German speed classes, not uniform
km/h bins (must match the GT-sidecar contract in `caches/map_gt` and any
serving-side tokenization):

- id 0: UNKNOWN -- NaN / missing input
- id 1: UNLIMITED -- explicitly no legal limit (German autobahn
  maxspeed=none), encoded as a negative sentinel (-1.0) on the data side
- id 2: WALK -- Schrittgeschwindigkeit (verkehrsberuhigter Bereich /
  parking lot). Any input in [0, 7] km/h selects it; the data-side join
  translates OSM ``maxspeed=walk`` (and ``DE:living_street``) to 7.0 km/h,
  which lands here. As an inference override, any value <= 7 (e.g. 5) works.
- ids 3..12: the German signposted classes
  [10, 20, 30, 50, 60, 70, 80, 100, 120, 130] km/h

Snap rule for other finite values > 7 km/h: NEAREST class; exact ties snap
DOWN -- conservative for compliance (40 -> 30, 90 -> 80, 110 -> 100);
values > 130 clamp to 130. The tokenizer is float-only: string OSM values
must be translated to floats on the data side (see
`rmind.scripts.map_gt.overpass.parse_maxspeed_value`).
"""

from typing import final, override

import torch
from torch import Tensor, nn

MAX_SPEED_VOCAB_KMH: tuple[float, ...] = (
    10.0,
    20.0,
    30.0,
    50.0,
    60.0,
    70.0,
    80.0,
    100.0,
    120.0,
    130.0,
)
MAX_SPEED_UNKNOWN_ID = 0
MAX_SPEED_UNLIMITED_ID = 1
MAX_SPEED_WALK_ID = 2
MAX_SPEED_NUM_SPECIAL = 3  # UNKNOWN, UNLIMITED, WALK
MAX_SPEED_WALK_MAX_KMH = 7.0
MAX_SPEED_VOCAB_SIZE = len(MAX_SPEED_VOCAB_KMH) + MAX_SPEED_NUM_SPECIAL  # 13


@final
class MaxSpeedTokenizer(nn.Module):
    """Float km/h -> token ids per the shared 13-token max-speed vocabulary.

    NaN -> UNKNOWN (0); any negative value (the -1.0 "explicitly unlimited"
    sentinel) -> UNLIMITED (1); values in [0, 7] km/h -> WALK (2,
    Schrittgeschwindigkeit); everything else -> 3 + index of the nearest
    vocabulary speed, exact ties snapping DOWN (40 -> 30) and values > 130
    clamping to 130. Shape-preserving; ONNX-traceable (isnan/where/argmin --
    argmin returns the FIRST minimal index, which on the ascending vocabulary
    is exactly the tie-down rule).
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
        nearest = (
            (x.unsqueeze(-1) - self.values).abs().argmin(dim=-1)
            + MAX_SPEED_NUM_SPECIAL
        )
        ids = torch.where(
            x <= MAX_SPEED_WALK_MAX_KMH,
            torch.full_like(nearest, MAX_SPEED_WALK_ID),
            nearest,
        )
        ids = torch.where(
            x < 0.0, torch.full_like(nearest, MAX_SPEED_UNLIMITED_ID), ids
        )
        return torch.where(
            torch.isnan(x), torch.full_like(nearest, MAX_SPEED_UNKNOWN_ID), ids
        )
