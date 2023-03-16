from typing import Annotated, Sequence

import torch
from jaxtyping import Float
from torch import Tensor


class MinMaxScaler(torch.nn.Module):
    def __init__(
        self,
        *,
        in_range: Annotated[Sequence[float], 2],
        out_range: Annotated[Sequence[float], 2],
    ):
        super().__init__()

        self.register_buffer("in_range", torch.tensor(in_range))
        self.register_buffer("out_range", torch.tensor(out_range))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x_min, x_max = self.in_range  # type: ignore
        if x.min() < x_min or x.max() > x_max:
            raise ValueError("input out of range")

        new_min, new_max = self.out_range  # type: ignore

        x_std = (x - x_min) / (x_max - x_min)
        x_scaled = x_std * (new_max - new_min) + new_min

        return x_scaled
