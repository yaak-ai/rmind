from collections.abc import Sequence
from typing import Annotated

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module
from typing_extensions import override


class Scaler(Module):
    def __init__(
        self,
        *,
        in_range: Annotated[Sequence[float], 2],
        out_range: Annotated[Sequence[float], 2],
    ) -> None:
        super().__init__()

        self.register_buffer("in_range", torch.tensor(in_range))
        self.register_buffer("out_range", torch.tensor(out_range))

    @override
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        x_min, x_max = self.in_range
        if x.min() < x_min or x.max() > x_max:
            msg = "input out of range"
            raise ValueError(msg)

        new_min, new_max = self.out_range

        x_std = (x - x_min) / (x_max - x_min)
        return x_std * (new_max - new_min) + new_min


class Clamp(Module):
    def __init__(self, *, min_value: float, max_value: float) -> None:
        super().__init__()

        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    @override
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.clamp(x, min=self.min_value, max=self.max_value)


class UniformBinner(Module):
    def __init__(self, in_range: Annotated[Sequence[float], 2], num_bins: int):
        super().__init__()

        self.in_range = in_range
        self.num_bins = num_bins

    @override
    def forward(self, x: Float[Tensor, "..."]) -> Int[Tensor, "..."]:
        x_min, x_max = self.in_range
        if x.min() < x_min or x.max() > x_max:
            msg = "input out of range"
            raise ValueError(msg)

        x_norm = (x - x_min) / (x_max - x_min)

        return (x_norm * self.num_bins).to(torch.long).clamp(max=self.num_bins - 1)

    @override
    def extra_repr(self) -> str:
        return f"{self.in_range} -> {self.num_bins}"
