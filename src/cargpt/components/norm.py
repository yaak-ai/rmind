from collections.abc import Sequence
from typing import Annotated

import torch
from jaxtyping import Float, Int, UInt
from torch import Tensor
from torch.nn import Module
from torchaudio.functional import mu_law_decoding
from torchaudio.transforms import MuLawEncoding as _MuLawEncoding
from typing_extensions import override

from .base import Invertible


class Clamp(Module):
    def __init__(self, *, min_value: float, max_value: float) -> None:
        super().__init__()

        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    @override
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.clamp(x, min=self.min_value, max=self.max_value)

    @override
    def extra_repr(self) -> str:
        return f"{[self.min_value.item(), self.max_value.item()]}"


class Scaler(Module, Invertible):
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
        in_min, in_max = self.in_range
        if x.min() < in_min or x.max() > in_max:
            msg = "input out of range"
            raise ValueError(msg)

        out_min, out_max = self.out_range
        out_std = (x - in_min) / (in_max - in_min)

        return out_std * (out_max - out_min) + out_min

    @override
    def invert(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        in_min, in_max = self.out_range
        if x.min() < in_min or x.max() > in_max:
            msg = "input out of range"
            raise ValueError(msg)

        out_min, out_max = self.in_range
        out_std = (x - in_min) / (in_max - in_min)

        return out_std * (out_max - out_min) + out_min

    @override
    def extra_repr(self) -> str:
        return f"{self.in_range.tolist()} -> {self.out_range.tolist()}"


class UniformBinner(Module, Invertible):
    def __init__(self, *, range: Annotated[Sequence[float], 2], bins: int):
        super().__init__()

        self.register_buffer("range", torch.tensor(range))
        self.register_buffer("bins", torch.tensor(bins))

    @override
    def forward(self, x: Float[Tensor, "..."]) -> Int[Tensor, "..."]:
        x_min, x_max = self.range
        if x.min() < x_min or x.max() > x_max:
            msg = "input out of range"
            raise ValueError(msg)

        x_norm = (x - x_min) / (x_max - x_min)

        return (x_norm * self.bins).to(torch.long).clamp(max=self.bins - 1)

    @override
    def invert(self, x: Int[Tensor, "..."]) -> Float[Tensor, "..."]:
        x_min, x_max = 0, self.bins - 1
        if x.min() < x_min or x.max() > x_max:
            msg = "input out of range"
            raise ValueError(msg)

        start, end = self.range
        width = (end - start) / self.bins

        return start + (x + 0.5) * width

    @override
    def extra_repr(self) -> str:
        return f"{self.range.tolist()} -> {self.bins.item()}"


class MuLawEncoding(_MuLawEncoding, Invertible):
    @override
    def invert(self, x_mu: UInt[Tensor, "..."]) -> Float[Tensor, "..."]:
        return mu_law_decoding(x_mu, self.quantization_channels)
