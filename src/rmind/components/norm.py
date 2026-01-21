from typing import final, override

import numpy as np
import torch
from pydantic import validate_call
from torch import Tensor
from torch.nn import Module
from torchaudio.functional import mu_law_decoding
from torchaudio.transforms import MuLawEncoding as _MuLawEncoding

from .base import Invertible


class Clamp(Module):
    min_value: Tensor
    max_value: Tensor

    @validate_call
    def __init__(self, *, min_value: float, max_value: float) -> None:
        super().__init__()

        self.register_buffer("min_value", torch.tensor(min_value))
        self.register_buffer("max_value", torch.tensor(max_value))

    @override
    def forward(self, input: Tensor) -> Tensor:
        return torch.clamp(input, min=self.min_value, max=self.max_value)

    @override
    def extra_repr(self) -> str:
        return f"{[self.min_value.item(), self.max_value.item()]}"


class Scaler(Module, Invertible):
    in_range: Tensor
    out_range: Tensor

    @validate_call
    def __init__(
        self, *, in_range: tuple[float, float], out_range: tuple[float, float]
    ) -> None:
        super().__init__()

        self.register_buffer("in_range", torch.tensor(in_range))
        self.register_buffer("out_range", torch.tensor(out_range))

    @override
    def forward(self, input: Tensor) -> Tensor:
        input_min, input_max = self.in_range

        if torch.compiler.is_exporting():  # ty:ignore[possibly-missing-attribute]
            input = torch.clamp(input, input_min, input_max)
        elif input.min() < input_min or input.max() > input_max:
            msg = "input out of range"
            raise ValueError(msg)

        out_min, out_max = self.out_range
        out_std = (input - input_min) / (input_max - input_min)

        return out_std * (out_max - out_min) + out_min

    @override
    def invert(self, input: Tensor) -> Tensor:
        input_min, input_max = self.out_range
        if input.min() < input_min or input.max() > input_max:
            msg = "input out of range"
            raise ValueError(msg)

        out_min, out_max = self.in_range
        out_std = (input - input_min) / (input_max - input_min)

        return out_std * (out_max - out_min) + out_min

    @override
    def extra_repr(self) -> str:
        return f"{self.in_range.tolist()} -> {self.out_range.tolist()}"


class UniformBinner(Module, Invertible):
    range: Tensor
    bins: Tensor

    @validate_call
    def __init__(self, *, range: tuple[float, float], bins: int) -> None:
        super().__init__()

        self.register_buffer("range", torch.tensor(range))
        self.register_buffer("bins", torch.tensor(bins))

    @override
    def forward(self, input: Tensor) -> Tensor:
        input_min, input_max = self.range

        if torch.compiler.is_exporting():  # ty:ignore[possibly-missing-attribute]
            input = torch.clamp(input, input_min, input_max)
        elif input.min() < input_min or input.max() > input_max:
            msg = "input out of range"
            raise ValueError(msg)

        x_norm = (input - input_min) / (input_max - input_min)

        return (x_norm * self.bins).to(torch.long).clamp(max=self.bins - 1)

    @override
    def invert(self, input: Tensor) -> Tensor:
        input_min, input_max = 0, self.bins - 1

        if input.min() < input_min or input.max() > input_max:
            msg = "input out of range"
            raise ValueError(msg)

        start, end = self.range
        width = (end - start) / self.bins

        return start + (input + 0.5) * width

    @override
    def extra_repr(self) -> str:
        return f"{self.range.tolist()} -> {self.bins.item()}"


class MuLawEncoding(_MuLawEncoding, Invertible):
    @override
    def invert(self, input: Tensor) -> Tensor:
        return mu_law_decoding(input, self.quantization_channels)


@final
class Normalize(Module):
    def __init__(self, p: int = 2, dim: int = -1) -> None:
        super().__init__()
        self.p: int = p
        self.dim: int = dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.normalize(input, p=self.p, dim=self.dim)


@final
class ScaleByVectorDimensionality(Module):
    def __init__(self, dim: int = 512, up: bool = False, down: bool = False) -> None:  # noqa: FBT001,FBT002
        super().__init__()
        self.up = up
        self.down = down
        self.factor = np.sqrt(dim)

    @override
    def forward(self, input: Tensor) -> Tensor:
        if self.up:
            return input * self.factor
        return input / self.factor
