from math import prod

from jaxtyping import Shaped
from torch import Tensor, nn


class Sequential(nn.Sequential):
    def forward(self, *args, **kwargs):
        """Allows passing arbitrary args/kwargs to the first module"""
        first, *rest = self._modules.values()
        input = first(*args, **kwargs)

        for module in rest:
            input = module(input)

        return input


class AvgPool2d(nn.AvgPool2d):
    def forward(
        self, input: Shaped[Tensor, "*b c h1 w1"]
    ) -> Shaped[Tensor, "*b c h2 w2"]:
        """Takes any number of batch dims"""
        if len(input.shape) <= 4:
            return super().forward(input)

        *B, C, H, W = input.shape
        input = input.view(prod(B), C, H, W)

        out = super().forward(input)

        *_, C, H, W = out.shape
        out = out.view(*B, C, H, W)

        return out
