from collections.abc import Callable
from functools import partial
from typing import Any, override

from torch import Tensor, nn

from .base import Invertible

_default_embedding_weight_init_fn = partial(nn.init.normal_, mean=0.0, std=0.02)


class Embedding(nn.Embedding):
    def __init__(
        self,
        *args: Any,
        weight_init_fn: Callable[[Tensor], Any] = _default_embedding_weight_init_fn,
        **kwargs: Any,
    ) -> None:
        self.weight_init_fn: Callable[[Tensor], Any] = weight_init_fn

        super().__init__(*args, **kwargs)

    @override
    def reset_parameters(self) -> None:
        self.weight_init_fn(self.weight)
        self._fill_padding_idx_with_zero()


class Sequential(nn.Sequential, Invertible):
    @override
    def invert(self, input: Tensor) -> Tensor:
        for module in reversed(self):
            input = module.invert(input)
        return input


class Identity(nn.Identity, Invertible):
    @override
    def invert(self, input: Tensor) -> Tensor:
        return input
