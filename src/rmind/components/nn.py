from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from pydantic import validate_call
from torch import Tensor, nn
from torch.nn import Module
from torch.utils._pytree import (  # ruff: ignore[import-private-name]
    MappingKey,
    PyTree,
    tree_map,
)

from rmind.utils.functional import diff_last
from rmind.utils.pytree import key_get_default

from .base import Invertible

default_weight_init_fn = partial(
    nn.init.trunc_normal_, mean=0.0, std=0.02, a=-0.04, b=0.04
)
default_linear_weight_init_fn = nn.init.xavier_uniform_
default_linear_bias_init_fn = partial(nn.init.constant_, val=0.0)


@final
class Embedding(nn.Embedding):
    def __init__(
        self,
        *args: Any,
        weight_init_fn: Callable[[Tensor], None] = default_weight_init_fn,  # ty:ignore[invalid-parameter-default]
        **kwargs: Any,
    ) -> None:
        self.weight_init_fn: Callable[[Tensor], None] = weight_init_fn

        super().__init__(*args, **kwargs)

    @override
    def reset_parameters(self) -> None:
        self.weight_init_fn(self.weight)
        self._fill_padding_idx_with_zero()


@final
class Linear(nn.Linear):
    def __init__(
        self,
        *args: Any,
        weight_init_fn: Callable[[Tensor], None] = default_linear_weight_init_fn,  # ty:ignore[invalid-parameter-default]
        bias_init_fn: Callable[[Tensor], None] = default_linear_bias_init_fn,  # ty:ignore[invalid-parameter-default]
        **kwargs: Any,
    ) -> None:
        self.weight_init_fn: Callable[[Tensor], None] = weight_init_fn
        self.bias_init_fn: Callable[[Tensor], None] = bias_init_fn

        super().__init__(*args, **kwargs)

    @override
    def reset_parameters(self) -> None:
        self.weight_init_fn(self.weight)
        if self.bias is not None:
            self.bias_init_fn(self.bias)


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


@final
class RandomIlluminationGradient(Module):
    """Multiply by a linear brightness ramp of random direction and strength.

    `ColorJitter` models a *global* light change. This models a directional one -
    a window, an overhead lamp, a shadow falling across part of the frame - which
    is what a printed sheet held up in a warehouse actually sees. Expects float
    input in [0, 1], i.e. before normalization.

    One ramp is sampled per call, matching torchvision v2's per-call semantics.
    """

    @validate_call
    def __init__(self, *, factor: tuple[float, float] = (0.65, 1.35)) -> None:
        super().__init__()

        self.factor: tuple[float, float] = factor

    @override
    def forward(self, input: Tensor) -> Tensor:
        *_, height, width = input.shape
        angle = torch.rand((), device=input.device) * 2 * torch.pi
        rows = torch.linspace(-0.5, 0.5, height, device=input.device)
        cols = torch.linspace(-0.5, 0.5, width, device=input.device)
        ramp = rows[:, None] * torch.cos(angle) + cols[None, :] * torch.sin(angle)
        # normalize to [0, 1] so `factor` bounds the actual gain regardless of angle
        ramp = (ramp - ramp.min()) / (
            ramp.max() - ramp.min() + torch.finfo(ramp.dtype).eps
        )
        low, high = self.factor

        return (input * (low + ramp * (high - low))).clamp(0.0, 1.0)

    @override
    def extra_repr(self) -> str:
        return f"factor={list(self.factor)}"


@final
class TrainOnly(Module):
    """Apply `module` in training mode only; identity otherwise.

    Lets train-time-only transforms (e.g. image augmentation) live inside the
    model's `input_transform`, so they are absent from `predict`/`validation`
    and traced away by `torch.export` (which runs the module in eval mode).
    """

    def __init__(self, *, module: Module) -> None:
        super().__init__()

        self.module: Module = module

    @override
    def forward(self, input: Tensor) -> Tensor:
        return self.module(input) if self.training else input


type Paths = Mapping[str, tuple[str, ...] | Paths]


@final
class Remapper(Module):
    @validate_call
    def __init__(self, paths: Paths) -> None:
        super().__init__()

        self._paths = tree_map(
            lambda path: tuple(map(MappingKey, path)),
            paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )

    @property
    def paths(self) -> PyTree:
        return self._paths

    @override
    def extra_repr(self) -> str:
        return str(
            tree_map(
                lambda path: tuple(x.key for x in path),
                self._paths,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        )

    @override
    def forward(self, input: PyTree) -> PyTree:
        return tree_map(
            lambda path: key_get_default(input, path, None),
            self._paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )


def _module_wrapper(
    fn: Callable[..., Tensor], *, name: str | None = None
) -> type[nn.Module]:
    @final
    class _Fn(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

            self._kwargs: Any = kwargs

        @override
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **(self._kwargs | kwargs))

    if name is not None:
        _Fn.__name__ = name

    return _Fn


AtLeast3D = _module_wrapper(torch.atleast_3d, name="AtLeast3D")
DiffLast = _module_wrapper(diff_last, name="DiffLast")
