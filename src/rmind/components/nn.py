from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from torch import Tensor, nn
from torch.nn import Module
from torch.utils._pytree import MappingKey, PyTree, tree_map  # noqa: PLC2701

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
    def forward(self, input: PyTree) -> PyTree:
        return tree_map(
            lambda path: key_get_default(input, path, None),
            self._paths,
            is_leaf=lambda x: isinstance(x, tuple),
        )


@final
class Frozen(Module):
    """Wrap a module so it never trains: params frozen and kept in eval mode."""

    @validate_call
    def __init__(self, *, module: InstanceOf[Module]) -> None:
        super().__init__()

        self.module = module.requires_grad_(False).eval()  # noqa: FBT003

    @override
    def train(self, mode: bool = True) -> "Frozen":  # noqa: FBT001, FBT002
        super().train(mode)
        self.module.eval()
        return self

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


@final
class StackFields(Module):
    """Gather ordered `paths` and stack them into one tensor under `out_key`.

    Trailing singleton dims are squeezed; emits `None` when a field is absent.
    """

    @validate_call
    def __init__(
        self, *, paths: Mapping[str, tuple[str, ...]], out_key: str
    ) -> None:
        super().__init__()

        self._paths = {
            name: tuple(map(MappingKey, path)) for name, path in paths.items()
        }
        self.out_key = out_key

    @override
    def forward(self, input: PyTree) -> PyTree:
        fields = [key_get_default(input, path, None) for path in self._paths.values()]

        if any(value is None for value in fields):
            return {**input, self.out_key: None}

        stacked = torch.stack(
            [value.squeeze(-1) if value.shape[-1] == 1 else value for value in fields],
            dim=-1,
        )
        return {**input, self.out_key: stacked}


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
