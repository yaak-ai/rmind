from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, TypeAlias, final

from typing_extensions import override

import torch
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


Paths: TypeAlias = "Mapping[str, tuple[str, ...] | Paths]"


def _is_path_leaf(obj: Any) -> bool:
    """Check if obj is a leaf path (tuple/list of strings)."""
    return isinstance(obj, (tuple, list)) and obj and all(isinstance(elem, str) for elem in obj)


def _is_mapping_key_path(obj: Any) -> bool:
    """Check if obj is a tuple of MappingKeys."""
    return isinstance(obj, tuple) and obj and all(isinstance(elem, MappingKey) for elem in obj)


def _convert_paths(paths: Paths | tuple[str, ...] | list[Any]) -> Any:
    """Recursively convert path tuples/lists to MappingKey tuples."""
    if _is_path_leaf(paths):
        return tuple(MappingKey(elem) for elem in paths)
    elif isinstance(paths, Mapping):
        return {k: _convert_paths(v) for k, v in paths.items()}
    else:
        return paths


def _map_paths(fn: Callable[[Any], Any], paths: Any) -> Any:
    """Apply fn to leaf tuples (MappingKey paths) in the structure."""
    if _is_mapping_key_path(paths):
        return fn(paths)
    elif isinstance(paths, dict):
        return {k: _map_paths(fn, v) for k, v in paths.items()}
    else:
        return paths


@final
class Remapper(Module):
    def __init__(self, paths: Paths) -> None:
        super().__init__()

        self._paths = _convert_paths(paths)
        self._paths_converted = True

    @property
    def paths(self) -> PyTree:
        return self._paths

    def _ensure_paths_converted(self) -> None:
        """Ensure paths are converted to MappingKey tuples (handles pickled models)."""
        if not getattr(self, "_paths_converted", False):
            self._paths = _convert_paths(self._paths)
            self._paths_converted = True

    @override
    def forward(self, input: PyTree) -> PyTree:
        self._ensure_paths_converted()
        return _map_paths(
            lambda path: key_get_default(input, path, None),
            self._paths,
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
