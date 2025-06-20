from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, final, overload, override

import torch
from pydantic import validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn import Module
from torch.utils._pytree import MappingKey, tree_map  # noqa: PLC2701

from rmind.utils.functional import diff_last
from rmind.utils.pytree import tree_remap

from .base import Invertible, TensorDictExport

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

    @overload
    def forward(self, input: TensorDict) -> TensorDict: ...

    @overload
    def forward(self, input: TensorDictExport) -> TensorDictExport: ...

    @override
    def forward(
        self, input: TensorDict | TensorDictExport
    ) -> TensorDict | TensorDictExport:
        remapped = tree_remap(self._paths, input)
        if isinstance(input, TensorDict):
            return input.empty().update(remapped)

        return remapped


def _module_wrapper(
    fn: Callable[..., Tensor], *, name: str | None = None
) -> type[nn.Module]:
    @final
    class _Fn(nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()

            self._kwargs = kwargs

        @override
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return fn(*args, **(self._kwargs | kwargs))

    if name is not None:
        _Fn.__name__ = name

    return _Fn


AtLeast3D = _module_wrapper(torch.atleast_3d, name="AtLeast3D")
DiffLast = _module_wrapper(diff_last, name="DiffLast")
