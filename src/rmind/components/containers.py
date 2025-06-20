from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, Unpack, overload, override

import torch
from more_itertools import always_iterable
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict as _ModuleDict
from torch.utils._pytree import (
    MappingKey,  # noqa: PLC2701
    _dict_flatten,  # pyright: ignore[reportPrivateUsage] # noqa: PLC2701
    _dict_flatten_with_keys,  # pyright: ignore[reportPrivateUsage] # noqa: PLC2701
    _dict_unflatten,  # pyright: ignore[reportPrivateUsage] # noqa: PLC2701
    key_get,  # noqa: PLC2701
    register_pytree_node,  # noqa: PLC2701
    tree_flatten_with_path,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
)

from rmind.components.base import TensorDictExport


class TensorDictKwargs(TypedDict, total=False):
    batch_size: Sequence[int] | None
    device: torch.device | None


type Modules = Mapping[str, InstanceOf[Module] | Modules]


class ModuleDict(_ModuleDict):
    """A convenience wrapper around torch.nn.ModuleDict."""

    __unspecified = object()

    @validate_call
    def __init__(self, modules: Modules) -> None:
        modules_ = {
            k: type(self)(v) if isinstance(v, Mapping) else v
            for k, v in modules.items()
        }

        super().__init__(modules=modules_)

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v.to_dict() if isinstance(v, type(self)) else v
            for k, v in self._modules.items()
        }

    @overload
    def get(self, key: str | tuple[str, ...]) -> Module: ...

    @overload
    def get(
        self, key: str | tuple[str, ...], *, default: object
    ) -> Module | object: ...

    def get(
        self, key: str | tuple[str, ...], *, default: object = __unspecified
    ) -> Module | object:
        key_path = tuple(map(MappingKey, always_iterable(key)))

        try:
            return key_get(self, key_path)
        except (KeyError, TypeError) as e:
            if default is self.__unspecified:
                raise KeyError from e

            return default

    def get_deepest(self, key: tuple[str, ...]) -> Module:
        obj = self

        for k in always_iterable(key):
            if isinstance(obj, ModuleDict):
                try:
                    obj = obj[k]
                except KeyError:
                    break
            else:
                break

        return obj

    @overload
    def forward(
        self, *args: TensorDict, **kwargs: Unpack[TensorDictKwargs]
    ) -> TensorDict: ...

    @overload
    def forward(
        self, *args: Tensor, **kwargs: Unpack[TensorDictKwargs]
    ) -> TensorDict | TensorDictExport: ...

    @overload
    def forward(self, *args: TensorDictExport) -> TensorDictExport: ...

    @override
    def forward(
        self,
        *args: Tensor | TensorDict | TensorDictExport,
        **kwargs: Unpack[TensorDictKwargs],
    ) -> TensorDict | TensorDictExport:
        modules = self.to_dict()

        if all(isinstance(arg, TensorDict) for arg in args):
            tree = tree_map(tree_map, modules, *(arg.to_dict() for arg in args))

            for k in TensorDictKwargs.__annotations__:
                if (
                    k not in kwargs
                    and len(values := {getattr(arg, k) for arg in args}) == 1
                ):
                    kwargs[k] = values.pop()

            return TensorDict(tree, **kwargs)

        if all(isinstance(arg, Tensor) for arg in args):
            tree = tree_map(lambda module: module.forward(*args), modules)

            if torch.compiler.is_exporting():
                if kwargs:
                    raise NotImplementedError

                return tree

            return TensorDict(tree, **kwargs)

        return tree_map(tree_map, modules, *args)

    def tree_paths(self) -> tuple[tuple[str, ...], ...]:
        items, _ = tree_flatten_with_path(self)
        key_paths = (item[0] for item in items)

        return tuple(tuple(elem.key for elem in key_path) for key_path in key_paths)


register_pytree_node(
    _ModuleDict,
    flatten_fn=lambda x: _dict_flatten(x._modules),  # noqa: SLF001
    flatten_with_keys_fn=lambda x: _dict_flatten_with_keys(x._modules),  # noqa: SLF001
    unflatten_fn=lambda values, context: _ModuleDict(
        modules=_dict_unflatten(values, context)
    ),
)

register_pytree_node(
    ModuleDict,
    flatten_fn=lambda x: _dict_flatten(x._modules),  # noqa: SLF001
    flatten_with_keys_fn=lambda x: _dict_flatten_with_keys(x._modules),  # noqa: SLF001
    unflatten_fn=lambda values, context: ModuleDict(
        modules=_dict_unflatten(values, context)
    ),
)
