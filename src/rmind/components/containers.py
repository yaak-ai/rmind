from collections.abc import Callable, Mapping
from typing import Any, overload, override

from more_itertools import always_iterable
from pydantic import InstanceOf, validate_call
from torch.nn import Module
from torch.nn import ModuleDict as _ModuleDict
from torch.utils._pytree import (
    MappingKey,  # noqa: PLC2701
    PyTree,
    _dict_flatten,  # noqa: PLC2701
    _dict_flatten_with_keys,  # noqa: PLC2701
    _dict_unflatten,  # noqa: PLC2701
    key_get,  # noqa: PLC2701
    register_pytree_node,  # noqa: PLC2701
    tree_flatten_with_path,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
)

type Modules = Mapping[str, InstanceOf[Module] | Modules | None]


class ModuleDict(_ModuleDict):
    """A convenience wrapper around torch.nn.ModuleDict."""

    __unspecified = object()

    @validate_call
    def __init__(self, modules: Modules) -> None:
        modules_ = {
            k: type(self)(v) if isinstance(v, Mapping) else v  # ty:ignore[invalid-argument-type]
            for k, v in modules.items()
        }

        super().__init__(modules=modules_)  # ty:ignore[invalid-argument-type]

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

    @override
    def forward(
        self,
        *args: PyTree,
        broadcast: bool | None = None,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        if broadcast is None and all(isinstance(arg, Mapping) for arg in args):
            broadcast = True

        modules = self.to_dict()

        return (
            # should be roughly equivalent to
            # `optree.tree_broadcast_map(operator.call, modules, *args, none_is_leaf=True, is_leaf=is_leaf)`
            tree_map(
                lambda mod, *xs: (
                    tree_map(
                        lambda *_xs: (
                            mod(*_xs) if all(x is not None for x in _xs) else None
                        ),
                        *xs,
                        is_leaf=is_leaf,
                    )
                    if mod is not None
                    else None
                ),
                modules,
                *args,
                is_leaf=is_leaf,
            )
            if broadcast
            else tree_map(lambda mod: mod(*args), modules, is_leaf=is_leaf)
        )

    def tree_paths(self) -> tuple[tuple[str, ...], ...]:
        items, _ = tree_flatten_with_path(self)
        key_paths = (item[0] for item in items)

        return tuple(tuple(elem.key for elem in key_path) for key_path in key_paths)  # ty:ignore[unresolved-attribute]


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
