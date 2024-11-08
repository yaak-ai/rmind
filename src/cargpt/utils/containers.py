from collections.abc import Iterable, Mapping, Sequence
from functools import reduce, singledispatchmethod
from typing import Any, TypedDict, Unpack, override

import torch
from more_itertools import always_iterable
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict as ModuleDictBase
from torch.utils._pytree import (
    Context,
    KeyEntry,
    _dict_flatten,
    _dict_flatten_with_keys,
    _generate_key_paths,
    register_pytree_node,
    tree_flatten_with_path,
)


class TensorDictKwargs(TypedDict):
    batch_size: Sequence[int] | None
    device: torch.device | None


class ModuleDict(ModuleDictBase):
    """A convenience wrapper around torch.nn.ModuleDict"""

    __unspecified = object()

    def __init__(self, **modules: Module | Mapping[str, Module]) -> None:
        """recursive instantiation for the sake of smaller YAMLs"""

        _modules = {
            k: v if isinstance(v, Module) else ModuleDict(**v)
            for k, v in modules.items()
        }

        super().__init__(modules=_modules)

    def get(self, key: str | tuple[str, ...], *, default: Any = __unspecified):
        """recursive access mimicking TensorDict.get"""
        try:
            return reduce(ModuleDict.__getitem__, always_iterable(key), self)  # noqa: DOC201 # pyright: ignore[reportArgumentType]
        except KeyError:
            if default is self.__unspecified:
                raise  # noqa: DOC501

            return default

    @override
    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    @singledispatchmethod
    def _forward(
        self,
        *input: TensorDict | Tensor,
        **kwargs: Unpack[TensorDictKwargs],  # noqa: UP044
    ) -> TensorDict:
        raise NotImplementedError

    @_forward.register
    def _(self, *args: Tensor, **kwargs: Unpack[TensorDictKwargs]) -> TensorDict:  # noqa: UP044
        return TensorDict(
            {
                k: v.forward(*args)
                for k, v in self.tree_flatten_with_path()  # pyright: ignore[reportArgumentType]
            },
            **kwargs,  # pyright: ignore[reportArgumentType]
        )

    @_forward.register
    def _(self, *args: TensorDict, **kwargs: Unpack[TensorDictKwargs]) -> TensorDict:  # noqa: UP044
        first, *others = args

        return first.named_apply(  # pyright: ignore[reportReturnType]
            lambda k, *v: self.get(k).forward(*v), *others, nested_keys=True, **kwargs
        )

    def tree_paths(self) -> Iterable[tuple[str, ...]]:
        paths, _ = zip(*_generate_key_paths((), self), strict=True)
        return (tuple(mk.key for mk in path) for path in paths)

    def tree_flatten_with_path(self) -> Iterable[tuple[tuple[str, ...], Module]]:
        kv, _ = tree_flatten_with_path(self)

        return ((tuple(mk.key for mk in k), v) for k, v in kv)  # pyright: ignore[reportAttributeAccessIssue]


def _moduledict_flatten(dct: ModuleDictBase) -> tuple[list[Module], list[str]]:
    return _dict_flatten(dct._modules)


def _moduledict_flatten_with_keys(
    dct: ModuleDictBase,
) -> tuple[list[tuple[KeyEntry, Any]], Context]:
    return _dict_flatten_with_keys(dct._modules)


def _moduledict_unflatten(*_args, **_kwargs):
    raise NotImplementedError


register_pytree_node(
    ModuleDictBase,
    flatten_fn=_moduledict_flatten,
    unflatten_fn=_moduledict_unflatten,
    flatten_with_keys_fn=_moduledict_flatten_with_keys,
)

register_pytree_node(
    ModuleDict,
    flatten_fn=_moduledict_flatten,
    unflatten_fn=_moduledict_unflatten,
    flatten_with_keys_fn=_moduledict_flatten_with_keys,
)
