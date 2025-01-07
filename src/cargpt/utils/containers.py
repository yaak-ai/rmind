from collections.abc import Iterable, Mapping, Sequence
from functools import singledispatchmethod
from typing import Any, TypedDict, Unpack, final, overload, override

import torch
from more_itertools import always_iterable
from optree import PyTree, register_pytree_node, tree_map, tree_paths
from optree.registry import _dict_flatten, _dict_unflatten
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import ModuleDict as _ModuleDict

OPTREE_NAMESPACE = "cargpt"


class TensorDictKwargs(TypedDict):
    batch_size: Sequence[int] | None
    device: torch.device | None


@final
class ModuleDict(_ModuleDict):
    """A convenience wrapper around torch.nn.ModuleDict"""

    __unspecified = object()

    def __init__(self, **modules: Module | Mapping[str, Any]) -> None:
        modules_ = {
            k: type(self)(**v) if isinstance(v, Mapping) else v
            for k, v in modules.items()
        }

        super().__init__(modules=modules_)

    def to_dict(self) -> PyTree[Module]:
        return {  # pyright: ignore[reportReturnType]
            k: v.to_dict() if isinstance(v, ModuleDict) else v
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
        obj = self

        for k in always_iterable(key):
            match obj:
                case ModuleDict():
                    try:
                        obj = obj[k]
                    except KeyError:
                        if default is self.__unspecified:
                            raise KeyError(key) from None

                        return default

                case _:
                    if default is self.__unspecified:
                        raise KeyError(key) from None

                    return default

        return obj

    def get_deepest(self, key: tuple[str, ...]) -> Module:
        obj = self

        for k in always_iterable(key):
            match obj:
                case ModuleDict():
                    try:
                        obj = obj[k]
                    except KeyError:
                        break

                case _:
                    break

        return obj

    @override
    def forward(self, *args, **kwargs) -> TensorDict:
        return self._forward(*args, **kwargs)

    @singledispatchmethod
    def _forward(
        self, *args: TensorDict | Tensor, **kwargs: Unpack[TensorDictKwargs]
    ) -> TensorDict:
        raise NotImplementedError

    @_forward.register
    def _(self, *args: Tensor, **kwargs: Unpack[TensorDictKwargs]) -> TensorDict:
        tree = tree_map(lambda module: module.forward(*args), self.to_dict())

        return TensorDict.from_dict(tree, **kwargs)

    @_forward.register
    def _(self, *args: TensorDict, **kwargs: Unpack[TensorDictKwargs]) -> TensorDict:
        first, *rest = args

        return first.named_apply(  # pyright: ignore[reportReturnType]
            lambda k, *v: self.get_deepest(k).forward(*v),
            *rest,
            nested_keys=True,
            **kwargs,
        )

    def tree_paths(self) -> Iterable[tuple[str, ...]]:
        return tree_paths(self, namespace=OPTREE_NAMESPACE)  # pyright: ignore[reportArgumentType]


register_pytree_node(
    (cls := ModuleDict),  # pyright: ignore[reportArgumentType]
    flatten_func=lambda obj: _dict_flatten(obj._modules),  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
    unflatten_func=lambda keys, values: ModuleDict(**_dict_unflatten(keys, values)),  # pyright: ignore[reportArgumentType]
    namespace=OPTREE_NAMESPACE,
)
