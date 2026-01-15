from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from torch.utils._pytree import (
    KeyPath,
    MappingKey,
    PyTree,
    key_get,  # noqa: PLC2701
    tree_flatten_with_path,  # noqa: PLC2701
)


def tree_paths(
    tree: PyTree, /, is_leaf: Callable[[PyTree], bool] | None = None
) -> tuple[KeyPath, ...]:
    items, _ = tree_flatten_with_path(tree, is_leaf=is_leaf)
    key_paths, _ = zip(*items, strict=True)

    return key_paths


K = TypeVar("K")
T = TypeVar("T")


def path_to_key(path: tuple[MappingKey[K, T], ...]) -> tuple[K, ...]:
    return tuple(elem.key for elem in path)


def key_get_default(obj: Any, kp: KeyPath, default: object) -> Any:
    try:
        return key_get(obj, kp)
    except KeyError:
        return default


def unflatten_keys(data: Mapping[tuple[Any, ...], Any]) -> dict[Any, Any]:
    out = {}
    for k, v in data.items():
        *k_prefix, k_last = k

        innermost = out
        for k_elem in k_prefix:
            if k_elem not in innermost:
                innermost[k_elem] = {}

            innermost = innermost[k_elem]

        innermost[k_last] = v

    return out
