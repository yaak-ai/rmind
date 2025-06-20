from collections.abc import Callable, Mapping
from typing import Any

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


def path_to_key[K, T](path: tuple[MappingKey[K, T], ...]) -> tuple[K, ...]:
    return tuple(elem.key for elem in path)


def key_get_default(obj: Any, kp: KeyPath, default: object) -> Any:
    try:
        return key_get(obj, kp)
    except KeyError:
        return default


def unflatten_keys[K, T](data: Mapping[tuple[Any, ...], T]) -> dict[Any, Any]:
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
