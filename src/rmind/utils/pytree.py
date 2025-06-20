from collections.abc import Callable, Mapping
from typing import Any

from torch.utils._pytree import (
    KeyPath,
    MappingKey,
    PyTree,
    key_get,
    tree_flatten_with_path,
    tree_map,
)


def tree_remap(paths: PyTree, tree: PyTree) -> PyTree:
    return tree_map(
        lambda path: key_get(tree, path), paths, is_leaf=lambda x: isinstance(x, tuple)
    )


def tree_paths(
    tree: PyTree, /, is_leaf: Callable[[PyTree], bool] | None = None
) -> tuple[KeyPath, ...]:
    items, _ = tree_flatten_with_path(tree, is_leaf=is_leaf)
    key_paths, _ = zip(*items, strict=True)

    return key_paths


def path_to_key[K, T](path: tuple[MappingKey[K, T], ...]) -> tuple[K, ...]:
    return tuple(elem.key for elem in path)


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
