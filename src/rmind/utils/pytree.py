from collections.abc import Callable, Mapping
from typing import Any

from torch.utils._pytree import (  # noqa: PLC2701
    KeyPath,
    MappingKey,
    PyTree,
    tree_flatten_with_path,
)


def tree_paths(
    tree: PyTree, /, is_leaf: Callable[[PyTree], bool] | None = None
) -> tuple[KeyPath, ...]:
    items, _ = tree_flatten_with_path(tree, is_leaf=is_leaf)
    key_paths, _ = zip(*items, strict=True)

    return key_paths


def path_to_key[K, T](path: tuple[MappingKey[K, T], ...]) -> tuple[K, ...]:
    return tuple(elem.key for elem in path)


def key_get_default(
    obj: Any, kp: tuple[MappingKey[Any, Any], ...], default: object
) -> Any:
    # Index via `MappingKey.key` rather than torch's `key_get`, whose `MappingKey.get(obj)` method call dynamo can't trace under export (torch>=2.12).
    try:
        for k in kp:
            obj = obj[k.key]
    except KeyError:
        return default
    return obj


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
