"""Monkey-patch for pytorch/tensordict#1003.

Two bugs prevent TensorDict from being used in torch.export with dynamic shapes:

1. _parse_batch_size rejects scalar SymInt (e.g. batch_size=x.shape[0]) because
   torch.SymInt does not subclass numbers.Number, falling through to ValueError.

2. _tensordict_flatten stored batch_size (torch.Size that may contain SymInts)
   in the pytree context. torch.export cannot serialize SymInts in the output
   pytree spec (as_python_constant raises).

Both are fixed in pytorch/tensordict#1704 (merged). This module is a local
workaround until a release ships that includes the fix. Remove once tensordict
is bumped past the fix commit.

Call `apply()` once before torch.export.export() (e.g. in the export script).
"""

from __future__ import annotations

import numbers
from typing import Any

import tensordict._pytree as _td_pytree  # noqa: PLC2701
import torch
import torch.utils._pytree as _torch_pytree  # noqa: PLC2701
from tensordict.utils import _shape, is_compiling  # noqa: PLC2701
from torch.utils._pytree import NodeDef  # noqa: PLC2701

# Fix 1: _tensordict_flatten / _tensordict_unflatten
#
# Match the merged upstream fix: store batch_dims (int) only during compilation
# so the pytree context is always serializable, while keeping batch_size for
# eager execution so that jacrev/jacfwd mismatch detection still works.


def _flatten(d: Any) -> tuple[list[Any], dict]:
    items = tuple(d.items())
    if items:
        keys, values = zip(*items, strict=False)
        keys, values = list(keys), list(values)
    else:
        keys, values = [], []
    ctx: dict = {
        "keys": keys,
        "names": d.names if d._has_names() else None,  # noqa: SLF001
        "device": d.device,
        "constructor": _td_pytree._CONSTRUCTORS[type(d)],  # noqa: SLF001
        "non_tensor_data": d.non_tensor_items(),
        "cls": type(d),
    }
    if is_compiling():
        ctx["batch_dims"] = len(d.batch_size)
    else:
        ctx["batch_size"] = d.batch_size
    return values, ctx


def _unflatten(values: list[Any], ctx: dict) -> Any:
    device = ctx["device"]
    if device is not None and not all(
        getattr(v, "device", device) == device for v in values
    ):
        device = None
    if any(v is None for v in values):
        return None
    shapes = [_shape(v) for v in values if hasattr(v, "shape")]
    if "batch_dims" in ctx:
        batch_dims = ctx["batch_dims"]
        batch_size = shapes[0][:batch_dims] if shapes else torch.Size([0] * batch_dims)
    else:
        batch_size = ctx["batch_size"]
        batch_dims = len(batch_size)
        if shapes and any(s[:batch_dims] != batch_size for s in shapes):
            min_dims = min(len(s) for s in shapes)
            max_prefix_len = min(min_dims, batch_dims + 1)
            common_dims = 0
            for i in range(max_prefix_len):
                if all(s[i] == shapes[0][i] for s in shapes):
                    common_dims = i + 1
                else:
                    break
            batch_size = torch.Size(shapes[0][:common_dims])
            ctx["names"] = None
    return ctx["constructor"](
        cls=ctx["cls"],
        keys=ctx["keys"],
        values=values,
        batch_size=batch_size,
        names=ctx["names"],
        device=device,
        non_tensor_items=ctx["non_tensor_data"],
    )


# Fix 2: _parse_batch_size
#
# torch.SymInt does not subclass numbers.Number, so scalar SymInt batch sizes
# (e.g. batch_size=x.shape[0]) fell through to ValueError during compilation.


def _patched_parse_batch_size(  # noqa: C901, PLR0911
    source: Any, batch_size: Any = None
) -> torch.Size:
    from tensordict.base import TensorDictBase  # noqa: PLC0415

    err = (
        "batch size was not specified when creating the TensorDict instance "
        "and it could not be retrieved from source."
    )
    if is_compiling():
        if isinstance(batch_size, torch.Size):
            return batch_size
        if isinstance(batch_size, tuple):
            return torch.Size(batch_size)
        if isinstance(batch_size, list):
            return torch.Size(tuple(batch_size))
        if batch_size is None:
            return torch.Size([])
        if isinstance(batch_size, (numbers.Number, torch.SymInt)):
            return torch.Size([batch_size])
        if isinstance(source, TensorDictBase):
            return source.batch_size
        raise ValueError
    try:
        return torch.Size(batch_size)
    except Exception:  # noqa: BLE001
        if batch_size is None:
            return torch.Size([])
        if isinstance(batch_size, numbers.Number):
            return torch.Size([batch_size])
        if isinstance(source, TensorDictBase):
            return source.batch_size
        raise ValueError(err) from None


_patched: dict[str, bool] = {}


def apply() -> None:
    """Apply both tensordict#1003 fixes. Safe to call multiple times."""
    from tensordict import TensorDict  # noqa: PLC0415
    from tensordict.base import TensorDictBase  # noqa: PLC0415

    # Fix 1: pytree context
    for cls, nodedef in list(_torch_pytree.SUPPORTED_NODES.items()):
        if (
            isinstance(cls, type)
            and issubclass(cls, TensorDictBase)
            and nodedef.flatten_fn is not _flatten
        ):
            _torch_pytree.SUPPORTED_NODES[cls] = NodeDef(
                type=cls,
                flatten_fn=_flatten,
                unflatten_fn=_unflatten,  # ty:ignore[invalid-argument-type]
                flatten_with_keys_fn=nodedef.flatten_with_keys_fn,
            )

    # Fix 2: _parse_batch_size scalar SymInt
    if not _patched.get("parse_batch_size"):
        TensorDict._parse_batch_size = staticmethod(  # noqa: SLF001  # ty:ignore[invalid-assignment]
            _patched_parse_batch_size
        )
        _patched["parse_batch_size"] = True
