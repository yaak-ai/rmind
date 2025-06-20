from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Invertible(Protocol):
    def invert(self, input: Tensor) -> Tensor: ...


type TensorDictExport = dict[str, Tensor | TensorDictExport]
