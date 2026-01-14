from typing import Protocol, TypeAlias, runtime_checkable

from torch import Tensor


@runtime_checkable
class Invertible(Protocol):
    def invert(self, input: Tensor) -> Tensor: ...


TensorTree: TypeAlias = dict[str, "Tensor | TensorTree"]
