from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Invertible(Protocol):
    def invert(self, input: Tensor) -> Tensor: ...


type TensorTree = dict[str, Tensor | TensorTree]
