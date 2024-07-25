from typing import Self

from tensordict.tensorclass import _eq, _getitem
from tensordict.utils import IndexType
from .branch import Branch

# monkey patch incorrect annotations
_getitem.__annotations__["item"] = IndexType
_eq.__annotations__["return"] = Self

__all__ = ["Branch"]
