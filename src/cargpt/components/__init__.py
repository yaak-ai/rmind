from typing import Self

from tensordict.tensorclass import _eq, _getitem
from tensordict.utils import IndexType

# monkey patch incorrect annotations
_getitem.__annotations__["item"] = IndexType
_eq.__annotations__["return"] = Self
