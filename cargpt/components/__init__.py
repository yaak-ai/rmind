from typing import Self

from tensordict.tensorclass import _eq, _getitem  # noqa: PLC2701
from tensordict.utils import IndexType

from cargpt.utils.attention import MemoryEfficientScaledDotProduct  # noqa #type: ignore

# monkey patch incorrect annotations
_getitem.__annotations__["item"] = IndexType
_eq.__annotations__["return"] = Self
