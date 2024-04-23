from functools import reduce
from typing import Any

from typing_extensions import override


class attrdeleter:
    """Like operator.attrgetter but for deletion"""

    def __init__(self, attr: str, *attrs: str):  # pyright: ignore[reportMissingSuperCall]
        self._attrs = tuple(_attr.split(".") for _attr in (attr, *attrs))

    def __call__(self, obj: Any) -> None:
        for *predecessors, leaf in self._attrs:
            reduce(getattr, predecessors, obj).__delattr__(leaf)

    @override
    def __repr__(self):
        return "{}.{}({})".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            ", ".join(map(repr, self._attrs)),
        )
