from typing import Any, Protocol


class Invertible(Protocol):
    def invert(self, *args, **kwargs) -> Any: ...
