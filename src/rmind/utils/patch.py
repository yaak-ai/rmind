from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any


@contextlib.contextmanager
def monkeypatched(obj: Any, name: str, patch: Any) -> Generator[Any, None, None]:
    old = getattr(obj, name)
    setattr(obj, name, patch)
    yield obj
    setattr(obj, name, old)
