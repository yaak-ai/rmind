from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def _validate_callback_hook(hook: str) -> str:
    if not callable(getattr(Callback, hook, None)):
        msg = f"`{hook}` is not a valid Lightning callback hook"
        raise TypeError(msg)

    return hook


class SafeCallback(Callback):
    @validate_call
    def __init__(
        self, *, fail_gracefully: bool = True, disable_on_error: bool = False
    ) -> None:
        self._fail_gracefully = fail_gracefully
        self._safe_disable_on_error = disable_on_error
        self._disabled_hooks: set[str] = set()

    def _safe_hook(self, hook: str, fn: Callable[..., T]) -> Callable[..., T | None]:
        hook = _validate_callback_hook(hook)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T | None:
            return self._safe_call(hook, fn, *args, **kwargs)

        return wrapper

    def _safe_call(
        self, hook: str, fn: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T | None:
        hook = _validate_callback_hook(hook)
        if not self._fail_gracefully:
            return fn(*args, **kwargs)

        if hook in self._disabled_hooks:
            return None

        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if self._safe_disable_on_error:
                self._disabled_hooks.add(hook)

            trainer = args[0] if args else kwargs.get("trainer")
            logger.warning(
                "safe callback hook failed",
                callback=type(self).__name__,
                hook=hook,
                disabled=self._safe_disable_on_error,
                epoch=getattr(trainer, "current_epoch", None),
                global_step=getattr(trainer, "global_step", None),
                error_type=type(exc).__name__,
                error=str(exc),
                exc_info=True,
            )

            return None
