import inspect
from typing import Any

import numpy as np
import pytorch_lightning as pl
from matplotlib.figure import Figure
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.loggers import WandbLogger


def _validate_hook(value: str) -> str:
    if not callable(getattr(ModelHooks, value, None)):
        raise ValueError  # noqa: TRY004

    return value


BATCH_HOOKS = frozenset({
    "on_train_batch_start",
    "on_train_batch_end",
    "on_validation_batch_start",
    "on_validation_batch_end",
    "on_test_batch_start",
    "on_test_batch_end",
    "on_predict_batch_start",
    "on_predict_batch_end",
})


def _validate_every_n_batch(
    *, when: str, every_n_batch: int | None, allowed_hooks: frozenset[str]
) -> None:
    if every_n_batch is None:
        return

    if when not in allowed_hooks:
        msg = (
            "`every_n_batch` is only supported for hooks: "
            + ", ".join(f"`{hook}`" for hook in sorted(allowed_hooks))
            + f". Got `{when}`"
        )
        raise ValueError(msg)


def _get_wandb_loggers(pl_module: pl.LightningModule) -> list[WandbLogger]:
    return [
        logger_ for logger_ in pl_module.loggers if isinstance(logger_, WandbLogger)
    ]


def _bind_hook_arguments(
    callback: Callback, when: str, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    base_hook_method = getattr(pl.Callback, when)
    sig = inspect.signature(base_hook_method)
    bound_args = sig.bind(callback, *args, **kwargs)
    bound_args.apply_defaults()

    return bound_args.arguments


def _figure_to_rgba(fig: Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()  # ty:ignore[unresolved-attribute]
