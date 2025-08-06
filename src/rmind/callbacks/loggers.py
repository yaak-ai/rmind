import inspect
from collections.abc import Callable, Sequence
from typing import Annotated, Any, final

import pytorch_lightning as pl
from pydantic import AfterValidator, validate_call
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.loggers import WandbLogger
from tensordict import TensorDict
from torch import Tensor
from wandb import Image


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


@final
class WandbImageParamLogger(Callback):
    @validate_call
    def __init__(
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        select: Sequence[str | tuple[str, ...]],
        apply: Callable[[Tensor], Tensor] | None = None,
        every_n_batch: int | None = None,
    ) -> None:
        self._key = key
        self._select = select
        self._apply = apply
        if every_n_batch is not None and when not in BATCH_HOOKS:
            msg = (
                "`every_n_batch` is only supported for batch-based hooks: "
                + ", ".join(f"`{hook}`" for hook in BATCH_HOOKS)
                + f". Got `{when}`"
            )
            raise ValueError(msg)
        self._every_n_batch = every_n_batch
        self._when = when
        setattr(self, when, self._call)

    def _call(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if trainer.sanity_checking or not (
            loggers := [
                logger
                for logger in pl_module.loggers
                if isinstance(logger, WandbLogger)
            ]
        ):
            return

        base_hook_method = getattr(pl.Callback, self._when)
        sig = inspect.signature(base_hook_method)

        bound_args = sig.bind(self, trainer, pl_module, *args, **kwargs)
        bound_args.apply_defaults()
        batch_idx = bound_args.arguments.get("batch_idx")

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ):
            return

        data = TensorDict.from_module(pl_module).select(*self._select)  # pyright: ignore[reportAttributeAccessIssue]

        if self._apply is not None:
            data = data.apply(self._apply, inplace=False)

        for logger in loggers:
            logger.log_image(
                key=self._key,
                images=[
                    Image(
                        ((v - v.min()) / (v.max() - v.min()) * 255)
                        .clamp(0, 255)
                        .unsqueeze(0),
                        caption=".".join(k[:-1]),
                    )
                    for k, v in data.cpu().items(include_nested=True, leaves_only=True)
                ],
                step=trainer.global_step,
            )
