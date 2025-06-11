from collections.abc import Callable, Sequence
from typing import Annotated, final

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
    ) -> None:
        self._key = key
        self._select = select
        self._apply = apply
        setattr(self, when, self._call)

    def _call(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or not (
            loggers := [
                logger
                for logger in pl_module.loggers
                if isinstance(logger, WandbLogger)
            ]
        ):
            return

        data = TensorDict.from_module(pl_module).select(*self._select)  # pyright: ignore[reportAttributeAccessIssue]

        if self._apply is not None:
            data = data.apply(self._apply, inplace=False)

        for logger in loggers:
            logger.log_image(
                key=self._key,
                images=[
                    Image(v, caption=".".join(k[:-1]))
                    for k, v in data.cpu().items(include_nested=True, leaves_only=True)
                ],
                step=trainer.global_step,
            )
