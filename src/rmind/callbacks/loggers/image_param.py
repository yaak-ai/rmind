from collections.abc import Callable, Sequence
from typing import Annotated, Any, final

import kornia.color as K
import pytorch_lightning as pl
from einops import rearrange
from pydantic import AfterValidator, validate_call
from tensordict import TensorDict
from torch import Tensor
from wandb import Image

from rmind.callbacks.safe import SafeCallback

from .common import (
    BATCH_HOOKS,
    _bind_hook_arguments,
    _get_wandb_loggers,
    _validate_every_n_batch,
    _validate_hook,
)


@final
class WandbImageParamLogger(SafeCallback):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        select: Sequence[str | tuple[str, ...]],
        apply: Callable[[Tensor], Tensor] | None = None,
        every_n_batch: int | None = None,
        cmap_type: K.ColorMapType | None = K.ColorMapType.viridis,
        fail_gracefully: bool = True,
        disable_on_error: bool = True,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
        self._key = key
        self._select = select
        self._apply = apply
        self._cmap_type = cmap_type
        self._cmap: K.ApplyColorMap | None = None
        _validate_every_n_batch(
            when=when, every_n_batch=every_n_batch, allowed_hooks=BATCH_HOOKS
        )
        self._every_n_batch = every_n_batch
        self._when = when
        setattr(self, when, self._safe_hook(when, self._call))

    def _call(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        loggers = _get_wandb_loggers(pl_module)
        if trainer.sanity_checking or not loggers or not trainer.is_global_zero:
            return

        bound_args = _bind_hook_arguments(
            self, self._when, trainer, pl_module, *args, **kwargs
        )
        batch_idx = bound_args.get("batch_idx")

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ):
            return

        data = TensorDict.from_module(pl_module).select(*self._select)

        if self._apply is not None:
            data = data.apply(self._apply, inplace=False)

        data = data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

        if self._cmap_type is not None:
            if self._cmap is None:
                self._cmap = K.ApplyColorMap(
                    K.ColorMap(self._cmap_type, device=pl_module.device)
                )
            data = (
                data
                .apply(lambda x: rearrange(x, "h w -> 1 1 h w"))
                .apply(self._cmap)
                .apply(lambda x: rearrange(x, "1 c h w -> h w c"))
            )

        for logger_ in loggers:
            logger_.log_image(
                key=self._key,
                images=[
                    Image((v * 255).byte().cpu().numpy(), caption=".".join(k[:-1]))
                    for k, v in data.items(include_nested=True, leaves_only=True)
                ],
                step=trainer.global_step,
            )
