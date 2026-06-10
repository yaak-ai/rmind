from collections.abc import Sequence
from typing import Annotated, Any, final

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pydantic import AfterValidator, validate_call
from torch import Tensor
from torch.nn.functional import cosine_similarity
from wandb import Image

from rmind.callbacks.safe import SafeCallback

from .common import (
    BATCH_HOOKS,
    _bind_hook_arguments,
    _figure_to_rgba,
    _get_wandb_loggers,
    _validate_every_n_batch,
    _validate_hook,
)


@final
class WandbEmbeddingSimilarityLogger(SafeCallback):
    """Logs the pairwise cosine-similarity matrix of embedding/parameter rows.

    For each selected tensor of shape `(n, d)` (e.g. a position embedding's
    weight), logs an annotated `n x n` heatmap of `cos(row_i, row_j)`.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        select: Sequence[str],
        every_n_batch: int | None = None,
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
        self._key = key
        self._select = tuple(select)
        _validate_every_n_batch(
            when=when, every_n_batch=every_n_batch, allowed_hooks=BATCH_HOOKS
        )
        self._every_n_batch = every_n_batch
        self._when = when
        setattr(self, when, self._safe_hook(when, self._call))

    @staticmethod
    def _resolve(pl_module: pl.LightningModule, path: str) -> Tensor:
        try:
            return pl_module.get_parameter(path)
        except AttributeError:
            return pl_module.get_buffer(path)

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

        images = [
            Image(self._similarity_heatmap(self._resolve(pl_module, path), path))
            for path in self._select
        ]

        for logger_ in loggers:
            logger_.log_image(key=self._key, images=images, step=trainer.global_step)

    @staticmethod
    def _similarity_heatmap(weight: Tensor, title: str) -> np.ndarray:
        weight = weight.detach().float()
        sim = (
            cosine_similarity(weight.unsqueeze(1), weight.unsqueeze(0), dim=-1)
            .cpu()
            .numpy()
        )
        n = sim.shape[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("position")
            ax.set_ylabel("position")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))

            for i in range(n):
                for j in range(n):
                    val = sim[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val < 0.5 else "black",  # noqa: PLR2004
                    )

            fig.tight_layout()
            return _figure_to_rgba(fig)
        finally:
            plt.close(fig)
