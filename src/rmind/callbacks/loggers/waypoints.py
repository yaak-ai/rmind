from enum import StrEnum, auto
from typing import Annotated, Any, final

import contextily as ctx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from contextily.tile import requests
from einops import rearrange
from pydantic import AfterValidator, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import Tensor
from torch.utils._pytree import MappingKey, key_get, tree_map  # noqa: PLC2701
from wandb import Image

from rmind.utils.pytree import key_get_default

from .common import (
    BATCH_HOOKS,
    _bind_hook_arguments,
    _figure_to_rgba,
    _get_wandb_loggers,
    _validate_every_n_batch,
    _validate_hook,
)

logger = get_logger(__name__)

NoneKey = (MappingKey(None),)


@final
class WandbWaypointsLogger(Callback):
    class DataColumns(StrEnum):
        IMAGE = auto()
        WAYPOINTS_XY_NORMALIZED = auto()
        WAYPOINTS_XY = auto()
        EGO_XY = auto()

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        data: dict[DataColumns, list[str | int]],
        caption: dict[str, list[str | int]],
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        every_n_batch: int | None = None,
        crs: str | None = None,
    ) -> None:
        self._key = key
        _validate_every_n_batch(
            when=when, every_n_batch=every_n_batch, allowed_hooks=BATCH_HOOKS
        )
        self._every_n_batch = every_n_batch
        self._when = when
        self._data_paths = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            data,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._caption_paths = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            caption,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._crs = crs
        setattr(self, when, self._call)

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
        batch = bound_args.get("batch")

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ) or (trainer.current_epoch != 0):
            return

        caption = " | ".join(
            f"{k}: {key_get(batch, v).item() if key_get(batch, v) is Tensor else key_get(batch, v)}"
            for k, v in self._caption_paths.items()
        )

        log_images: list[Image] = []
        if (
            image := key_get_default(
                batch, self._data_paths.get(self.DataColumns.IMAGE, NoneKey), None
            )
        ) is not None:
            log_images.append(
                Image(rearrange(image, "w h c -> c w h"), caption=caption)
            )
        if (
            wpts_xy_normalized := key_get_default(
                batch,
                self._data_paths.get(self.DataColumns.WAYPOINTS_XY_NORMALIZED, NoneKey),
                None,
            )
        ) is not None:
            log_images.append(
                self._plot_waypoints_normalized(
                    wpts_xy_normalized=wpts_xy_normalized, caption=caption
                )
            )
        if (
            wpts_xy := key_get_default(
                batch,
                self._data_paths.get(self.DataColumns.WAYPOINTS_XY, NoneKey),
                None,
            )
        ) is not None:
            log_images.append(
                self._plot_waypoints_on_map(
                    wpts_xy=wpts_xy,
                    ego_xy=key_get_default(
                        batch,
                        self._data_paths.get(self.DataColumns.EGO_XY, NoneKey),
                        None,
                    ),
                    caption=caption,
                )
            )
        if not log_images:
            return

        for logger_ in loggers:
            logger_.log_image(
                key=self._key, images=log_images, step=trainer.global_step
            )

    @staticmethod
    def _plot_waypoints_normalized(
        wpts_xy_normalized: Tensor, caption: str | None = None
    ) -> Image:
        wpts_xy_normalized = wpts_xy_normalized.cpu()

        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
        try:
            wpts_x = wpts_xy_normalized[:, 0]
            wpts_y = wpts_xy_normalized[:, 1]
            ax.plot(wpts_x, wpts_y, "bo-", markersize=5)
            for i, (x, y) in enumerate(zip(wpts_x, wpts_y, strict=True)):
                ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")
            ax.plot(0, 0, "ro")
            ax.grid(True)  # noqa: FBT003
            ax.axis("equal")
            return Image(_figure_to_rgba(fig), caption=caption)
        finally:
            plt.close(fig)

    @staticmethod
    def _plot_waypoints_on_map(
        wpts_xy: Tensor,
        ego_xy: Tensor | None = None,
        caption: str | None = None,
        map_zoom_factor: float = 2.5,
        crs: str | None = "EPSG:25832",
    ) -> Image:
        wpts_xy = wpts_xy.cpu()

        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
        try:
            ax.scatter(
                wpts_xy[:, 0],
                wpts_xy[:, 1],
                color="deeppink",
                s=150,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )

            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            x_range = x_max - x_min
            y_range = y_max - y_min

            x_center = x_min + x_range / 2
            y_center = y_min + y_range / 2

            plot_range = max(x_range, y_range) * map_zoom_factor

            ax.set_xlim(x_center - plot_range / 2, x_center + plot_range / 2)
            ax.set_ylim(y_center - plot_range / 2, y_center + plot_range / 2)

            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=crs)  # ty:ignore[unresolved-attribute]
            except requests.exceptions.RequestException as exc:
                logger.warning("Failed to load tiles for basemap", error=str(exc))

            ax.set_axis_off()
            if ego_xy is not None:
                ego_xy = ego_xy.cpu()
                ax.scatter(
                    ego_xy[0],
                    ego_xy[1],
                    color="blue",
                    marker="*",
                    s=400,  # make it bigger
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=20,
                )

            return Image(_figure_to_rgba(fig), caption=caption)
        finally:
            plt.close(fig)
