import inspect
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Annotated, Any, final

import contextily as ctx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from einops import rearrange
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


@final
class WandbWaypointsLogger(Callback):
    in_batch_idx: int = 0
    in_clip_idx: int = -1
    map_zoom_factor: float = 2.5

    def __init__(
        self,
        *,
        select: dict[str, str | tuple[str, ...]],
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        every_n_batch: int | None = None,
    ) -> None:
        self._key = key
        if every_n_batch is not None and when not in BATCH_HOOKS:
            msg = (
                "`every_n_batch` is only supported for batch-based hooks: "
                + ", ".join(f"`{hook}`" for hook in BATCH_HOOKS)
                + f". Got `{when}`"
            )
            raise ValueError(msg)
        self._every_n_batch = every_n_batch
        self._when = when
        self._select = {k: tuple(v) for k, v in select.items()}
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
        batch = bound_args.arguments.get("batch")

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ) or (trainer.current_epoch != 0):
            return

        batch = TensorDict(batch).auto_batch_size_()[self.in_batch_idx]
        caption = []
        if "input_id" in self._select:
            caption.append(
                batch.get(self._select["input_id"]).data[self.in_batch_idx].item()
            )
        if "time_stamp" in self._select:
            time_stamp = (
                batch.get_at(self._select["time_stamp"], self.in_clip_idx).cpu().item()
            )
            caption.append(
                datetime.fromtimestamp(time_stamp * 1e-6, tz=UTC).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        if "frame_idx" in self._select:
            frame_idx = (
                batch.get_at(self._select["frame_idx"], self.in_clip_idx).cpu().item()
            )
            caption.append(f"frame_idx: {frame_idx}")
        caption = " | ".join(caption)

        log_images = []
        if "image" in self._select:
            log_images.append(
                Image(
                    rearrange(
                        batch.get_at(self._select["image"], self.in_clip_idx),
                        "w h c -> c w h",
                    ),
                    caption=caption,
                )
            )
        if "waypoints_normalized" in self._select:
            log_images.append(self._plot_waypoints_normalized(batch, caption))
        if "waypoints_gps" in self._select:
            log_images.append(self._plot_waypoints_gps(batch, caption))

        for logger in loggers:
            logger.log_image(key=self._key, images=log_images, step=trainer.global_step)

    def _plot_waypoints_normalized(
        self, batch: TensorDict, caption: str | None = None
    ) -> Image:
        wpts_normalized = batch.get_at(
            self._select["waypoints_normalized"], self.in_clip_idx
        ).cpu()

        fig = plt.figure(figsize=(8, 8))
        wpts_x = wpts_normalized[:, 0]
        wpts_y = wpts_normalized[:, 1]
        plt.plot(wpts_x, wpts_y, "bo-", label="Waypoints", markersize=5)  # type: ignore[reportUnknownReturnType]
        for i, (x, y) in enumerate(zip(wpts_x, wpts_y, strict=True)):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")  # type: ignore[reportUnknownReturnType]
        plt.grid(True)  # noqa: FBT003
        plt.axis("equal")  # type: ignore[reportUnknownReturnType]
        return Image(fig, caption=caption)

    def _plot_waypoints_gps(
        self, batch: TensorDict, caption: str | None = None
    ) -> Image:
        wpts_gps = batch.get_at(self._select["waypoints_gps"], self.in_clip_idx).cpu()
        fig, ax = plt.subplots(figsize=(8, 8), frameon=False)
        ax.scatter(
            wpts_gps[:, 0],
            wpts_gps[:, 1],
            color="deeppink",
            s=150,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )  # type: ignore[reportUnknownReturnType]

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_center = x_min + x_range / 2
        y_center = y_min + y_range / 2

        plot_range = max(x_range, y_range) * self.map_zoom_factor

        ax.set_xlim(x_center - plot_range / 2, x_center + plot_range / 2)  # type: ignore[reportUnknownReturnType]
        ax.set_ylim(y_center - plot_range / 2, y_center + plot_range / 2)  # type: ignore[reportUnknownReturnType]

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)  # type: ignore[reportUnknownReturnType]
        ax.set_axis_off()
        if "ego_gps" in self._select:
            ego_gps = batch.get_at(self._select["ego_gps"], self.in_clip_idx).cpu()
            ax.scatter(
                ego_gps[0],
                ego_gps[1],
                color="blue",
                marker="*",
                s=400,  # make it bigger
                edgecolor="black",
                linewidth=1.5,
                zorder=20,
            )  # type: ignore[reportUnknownReturnType]

        return Image(fig, caption=caption)
