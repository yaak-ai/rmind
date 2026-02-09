from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from random import randint
from typing import Annotated, Any, final

import contextily as ctx
import kornia.color as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from contextily.tile import requests
from einops import rearrange
from pydantic import AfterValidator, validate_call
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.loggers import WandbLogger
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils._pytree import MappingKey, key_get, tree_map  # noqa: PLC2701
from wandb import Image

from rmind.utils.pytree import key_get_default

logger = get_logger(__name__)


class StrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name.lower()


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


BATCH_END_HOOKS = frozenset(hook for hook in BATCH_HOOKS if hook.endswith("_batch_end"))


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


@final
class WandbImageParamLogger(Callback):
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
    ) -> None:
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
        setattr(self, when, self._call)

    def _call(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if (
            trainer.sanity_checking
            or not (
                loggers := [
                    logger_
                    for logger_ in pl_module.loggers
                    if isinstance(logger_, WandbLogger)
                ]
            )
            or not trainer.is_global_zero
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
        if (
            trainer.sanity_checking
            or not (
                loggers := [
                    logger_
                    for logger_ in pl_module.loggers
                    if isinstance(logger_, WandbLogger)
                ]
            )
            or not trainer.is_global_zero
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

        fig = plt.figure(figsize=(8, 8), frameon=False)
        wpts_x = wpts_xy_normalized[:, 0]
        wpts_y = wpts_xy_normalized[:, 1]
        plt.plot(wpts_x, wpts_y, "bo-", markersize=5)
        for i, (x, y) in enumerate(zip(wpts_x, wpts_y, strict=True)):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")
        plt.plot(0, 0, "ro")
        plt.grid(True)  # noqa: FBT003
        plt.axis("equal")
        return Image(fig, caption=caption)

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
        except requests.exceptions.ConnectionError:
            logger.warning("Failed to load tiles for basemap")

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

        return Image(fig, caption=caption)


@final
class WandbPatchSimilarityLogger(Callback):
    """Logs cosine similarity heatmaps between a random reference patch and all patches in image embeddings.

    Visualizes spatial representation quality by comparing ground truth vs predicted patch embeddings
    from a specified objective's outputs. Random patch selection occurs per image_source per logging step.
    """

    @dataclass(frozen=True, slots=True)
    class PatchCoords:
        row: int
        col: int

    @dataclass(frozen=True, slots=True)
    class VisualizationParams:
        img_np: np.ndarray | None
        h: int
        w: int
        h_patch: float
        w_patch: float

        @property
        def has_image(self) -> bool:
            return self.img_np is not None

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        image_sources: dict[str, list[str | int]],
        embeddings_predict: list[str | int],
        embeddings_target: list[str | int],
        patch_grid_size: int = 16,
        every_n_batch: int | None = None,
    ) -> None:
        self._key = key
        self._image_sources_path = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            image_sources,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._patch_grid_size = patch_grid_size
        self._embeddings_predict_path = tuple(map(MappingKey, embeddings_predict))
        self._embeddings_target_path = tuple(map(MappingKey, embeddings_target))
        _validate_every_n_batch(
            when=when, every_n_batch=every_n_batch, allowed_hooks=BATCH_END_HOOKS
        )
        self._every_n_batch = every_n_batch
        self._when = when
        self._objective_valid: bool | None = None

        setattr(self, when, self._call)

    def _call(  # noqa: PLR0914
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if (
            trainer.sanity_checking
            or not (
                loggers := [
                    logger_
                    for logger_ in pl_module.loggers
                    if isinstance(logger_, WandbLogger)
                ]
            )
            or not trainer.is_global_zero
        ):
            return

        base_hook_method = getattr(pl.Callback, self._when)
        sig = inspect.signature(base_hook_method)

        bound_args = sig.bind(self, trainer, pl_module, *args, **kwargs)
        bound_args.apply_defaults()
        batch_idx = bound_args.arguments.get("batch_idx")
        batch: dict = bound_args.arguments.get("batch")  # ty:ignore[invalid-assignment]
        outputs = bound_args.arguments.get("outputs")

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ):
            return

        if self._objective_valid is None:
            if not self._validate_outputs(
                outputs, self._embeddings_predict_path, self._embeddings_target_path
            ):
                logger.warning(
                    "Outputs don't support patch similarity logging, logger will no-op",
                    embeddings_predict_path=self._embeddings_predict_path,
                    embeddings_target_path=self._embeddings_target_path,
                )
                self._objective_valid = False
                return

            self._objective_valid = True

        if not self._objective_valid:
            return

        for image_source_key in self._image_sources_path:
            patch_coords = self.PatchCoords(
                row=randint(0, self._patch_grid_size - 1),  # noqa: S311
                col=randint(0, self._patch_grid_size - 1),  # noqa: S311
            )

            try:
                pred_emb_full = key_get(outputs, self._embeddings_predict_path)
                gt_emb_full = key_get(outputs, self._embeddings_target_path)

                pred_emb = key_get(
                    pred_emb_full, self._image_sources_path[image_source_key]
                )
                gt_emb = key_get(
                    gt_emb_full, self._image_sources_path[image_source_key]
                )

                image_tensor = batch["data"][image_source_key][0, -1]
                if image_tensor.dtype != torch.uint8:
                    image_tensor = (image_tensor * 255).to(torch.uint8)
                orig_image = image_tensor.cpu().numpy()

                gt_sim, pred_sim = self._compute_similarities(
                    gt_emb, pred_emb, patch_coords
                )
                fig = self._create_heatmap_figure(
                    gt_sim, pred_sim, orig_image, patch_coords
                )

                for logger_ in loggers:
                    logger_.log_image(
                        key=f"{self._key} | {image_source_key}",
                        images=[Image(fig)],
                        step=trainer.global_step,
                    )

                plt.close(fig)

            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(
                    "Failed to generate patch similarity visualization for image_source %s",
                    image_source_key,
                    error=str(e),
                    exc_info=True,
                )

    @staticmethod
    def _validate_outputs(
        outputs: Any,
        embeddings_predict_path: tuple[MappingKey, ...],
        embeddings_target_path: tuple[MappingKey, ...],
    ) -> bool:
        try:
            key_get(outputs, embeddings_predict_path)  # ty:ignore[invalid-argument-type]
            key_get(outputs, embeddings_target_path)  # ty:ignore[invalid-argument-type]
        except (TypeError, KeyError, AttributeError):
            return False
        else:
            return True

    def _compute_similarities(
        self, gt_tensor: Tensor, pred_tensor: Tensor, patch_coords: PatchCoords
    ) -> tuple[np.ndarray, np.ndarray]:
        gt_tensor = gt_tensor.detach().float()
        pred_tensor = pred_tensor.detach().float()

        ref_patch_idx = patch_coords.row * self._patch_grid_size + patch_coords.col

        return (
            cosine_similarity(gt_tensor[ref_patch_idx], gt_tensor, dim=-1)
            .cpu()
            .numpy()
            .reshape(self._patch_grid_size, self._patch_grid_size)
        ), (
            cosine_similarity(pred_tensor[ref_patch_idx], pred_tensor, dim=-1)
            .cpu()
            .numpy()
            .reshape(self._patch_grid_size, self._patch_grid_size)
        )

    def _create_heatmap_figure(
        self,
        gt_sim: np.ndarray,
        pred_sim: np.ndarray,
        orig_image: np.ndarray | None,
        patch_coords: PatchCoords,
    ) -> plt.Figure:
        if orig_image is not None:
            img_np = orig_image
            h, w = orig_image.shape[:2]
        else:
            img_np = None
            h, w = 12 * self._patch_grid_size, 12 * self._patch_grid_size

        h_patch, w_patch = (
            (h / self._patch_grid_size, w / self._patch_grid_size)
            if img_np is not None
            else (1.0, 1.0)
        )

        params = self.VisualizationParams(
            img_np=img_np, h=h, w=w, h_patch=h_patch, w_patch=w_patch
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        im1 = self._plot_heatmap(ax1, gt_sim, params, "Ground Truth", patch_coords)
        im2 = self._plot_heatmap(ax2, pred_sim, params, "Predicted", patch_coords)

        for ax, similarity_grid in [(ax1, gt_sim), (ax2, pred_sim)]:
            self._add_grid_annotations(ax, similarity_grid, params)
            self._highlight_reference_patch(ax, params, patch_coords)
            self._draw_patch_grid(ax, params)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, label="Cosine Similarity")
        plt.colorbar(im2, ax=ax2, label="Cosine Similarity")
        plt.tight_layout()

        return fig

    def _plot_heatmap(
        self,
        ax: plt.Axes,
        sim: np.ndarray,
        params: VisualizationParams,
        title_prefix: str,
        patch_coords: PatchCoords,
    ) -> Any:
        if params.has_image:
            ax.imshow(params.img_np, alpha=1.0)  # ty:ignore[invalid-argument-type]

        im = ax.imshow(
            sim,
            cmap="viridis",
            alpha=0.55 if params.has_image else 0.7,
            vmin=0,
            vmax=1,
            extent=(
                (0, params.w, params.h, 0)
                if params.has_image
                else (0, self._patch_grid_size, self._patch_grid_size, 0)
            ),
            interpolation="nearest",
        )
        ax.set_title(
            f"{title_prefix} Cosine Similarity - Patch [{patch_coords.row},{patch_coords.col}]"
        )
        return im

    def _add_grid_annotations(
        self, ax: plt.Axes, similarity_grid: np.ndarray, params: VisualizationParams
    ) -> None:
        if params.has_image:
            for col in range(self._patch_grid_size):
                col_x = (col + 0.5) * params.w_patch
                ax.text(
                    col_x,
                    params.h + 0.7 * params.h_patch,
                    str(col),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="midnightblue",
                    fontweight="bold",
                    clip_on=False,
                )
            for row in range(self._patch_grid_size):
                row_y = (row + 0.5) * params.h_patch
                ax.text(
                    -0.7 * params.w_patch,
                    row_y,
                    str(row),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="midnightblue",
                    fontweight="bold",
                    clip_on=False,
                )

        for row in range(self._patch_grid_size):
            for col in range(self._patch_grid_size):
                val = similarity_grid[row, col]
                text_color = "white" if val < 0.5 else "black"  # noqa: PLR2004
                center_x = (col + 0.5) * params.w_patch if params.has_image else col
                center_y = (row + 0.5) * params.h_patch if params.has_image else row
                ax.text(
                    center_x,
                    center_y,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    def _highlight_reference_patch(  # noqa: PLR6301
        self, ax: plt.Axes, params: VisualizationParams, patch_coords: PatchCoords
    ) -> None:
        rect_x = (
            patch_coords.col * params.w_patch
            if params.has_image
            else (patch_coords.col - 0.5)
        )
        rect_y = (
            patch_coords.row * params.h_patch
            if params.has_image
            else (patch_coords.row - 0.5)
        )
        ax.add_patch(
            plt.Rectangle(
                (rect_x, rect_y),
                params.w_patch,
                params.h_patch,
                fill=False,
                edgecolor="black",
                linewidth=3,
            )
        )

    def _draw_patch_grid(self, ax: plt.Axes, params: VisualizationParams) -> None:
        if params.has_image:
            for g in range(1, self._patch_grid_size):
                ax.axhline(g * params.h_patch, color="gray", lw=0.7, alpha=0.3)
                ax.axvline(g * params.w_patch, color="gray", lw=0.7, alpha=0.3)
            ax.set_xlim((0, params.w))
            ax.set_ylim((params.h, 0))
        else:
            ax.set_xlim((-0.5, self._patch_grid_size - 0.5))
            ax.set_ylim((self._patch_grid_size - 0.5, -0.5))

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
