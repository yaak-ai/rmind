import inspect
from collections.abc import Callable, Sequence
from enum import StrEnum, auto
from random import randint
from typing import Annotated, Any, Literal, final

import contextily as ctx
import kornia.color as K
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from contextily.tile import requests
from einops import rearrange
from matplotlib import cm
from pydantic import AfterValidator, validate_call
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.hooks import ModelHooks
from pytorch_lightning.loggers import WandbLogger
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils._pytree import MappingKey, key_get, tree_map  # noqa: PLC2701

from rmind.utils.pytree import key_get_default
from wandb import Image

logger = get_logger(__name__)


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


NoneKey = (MappingKey(None),)


@final
class WandbWaypointsLogger(Callback):
    class DataColumns(StrEnum):
        IMAGE = auto()
        WAYPOINTS_XY_NORMALIZED = auto()
        WAYPOINTS_XY = auto()
        EGO_XY = auto()

    def __init__(  # noqa: PLR0913
        self,
        *,
        data: dict[DataColumns, list[str]],
        caption: dict[str, list[str]],
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        every_n_batch: int | None = None,
        crs: str | None = None,
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
    class Cameras(StrEnum):
        CAM_FRONT_LEFT = auto()

    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        patch_coords: tuple[int, int] | Literal["random"],
        data: dict[Cameras, list[str]],
        objective_name: str = "forward_dynamics",
        patch_grid_size: int = 16,
        every_n_batch: int | None = None,
    ) -> None:
        self._key = key
        self._patch_coords = patch_coords
        self._patch_i: int | None = None
        self._patch_j: int | None = None

        # Validate and set patch coordinates
        if isinstance(patch_coords, tuple):
            if len(patch_coords) != 2:  # noqa: PLR2004
                msg = f"patch_coords tuple must have length 2, got {len(patch_coords)}"
                raise ValueError(msg)
            i, j = patch_coords
            if not (0 <= i < patch_grid_size and 0 <= j < patch_grid_size):
                msg = (
                    f"patch_coords ({i}, {j}) out of bounds for "
                    f"grid size {patch_grid_size}"
                )
                raise ValueError(msg)
            self._patch_i, self._patch_j = patch_coords
        elif patch_coords != "random":
            msg = f"patch_coords must be tuple or 'random', got {patch_coords!r}"
            raise ValueError(msg)

        self._data_path = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            data,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._objective_name = objective_name
        self._patch_grid_size = patch_grid_size
        self._every_n_batch = every_n_batch
        self._when = when

        if every_n_batch is not None and when not in BATCH_HOOKS:
            msg = (
                "`every_n_batch` is only supported for batch-based hooks: "
                + ", ".join(f"`{hook}`" for hook in BATCH_HOOKS)
                + f". Got `{when}`"
            )
            raise ValueError(msg)

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
        batch: dict = bound_args.arguments.get("batch")  # ty:ignore[invalid-assignment]

        if (
            (self._every_n_batch is not None)
            and (batch_idx is not None)
            and (batch_idx % self._every_n_batch != 0)
        ):
            return

        # Check if objective exists and has required attributes
        objective = pl_module.objectives.get(self._objective_name)  # ty:ignore[call-non-callable, possibly-missing-attribute]
        if objective is None:
            logger.warning(
                "Objective not found in model, skipping patch similarity logging",
                objective_name=self._objective_name,
            )
            return

        if not self._validate_objective(objective):
            logger.warning(
                "Objective doesn't support patch similarity logging",
                objective_name=self._objective_name,
                required_attrs=["_last_embeddings", "_last_targets"],
            )
            return

        for camera_key in self.Cameras:
            if self._patch_coords == "random":
                self._patch_i = randint(0, self._patch_grid_size - 1)  # noqa: S311
                self._patch_j = randint(0, self._patch_grid_size - 1)  # noqa: S311
            try:
                # Extract embeddings
                pred_emb = key_get(
                    objective._last_embeddings,  # noqa: SLF001
                    self._data_path.get(camera_key),
                )
                gt_emb = key_get(
                    objective._last_targets,  # noqa: SLF001
                    self._data_path.get(camera_key),
                )
                orig_image = self._get_image_from_batch(batch, camera_key)

                gt_sim, pred_sim = self._compute_similarities(gt_emb, pred_emb)

                fig = self._create_heatmap_figure(gt_sim, pred_sim, orig_image)

                for logger_ in loggers:
                    logger_.log_image(
                        key=self._key + f" | {camera_key}",
                        images=[Image(fig)],
                        step=trainer.global_step,
                    )

                plt.close(fig)

            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(
                    "Failed to generate patch similarity visualization for camera %s",
                    camera_key,
                    error=str(e),
                    exc_info=True,
                )

    @staticmethod
    def _validate_objective(objective: Any) -> bool:
        """Check if objective has required attributes for logging."""
        return hasattr(objective, "_last_embeddings") and hasattr(
            objective, "_last_targets"
        )

    @staticmethod
    def _get_image_from_batch(
        batch: dict[str, Any], camera_key: Cameras
    ) -> np.ndarray | None:
        """Extract original image from batch for visualization."""
        try:
            image_tensor = batch["data"][camera_key]
            image = image_tensor[0, -1].cpu().numpy()
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(
                "Failed to extract image from batch",
                camera_key=camera_key,
                error=str(e),
            )
            return None
        else:
            return image

    def _compute_similarities(
        self, gt_tensor: Tensor, pred_tensor: Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cosine similarities between reference patch and all patches."""
        gt_tensor = gt_tensor.detach().cpu().float()
        pred_tensor = pred_tensor.detach().cpu().float()

        assert self._patch_i is not None  # noqa: S101
        assert self._patch_j is not None  # noqa: S101
        ref_patch_idx = self._patch_i * self._patch_grid_size + self._patch_j

        ref_patch_gt = gt_tensor[ref_patch_idx]
        ref_patch_pred = pred_tensor[ref_patch_idx]

        gt_similarity_list = cosine_similarity(ref_patch_gt, gt_tensor, dim=-1)
        pred_similarity_list = cosine_similarity(ref_patch_pred, pred_tensor, dim=-1)

        gt_similarity_grid = np.array(gt_similarity_list).reshape(
            self._patch_grid_size, self._patch_grid_size
        )
        pred_similarity_grid = np.array(pred_similarity_list).reshape(
            self._patch_grid_size, self._patch_grid_size
        )

        return gt_similarity_grid, pred_similarity_grid

    def _create_heatmap_figure(
        self, gt_sim: np.ndarray, pred_sim: np.ndarray, orig_image: np.ndarray | None
    ) -> plt.Figure:
        """Create side-by-side heatmap visualization."""
        img_np, h, w = self._prepare_image(orig_image)
        h_patch, w_patch = self._calculate_patch_dimensions(h, w, img_np)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        im1 = self._plot_heatmap(ax1, gt_sim, img_np, h, w, "Ground Truth")
        im2 = self._plot_heatmap(ax2, pred_sim, img_np, h, w, "Predicted")

        for ax, similarity_grid in [(ax1, gt_sim), (ax2, pred_sim)]:
            self._add_grid_annotations(
                ax, similarity_grid, img_np, h, w, h_patch, w_patch
            )
            self._highlight_reference_patch(ax, img_np, h_patch, w_patch)
            self._draw_patch_grid(ax, img_np, h, w, h_patch, w_patch)

        # Add colorbars
        plt.colorbar(im1, ax=ax1, label="Cosine Similarity")
        plt.colorbar(im2, ax=ax2, label="Cosine Similarity")
        plt.tight_layout()

        return fig

    def _prepare_image(
        self, orig_image: np.ndarray | None
    ) -> tuple[np.ndarray | None, int, int]:
        if orig_image is not None:
            if hasattr(orig_image, "convert"):
                img_np = np.array(orig_image.convert("RGB"))  # ty:ignore[call-non-callable]
            else:
                img_np = np.array(orig_image)
            h, w = img_np.shape[:2]
        else:
            img_np = None
            h, w = 12 * self._patch_grid_size, 12 * self._patch_grid_size
        return img_np, h, w

    def _calculate_patch_dimensions(
        self, h: int, w: int, img_np: np.ndarray | None
    ) -> tuple[float, float]:
        if img_np is not None:
            return h / self._patch_grid_size, w / self._patch_grid_size
        return 1.0, 1.0

    def _plot_heatmap(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        sim: np.ndarray,
        img_np: np.ndarray | None,
        h: int,
        w: int,
        title_prefix: str,
    ) -> Any:
        if img_np is not None:
            ax.imshow(img_np, alpha=1.0)

        im = ax.imshow(
            sim,
            cmap="viridis",
            alpha=0.55 if img_np is not None else 0.7,
            vmin=0,
            vmax=1,
            extent=(
                (0, w, h, 0)
                if img_np is not None
                else (0, self._patch_grid_size, self._patch_grid_size, 0)
            ),
            interpolation="nearest",
        )
        assert self._patch_i is not None  # noqa: S101
        assert self._patch_j is not None  # noqa: S101
        ax.set_title(
            f"{title_prefix} Cosine Similarity - Patch [{self._patch_i},{self._patch_j}]"
        )
        return im

    def _add_grid_annotations(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        similarity_grid: np.ndarray,
        img_np: np.ndarray | None,
        h: int,
        w: int,  # noqa: ARG002
        h_patch: float,
        w_patch: float,
    ) -> None:
        if img_np is not None:
            for col in range(self._patch_grid_size):
                col_x = (col + 0.5) * w_patch
                ax.text(
                    col_x,
                    h + 0.7 * h_patch,
                    str(col),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="midnightblue",
                    fontweight="bold",
                    clip_on=False,
                )
            for row in range(self._patch_grid_size):
                row_y = (row + 0.5) * h_patch
                ax.text(
                    -0.7 * w_patch,
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
                center_x = (col + 0.5) * w_patch if img_np is not None else col
                center_y = (row + 0.5) * h_patch if img_np is not None else row
                ax.text(
                    center_x,
                    center_y,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    def _highlight_reference_patch(
        self, ax: plt.Axes, img_np: np.ndarray | None, h_patch: float, w_patch: float
    ) -> None:
        assert self._patch_i is not None  # noqa: S101
        assert self._patch_j is not None  # noqa: S101
        rect_x = (
            self._patch_j * w_patch if img_np is not None else (self._patch_j - 0.5)
        )
        rect_y = (
            self._patch_i * h_patch if img_np is not None else (self._patch_i - 0.5)
        )
        rect_width = w_patch
        rect_height = h_patch
        ax.add_patch(
            plt.Rectangle(
                (rect_x, rect_y),
                rect_width,
                rect_height,
                fill=False,
                edgecolor="black",
                linewidth=3,
            )
        )

    def _draw_patch_grid(  # noqa: PLR0913, PLR0917
        self,
        ax: plt.Axes,
        img_np: np.ndarray | None,
        h: int,
        w: int,
        h_patch: float,
        w_patch: float,
    ) -> None:
        if img_np is not None:
            for g in range(1, self._patch_grid_size):
                ax.axhline(g * h_patch, color="gray", lw=0.7, alpha=0.3)
                ax.axvline(g * w_patch, color="gray", lw=0.7, alpha=0.3)
            ax.set_xlim((0, w))
            ax.set_ylim((h, 0))
        else:
            ax.set_xlim((-0.5, self._patch_grid_size - 0.5))
            ax.set_ylim((self._patch_grid_size - 0.5, -0.5))

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
