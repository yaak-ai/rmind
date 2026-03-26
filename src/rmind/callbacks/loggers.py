import hashlib
import inspect
from collections.abc import Callable, Sequence
from enum import StrEnum, auto
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
        last_input = getattr(
            getattr(pl_module, "episode_builder", None), "_last_input", None
        )
        if last_input is not None:
            wpts_xy_normalized = last_input["context"]["waypoints"][0, -1]
        else:
            wpts_xy_normalized = key_get_default(
                batch,
                self._data_paths.get(self.DataColumns.WAYPOINTS_XY_NORMALIZED, NoneKey),
                None,
            )
        if wpts_xy_normalized is not None:
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
        try:
            wpts_x = wpts_xy_normalized[:, 0]
            wpts_y = wpts_xy_normalized[:, 1]
            plt.plot(wpts_x, wpts_y, "bo-", markersize=5)
            for i, (x, y) in enumerate(zip(wpts_x, wpts_y, strict=True)):
                plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")
            plt.plot(0, 0, "ro")
            plt.grid(True)  # noqa: FBT003
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            return Image(fig, caption=caption)
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
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        when: Annotated[str, AfterValidator(_validate_hook)],
        key: str,
        image_sources: dict[str, list[str | int]],
        embeddings_predict: list[str | int],
        embeddings_target: list[str | int],
        sample_id_path: list[str | int],
        every_n_sample: int,
        patch_grid_size: int = 16,
    ) -> None:
        self._key = key
        self._image_sources_path = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            image_sources,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._patch_grid_size = patch_grid_size
        self._num_patches = patch_grid_size * patch_grid_size
        self._embeddings_predict_path = tuple(map(MappingKey, embeddings_predict))
        self._embeddings_target_path = tuple(map(MappingKey, embeddings_target))
        self._sample_id_path = tuple(map(MappingKey, sample_id_path))
        self._every_n_sample = every_n_sample
        self._when = when

        setattr(self, when, self._call)

    @staticmethod
    def _stable_ref_patch_idx(
        sample_id: int, image_source_key: str, num_patches: int
    ) -> int:
        digest = hashlib.sha256(f"{sample_id}:{image_source_key}".encode()).digest()
        return int.from_bytes(digest[:4]) % num_patches

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
        batch: dict = bound_args.arguments.get("batch")  # ty:ignore[invalid-assignment]
        outputs = bound_args.arguments.get("outputs")

        sample_ids = key_get(batch, self._sample_id_path)
        mask = sample_ids % self._every_n_sample == 0
        if not mask.any():
            return

        matching_indices = mask.nonzero(as_tuple=False).flatten()
        pred_emb_full = key_get(outputs, self._embeddings_predict_path)
        gt_emb_full = key_get(outputs, self._embeddings_target_path)

        for local_idx in matching_indices:
            sample_id = sample_ids[local_idx].item()
            for image_source_key in self._image_sources_path:
                ref_patch_idx = self._stable_ref_patch_idx(
                    sample_id, image_source_key, self._num_patches
                )

                pred_emb = key_get(
                    pred_emb_full, self._image_sources_path[image_source_key]
                )[local_idx, -1]
                gt_emb = key_get(
                    gt_emb_full, self._image_sources_path[image_source_key]
                )[local_idx, -1]

                image_tensor = batch["data"][image_source_key][local_idx, -1]
                if image_tensor.dtype != torch.uint8:
                    image_tensor = (image_tensor * 255).to(torch.uint8)
                orig_image = image_tensor.cpu().numpy()

                gt_sim, pred_sim = self._compute_similarities(
                    gt_emb, pred_emb, ref_patch_idx, self._patch_grid_size
                )

                gt_image = self._create_heatmap_image(
                    gt_sim,
                    orig_image,
                    "Ground Truth",
                    ref_patch_idx,
                    self._patch_grid_size,
                )
                pred_image = self._create_heatmap_image(
                    pred_sim,
                    orig_image,
                    "Predicted",
                    ref_patch_idx,
                    self._patch_grid_size,
                )

                for logger_ in loggers:
                    logger_.log_image(
                        key=f"{self._key}/{image_source_key}/sample_{sample_id}",
                        images=[Image(gt_image), Image(pred_image)],
                        step=trainer.global_step,
                    )

    @staticmethod
    def _compute_similarities(
        gt_tensor: Tensor, pred_tensor: Tensor, ref_patch_idx: int, grid_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        gt_tensor = gt_tensor.detach().float()
        pred_tensor = pred_tensor.detach().float()

        return (
            cosine_similarity(gt_tensor[ref_patch_idx], gt_tensor, dim=-1)
            .cpu()
            .numpy()
            .reshape(grid_size, grid_size)
        ), (
            cosine_similarity(pred_tensor[ref_patch_idx], pred_tensor, dim=-1)
            .cpu()
            .numpy()
            .reshape(grid_size, grid_size)
        )

    @staticmethod
    def _create_heatmap_image(
        sim: np.ndarray,
        orig_image: np.ndarray,
        title: str,
        ref_patch_idx: int,
        patch_grid_size: int,
    ) -> np.ndarray:
        h, w = orig_image.shape[:2]
        h_patch = h / patch_grid_size
        w_patch = w / patch_grid_size

        fig, ax = plt.subplots(figsize=(12, 12))
        try:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            ax.imshow(orig_image, alpha=1.0)
            ax.imshow(
                sim,
                cmap="viridis",
                alpha=0.55,
                vmin=0,
                vmax=1,
                extent=(0, w, h, 0),
                interpolation="nearest",
            )
            ax.set_title(title, color="white")
            ax.axis("off")

            for col in range(patch_grid_size):
                ax.text(
                    (col + 0.5) * w_patch,
                    h + 0.7 * h_patch,
                    str(col),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="midnightblue",
                    fontweight="bold",
                    clip_on=False,
                )
            for row in range(patch_grid_size):
                ax.text(
                    -0.7 * w_patch,
                    (row + 0.5) * h_patch,
                    str(row),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="midnightblue",
                    fontweight="bold",
                    clip_on=False,
                )

            # Similarity values
            for row in range(patch_grid_size):
                for col in range(patch_grid_size):
                    val = sim[row, col]
                    ax.text(
                        (col + 0.5) * w_patch,
                        (row + 0.5) * h_patch,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val < 0.5 else "black",  # noqa: PLR2004
                    )

            # Highlight reference patch
            ref_row, ref_col = divmod(ref_patch_idx, patch_grid_size)
            ax.add_patch(
                plt.Rectangle(
                    (ref_col * w_patch, ref_row * h_patch),
                    w_patch,
                    h_patch,
                    fill=False,
                    edgecolor="black",
                    linewidth=3,
                )
            )

            # Patch grid lines
            for g in range(1, patch_grid_size):
                ax.axhline(g * h_patch, color="gray", lw=0.7, alpha=0.3)
                ax.axvline(g * w_patch, color="gray", lw=0.7, alpha=0.3)
            ax.set_xlim((0, w))
            ax.set_ylim((h, 0))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            fig.tight_layout(pad=0)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.canvas.draw()

            return np.asarray(fig.canvas.buffer_rgba()).copy()  # ty:ignore[unresolved-attribute]
        finally:
            plt.close(fig)
