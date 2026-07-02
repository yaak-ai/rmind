import hashlib
from typing import Annotated, Any, final

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import AfterValidator, validate_call
from torch import Tensor
from torch.nn.functional import cosine_similarity
from torch.utils._pytree import MappingKey, key_get, tree_map  # noqa: PLC2701

from rmind.callbacks.safe import SafeCallback
from wandb import Image

from .common import (
    _bind_hook_arguments,
    _figure_to_rgba,
    _get_wandb_loggers,
    _validate_hook,
)


@final
class WandbPatchSimilarityLogger(SafeCallback):
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
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
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

        setattr(self, when, self._safe_hook(when, self._call))

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
        loggers = _get_wandb_loggers(pl_module)
        if trainer.sanity_checking or not loggers or not trainer.is_global_zero:
            return

        bound_args = _bind_hook_arguments(
            self, self._when, trainer, pl_module, *args, **kwargs
        )
        batch: dict = bound_args.get("batch")  # ty:ignore[invalid-assignment]
        outputs = bound_args.get("outputs")

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
            return _figure_to_rgba(fig)
        finally:
            plt.close(fig)
