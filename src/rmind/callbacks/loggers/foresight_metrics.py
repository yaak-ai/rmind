"""Validation-time foresight prediction metrics.

Logs two single-scalar quantities per epoch (one per configured image
source) that summarise how well the forward-dynamics head is predicting
the next-frame image embedding at motion-rich locations:

- ``{key}/{src}/r2_hot`` -- linear-probe R²(pred → target) on the
  top-{hot_quantile} highest-motion patches at the deepest FD position
  (predicting img@T-1). Drops if the model collapses to autoencoding.

- ``{key}/{src}/search_hit_hot`` -- cross-patch search accuracy on the
  same hot patches: for each pred[p], fraction of the time
  ``argmin_q ||pred[p] - target[q]||²`` equals ``p`` (chance ~1/256).
  Complementary to R²; satisfied only by absolute spatial commitment.

Motivation, definitions, and the offline-probe results that justify this
pair of metrics are written up in the foresight investigation:
https://app.notion.com/p/Foresight-in-the-rmind-Control-Transformer-Does-the-Unsupervised-Last-Position-Predict-the-Future-37bd658ccf8780a3b298cc7c45304af2

Notes:
- We measure at the deepest FD position the artifacts buffer exposes
  (``last_embeddings[..., -1]``, which predicts img@T-1). The
  unsupervised T-1 slot the policy actually reads at serving is one
  step further; per the investigation linked above, the two positions
  behave near-identically.
- Stratification is computed from the temporal delta between the last
  two target frames in the FD artifacts buffer (img@T-2 vs img@T-1), so
  the callback needs no extra forward passes.
- Stored buffers are capped at ``max_hot_pairs`` to bound CPU memory.
  Beyond the cap, additional batches are skipped for the R² fit; the
  search counter keeps accumulating.
"""

import math
from typing import Annotated, Any, final

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from torch import Tensor
from torch.utils._pytree import MappingKey, key_get, tree_map  # noqa: PLC2701

from rmind.callbacks.safe import SafeCallback

from .common import _get_wandb_loggers


@final
class WandbForesightMetricsLogger(SafeCallback):
    """See module docstring."""

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str,
        image_sources: dict[str, list[str | int]],
        embeddings_predict: list[str | int],
        embeddings_target: list[str | int],
        hot_quantile: Annotated[float, "fraction in (0, 1]"] = 0.05,
        max_hot_pairs: int = 50_000,
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(
            fail_gracefully=fail_gracefully, disable_on_error=disable_on_error
        )
        if not 0.0 < hot_quantile <= 1.0:
            msg = f"hot_quantile must be in (0, 1], got {hot_quantile}"
            raise ValueError(msg)
        self._key = key
        self._image_sources_path = tree_map(
            lambda v: tuple(map(MappingKey, v)),
            image_sources,
            is_leaf=lambda x: isinstance(x, list),
        )
        self._embeddings_predict_path = tuple(map(MappingKey, embeddings_predict))
        self._embeddings_target_path = tuple(map(MappingKey, embeddings_target))
        self._hot_quantile = float(hot_quantile)
        self._max_hot_pairs = int(max_hot_pairs)

        # Populated per epoch.
        self._pred_buf: dict[str, list[Tensor]] = {}
        self._target_buf: dict[str, list[Tensor]] = {}
        self._n_pairs: dict[str, int] = {}
        self._search_hits: dict[str, int] = {}
        self._search_total: dict[str, int] = {}

    def _reset(self) -> None:
        srcs = list(self._image_sources_path)
        self._pred_buf = {src: [] for src in srcs}
        self._target_buf = {src: [] for src in srcs}
        self._n_pairs = dict.fromkeys(srcs, 0)
        self._search_hits = dict.fromkeys(srcs, 0)
        self._search_total = dict.fromkeys(srcs, 0)

    @staticmethod
    def _gather_hot(
        pred: Tensor, tgt: Tensor, tgt_prev: Tensor, n_hot: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """pred, tgt, tgt_prev: (B, P, d). Returns (pred_hot, tgt_hot, hot_idx)."""
        delta = ((tgt - tgt_prev) ** 2).sum(dim=-1)
        sorted_idx = delta.argsort(dim=-1, descending=True)
        hot_idx = sorted_idx[:, :n_hot]
        b_idx = torch.arange(pred.shape[0], device=pred.device)[:, None]
        return pred[b_idx, hot_idx], tgt[b_idx, hot_idx], hot_idx

    @staticmethod
    def _cross_patch_hits_at_hot(
        pred: Tensor, tgt: Tensor, hot_idx: Tensor
    ) -> tuple[int, int]:
        """For each hot patch p in pred, check if argmin_q ||pred[p] - tgt[q]||² == p.

        Returns (n_hits, n_total).
        """
        pred_sq = (pred**2).sum(dim=-1, keepdim=True)  # (B, P, 1)
        tgt_sq = (tgt**2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (B, 1, P)
        inner = torch.einsum("bpd,bqd->bpq", pred, tgt)  # (B, P, P)
        dists = pred_sq + tgt_sq - 2 * inner  # (B, P, P)
        argmin_q = dists.argmin(dim=-1)  # (B, P)
        p_range = torch.arange(pred.shape[1], device=pred.device)
        hit_per_patch = (argmin_q == p_range).to(torch.bool)  # (B, P)
        hit_at_hot = hit_per_patch.gather(1, hot_idx)  # (B, n_hot)
        return int(hit_at_hot.sum().item()), int(hit_at_hot.numel())

    @staticmethod
    def _fit_r2(x: Tensor, y: Tensor) -> float:
        """Linear-probe R²(x → y) via lstsq with bias column, scalar SS form.

        Mirrors klindtlab/lejepa-identifiability's ``bidirectional_r2``. The
        fit is in-sample (no train/test split) — interpretable only when
        ``n_samples`` is well above ``n_features`` (rule of thumb: n/d ≳ 50).
        """
        if x.shape[0] <= x.shape[1] + 1:
            return float("nan")
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)
        w = torch.linalg.lstsq(x_aug, y).solution
        ss_res = ((y - x_aug @ w) ** 2).sum()
        ss_tot = ((y - y.mean(0)) ** 2).sum()
        return float((1 - ss_res / (ss_tot + 1e-12)).item())

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._safe_call(
            "on_validation_epoch_start",
            self._on_validation_epoch_start,
            trainer,
            pl_module,
        )

    def on_validation_batch_end(  # noqa: PLR0913, PLR0917
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._safe_call(
            "on_validation_batch_end",
            self._on_validation_batch_end,
            trainer,
            pl_module,
            outputs,
            batch,
            batch_idx,
            dataloader_idx,
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._safe_call(
            "on_validation_epoch_end", self._on_validation_epoch_end, trainer, pl_module
        )

    @torch.no_grad()
    def _on_validation_epoch_start(
        self,
        trainer: pl.Trainer,  # noqa: ARG002
        pl_module: pl.LightningModule,  # noqa: ARG002
    ) -> None:
        self._reset()

    @torch.no_grad()
    def _on_validation_batch_end(  # noqa: PLR0913, PLR0917
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        if trainer.sanity_checking or not _get_wandb_loggers(pl_module):
            return
        if not trainer.is_global_zero:
            return

        try:
            pred_full = key_get(outputs, self._embeddings_predict_path)
            tgt_full = key_get(outputs, self._embeddings_target_path)
        except (KeyError, IndexError, TypeError):
            return

        for src, path in self._image_sources_path.items():
            try:
                pred_seq: Tensor = key_get(pred_full, path)
                tgt_seq: Tensor = key_get(tgt_full, path)
            except (KeyError, IndexError, TypeError):
                continue
            if pred_seq.ndim != 4 or pred_seq.shape[1] < 2:  # noqa: PLR2004
                continue

            pred = pred_seq[:, -1].detach().float()
            tgt = tgt_seq[:, -1].detach().float()
            tgt_prev = tgt_seq[:, -2].detach().float()

            n_patches = pred.shape[1]
            n_hot = max(1, int(self._hot_quantile * n_patches))
            pred_hot, tgt_hot, hot_idx = self._gather_hot(pred, tgt, tgt_prev, n_hot)

            if self._n_pairs[src] < self._max_hot_pairs:
                d = pred_hot.shape[-1]
                self._pred_buf[src].append(pred_hot.reshape(-1, d).cpu())
                self._target_buf[src].append(tgt_hot.reshape(-1, d).cpu())
                self._n_pairs[src] += pred_hot.reshape(-1, d).shape[0]

            hits, total = self._cross_patch_hits_at_hot(pred, tgt, hot_idx)
            self._search_hits[src] += hits
            self._search_total[src] += total

    @torch.no_grad()
    def _on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not _get_wandb_loggers(pl_module):
            return
        if not trainer.is_global_zero:
            return

        for src in self._image_sources_path:
            r2 = float("nan")
            if self._pred_buf[src]:
                x = torch.cat(self._pred_buf[src], dim=0)
                y = torch.cat(self._target_buf[src], dim=0)
                r2 = self._fit_r2(x, y)
            search_acc = (
                self._search_hits[src] / self._search_total[src]
                if self._search_total[src] > 0
                else float("nan")
            )
            # Skip NaN epochs (e.g. too few hot pairs to fit, or no patches
            # gathered) rather than logging NaN to wandb.
            if not math.isnan(r2):
                pl_module.log(
                    f"{self._key}/{src}/r2_hot",
                    r2,
                    sync_dist=False,
                    rank_zero_only=True,
                )
            if not math.isnan(search_acc):
                pl_module.log(
                    f"{self._key}/{src}/search_hit_hot",
                    search_acc,
                    sync_dist=False,
                    rank_zero_only=True,
                )
        self._reset()
