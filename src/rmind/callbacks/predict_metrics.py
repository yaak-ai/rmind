from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast, override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from rmind.callbacks.loggers.common import _get_wandb_loggers
from rmind.models.control_transformer import PredictionConfig


class PredictMetricsCallback(Callback):
    @validate_call
    def __init__(
        self,
        prediction_config: PredictionConfig,
        cluster_fn: Callable | None = None,
        cluster_metrics: dict[str, list[str]] | None = None,
    ) -> None:
        self._cluster_fn = cluster_fn
        self._cluster_metrics = cluster_metrics
        self._prediction_config = prediction_config
        self._accumulated: dict[str, list[Tensor]] = defaultdict(list)
        self._accumulated_by_cluster: dict[str, dict[str, list[tuple[Tensor, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._prev_prediction_config: PredictionConfig | None = None

    @override
    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._accumulated = defaultdict(list)
        self._accumulated_by_cluster = defaultdict(lambda: defaultdict(list))
        if not hasattr(pl_module, "prediction_config"):
            return
        self._prev_prediction_config = cast(
            "PredictionConfig", pl_module.prediction_config
        )
        pl_module.prediction_config = self._prediction_config  # ty: ignore[unresolved-attribute]

    @override
    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        predictions = pl_module.predict_step(batch)

        clusters = (
            self._cluster_fn(batch, predictions)
            if self._cluster_fn is not None
            else None
        )

        cluster_masks: dict[str, Tensor] | None = None
        if clusters is not None:
            cluster_masks = {
                cname: torch.tensor([c == cname for c in clusters])
                for cname in set(clusters)
            }

        all_allowed: set[str] | None = (
            set().union(*self._cluster_metrics.values())
            if self._cluster_metrics is not None
            else None
        )

        for key, v in predictions.flatten_keys("/").items():
            if not isinstance(v, Tensor):
                continue
            if all_allowed is not None and key not in all_allowed:
                continue

            self._accumulated[key].append(v.float().mean())

            if cluster_masks is not None:
                for cname, mask in cluster_masks.items():
                    if self._cluster_metrics is not None:
                        allowed = self._cluster_metrics.get(cname)
                        if allowed is None or key not in allowed:
                            continue
                    filtered = v[mask].float()
                    self._accumulated_by_cluster[cname][key].append((
                        filtered.sum(),
                        filtered.numel(),
                    ))

    @override
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            hasattr(pl_module, "prediction_config")
            and self._prev_prediction_config is not None
        ):
            pl_module.prediction_config = self._prev_prediction_config  # ty: ignore[unresolved-attribute]
            self._prev_prediction_config = None

        metrics: dict[str, Tensor] = {
            f"predict/{k}": torch.stack(vs).mean()
            for k, vs in self._accumulated.items()
        }
        for cname, cname_acc in self._accumulated_by_cluster.items():
            for k, vs in cname_acc.items():
                total = cast("Tensor", sum(s for s, _ in vs))
                count = sum(n for _, n in vs)
                metrics[f"predict/{cname}/{k}"] = total / count

        scalar_metrics: dict[str, float] = {k: v.item() for k, v in metrics.items()}
        for wandb_logger in _get_wandb_loggers(pl_module):
            wandb_logger.log_metrics(scalar_metrics, step=trainer.global_step)
