from collections import defaultdict
from collections.abc import Callable
from typing import Any, ClassVar, cast, override

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from tensordict import TensorDict
from torch import Tensor

from rmind.callbacks.loggers.common import _get_wandb_loggers
from rmind.models.control_transformer import PredictionConfig


class RuleBasedCluster:
    """Assigns named cluster labels by applying priority-ordered rules to batch/prediction fields.

    Key paths start with "batch" or "predictions". Prediction paths use the flat key produced
    by predictions.flatten_keys("/"), so the second element is the full flat key string.

    Each field is reduced to a per-episode scalar [B] (mean over non-batch dims).
    Rules are evaluated in order; the first matching rule wins.
    Episodes matching no rule get the `default` label.

    Supported condition ops: ge, gt, le, lt, abs_ge, abs_gt, abs_le, abs_lt.

    Example config:
        _target_: rmind.callbacks.RuleBasedCluster
        fields:
          gas:   [predictions, "policy/ground_truth/value/continuous/gas_pedal"]
          brake: [predictions, "policy/ground_truth/value/continuous/brake_pedal"]
          steer: [predictions, "policy/ground_truth/value/continuous/steering_angle"]
          dp_g:  [predictions, "policy/ground_truth_diff_prev/value/continuous/gas_pedal"]
        rules:
          - name: braking_turn
            when: {brake: {ge: 0.02}, steer: {abs_ge: 0.05}}
          - name: braking
            when: {brake: {ge: 0.02}, steer: {abs_lt: 0.05}}
        default: idle_coast
    """

    _OPS: ClassVar = {
        "ge": lambda v, t: v >= t,
        "lt": lambda v, t: v < t,
        "abs_ge": lambda v, t: v.abs() >= t,
        "abs_lt": lambda v, t: v.abs() < t,
    }

    def __init__(
        self, fields: dict[str, dict], rules: list[dict], default: str = "other"
    ) -> None:
        self._fields = fields
        self._rules = rules
        self._default = default

    @staticmethod
    def _extract(data: dict[str, Any], spec: dict) -> Tensor:
        values: Tensor = data["data"][spec["key"]]
        match reduce := spec["reduce"]:
            case "last":
                result = values[:, -1]  # (b,)
            case "last_diff":
                result = values[:, -1] - values[:, -2]  # (b,)
            case _:
                msg = f"unsupported reduce: {reduce!r}"
                raise ValueError(msg)
        return result

    def __call__(
        self, batch: dict[str, TensorDict], _predictions: TensorDict
    ) -> list[str]:
        scalars = {
            name: self._extract(batch, spec).cpu()
            for name, spec in self._fields.items()
        }
        b = next(iter(scalars.values())).shape[0]

        labels = [self._default] * b
        assigned = torch.zeros(b, dtype=torch.bool)

        for rule in self._rules:
            mask = torch.ones(b, dtype=torch.bool)
            for field, cond in rule["when"].items():
                for op, threshold in cond.items():
                    mask &= self._OPS[op](scalars[field], threshold)

            newly_matched = mask & ~assigned
            for i in newly_matched.nonzero(as_tuple=True)[0].tolist():
                labels[i] = rule["name"]
            assigned |= newly_matched

        return labels


class PredictMetricsCallback(Callback):
    def __init__(
        self,
        cluster_fn: Callable | None = None,
        cluster_metrics: dict[str, list[str]] | None = None,
        prediction_config: PredictionConfig | None = None,
    ) -> None:
        self._cluster_fn = cluster_fn
        self._cluster_metrics = cluster_metrics
        if isinstance(prediction_config, dict):
            prediction_config = PredictionConfig(**prediction_config)
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
        self._prev_prediction_config = cast(
            "PredictionConfig", pl_module.prediction_config
        )
        if self._prediction_config is not None:
            pl_module.prediction_config = self._prediction_config  # ty:ignore[unresolved-attribute]

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
        if self._prev_prediction_config is not None:
            pl_module.prediction_config = self._prev_prediction_config  # ty:ignore[unresolved-attribute]

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
