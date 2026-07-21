from collections import defaultdict
from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class TrainingQualityLogger(Callback):
    """Debug signals for training health, logged to the trainer's logger (wandb).

    Every `every_n_steps` optimizer steps (pre-clipping):
      - `quality/grad_norm/global` and per-module-group grad norms
      - `quality/weight_norm/<group>` and `quality/grad_to_weight/<group>`
      - `quality/dead_grad_frac/<group>`: fraction of parameters with exactly
        zero gradient (dead units / unused embedding rows)

    At each validation epoch end:
      - `quality/gap/<objective>/<loss>`: val loss minus the mean train loss
        over the elapsed train window, for every matching `val/.../loss/...`
        key (positive gap growing over time = overfitting early-warning).

    Module groups: top-level (`episode_builder`, `encoder`) plus one extra
    level under `objectives` (per-objective) and `episode_builder`
    (embeddings / projections / ...).
    """

    @validate_call
    def __init__(
        self, every_n_steps: int = 50, *, repr_max_tokens: int = 8192
    ) -> None:
        super().__init__()
        self.every_n_steps = every_n_steps
        self.repr_max_tokens = repr_max_tokens
        self._train_loss_sums: dict[str, float] = defaultdict(float)
        self._train_loss_counts: dict[str, int] = defaultdict(int)
        self._repr_sample: Tensor | None = None  # last val encoder embedding
        self._hook_handle: Any = None

    @staticmethod
    def _effective_rank(x: Tensor) -> float:
        """Entropy-based effective rank of a (N, D) feature matrix.

        exp(H) of the normalized singular-value spectrum of the centered
        features; collapses toward 1 under representation collapse and toward
        min(N, D) for an isotropic (full-rank) representation. This is the
        primary signal LeJEPA/SIGReg is meant to keep high.
        """
        x = x - x.mean(0, keepdim=True)
        try:
            s = torch.linalg.svdvals(x.float())
        except RuntimeError:
            return float("nan")
        p = s / s.sum().clamp_min(1e-12)
        return float(torch.exp(-(p * (p + 1e-12).log()).sum()))

    @override
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # capture the encoder output during validation for representation health
        encoder = getattr(pl_module, "encoder", None)
        if encoder is not None and self._hook_handle is None:
            def _hook(_module: Any, _inp: Any, out: Any) -> None:
                if not pl_module.training and self._repr_sample is None:
                    t = out[0] if isinstance(out, tuple) else out
                    if isinstance(t, Tensor):
                        flat = t.detach().flatten(0, -2)
                        self._repr_sample = flat[: self.repr_max_tokens].float().cpu()
            self._hook_handle = encoder.register_forward_hook(_hook)

    @staticmethod
    def _group(name: str) -> str:
        parts = name.split(".")
        if parts[0] in {"objectives", "episode_builder"} and len(parts) > 1:
            return f"{parts[0]}.{parts[1]}"
        return parts[0]

    @override
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return

        grad_sq: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0))
        weight_sq: dict[str, Tensor] = defaultdict(lambda: torch.tensor(0.0))
        dead: dict[str, int] = defaultdict(int)
        numel: dict[str, int] = defaultdict(int)

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue
            g = self._group(name)
            grad = param.grad.detach()
            grad_sq[g] += grad.square().sum().cpu()
            weight_sq[g] += param.detach().square().sum().cpu()
            dead[g] += int((grad == 0).sum().item())
            numel[g] += grad.numel()

        if not grad_sq:
            return

        metrics: dict[str, float] = {}
        total_sq = torch.stack(list(grad_sq.values())).sum()
        metrics["quality/grad_norm/global"] = total_sq.sqrt().item()
        for g, sq in grad_sq.items():
            gn = sq.sqrt().item()
            wn = weight_sq[g].sqrt().item()
            metrics[f"quality/grad_norm/{g}"] = gn
            metrics[f"quality/weight_norm/{g}"] = wn
            metrics[f"quality/grad_to_weight/{g}"] = gn / (wn + 1e-12)
            metrics[f"quality/dead_grad_frac/{g}"] = dead[g] / max(numel[g], 1)

        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # accumulate per-objective train losses for the train/val gap
        for key, value in trainer.callback_metrics.items():
            if key.startswith("train/") and "/loss/" in key:
                self._train_loss_sums[key] += float(value)
                self._train_loss_counts[key] += 1

    @override
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            self._train_loss_sums.clear()
            self._train_loss_counts.clear()
            self._repr_sample = None
            return

        metrics: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if not key.startswith("val/"):
                continue
            train_key = "train/" + key.removeprefix("val/")
            if self._train_loss_counts.get(train_key):
                train_mean = (
                    self._train_loss_sums[train_key]
                    / self._train_loss_counts[train_key]
                )
                gap = float(value) - train_mean
                suffix = key.removeprefix("val/")
                metrics[f"quality/gap/{suffix}"] = gap

        # representation health from the captured val encoder embedding
        if self._repr_sample is not None:
            z = self._repr_sample
            metrics["quality/repr/effective_rank"] = self._effective_rank(z)
            metrics["quality/repr/feature_std"] = float(z.std(0).mean())
            metrics["quality/repr/feature_std_min"] = float(z.std(0).min())
            self._repr_sample = None

        if metrics:
            for logger in trainer.loggers:
                logger.log_metrics(metrics, step=trainer.global_step)

        self._train_loss_sums.clear()
        self._train_loss_counts.clear()
