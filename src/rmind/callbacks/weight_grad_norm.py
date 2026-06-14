from typing import override

import pytorch_lightning as pl
import torch
from pydantic import validate_call
from pytorch_lightning.callbacks import Callback
from torch import nn

from rmind.callbacks.loggers.common import _get_wandb_loggers


def _l2_norm(module: nn.Module) -> float:
    """Total L2 norm of a module's parameters (flattened over all params)."""
    sq = 0.0
    for p in module.parameters(recurse=True):
        sq += p.detach().float().pow(2).sum().item()
    return sq**0.5


def _grad_l2_norm(module: nn.Module) -> float:
    """Total L2 norm of a module's parameter gradients (None grads skipped)."""
    sq = 0.0
    for p in module.parameters(recurse=True):
        if p.grad is None:
            continue
        sq += p.grad.detach().float().pow(2).sum().item()
    return sq**0.5


def compute_weight_norms(heads: nn.Module) -> dict[str, float]:
    """Per-action-head + total parameter L2 norms.

    ``heads`` is the active action-head ``ModuleDict`` (tree
    ``continuous/{gas_pedal,brake_pedal,steering_angle}`` + ``discrete/turn_signal``).
    Returns keys like ``train/weight_norm/total`` and
    ``train/weight_norm/continuous/steering_angle``. Robust to plain ``nn.Module``s
    (no ``tree_paths``): then only ``total`` is reported.
    """
    out: dict[str, float] = {"train/weight_norm/total": _l2_norm(heads)}

    tree_paths = getattr(heads, "tree_paths", None)
    get = getattr(heads, "get", None)
    if callable(tree_paths) and callable(get):
        for path in tree_paths():
            sub = get(path)
            if isinstance(sub, nn.Module):
                out[f"train/weight_norm/{'/'.join(path)}"] = _l2_norm(sub)

    return out


class WeightGradNormLogger(Callback):
    """Logs trainable-head weight + gradient L2 norms to the wandb logger.

    Locates overfitting by tracking how the head's weights grow and how large the
    gradients are. Per epoch (``on_train_epoch_end``) it logs:

    - ``train/weight_norm/total`` -- total param L2 norm of the trainable head
      (``pl_module.objectives.policy.heads``).
    - ``train/weight_norm/{head}`` -- per-action-head param L2 norm, e.g.
      ``train/weight_norm/continuous/steering_angle``,
      ``train/weight_norm/discrete/turn_signal``.
    - ``train/grad_norm/total`` -- total gradient L2 norm captured at the last
      optimizer step of the epoch (over the same head module). Robust to ``None``
      grads.

    Optionally, with ``every_n_steps`` set, ``train/grad_norm/total`` is also logged
    every N optimizer steps (in ``on_before_optimizer_step``).

    Scope note: grad norm is computed over ``pl_module.objectives.policy.heads`` --
    the same module as the weight norms (apples-to-apples). With the encoder frozen
    this is the full set of trainable params anyway.
    """

    @validate_call
    def __init__(self, *, every_n_steps: int | None = None) -> None:
        self._every_n_steps = every_n_steps
        self._last_grad_norm: float | None = None

    def _heads(self, pl_module: pl.LightningModule) -> nn.Module | None:
        objectives = getattr(pl_module, "objectives", None)
        policy = getattr(objectives, "policy", None) if objectives is not None else None
        heads = getattr(policy, "heads", None) if policy is not None else None
        return heads if isinstance(heads, nn.Module) else None

    @override
    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: object,
        *args: object,
    ) -> None:
        heads = self._heads(pl_module)
        if heads is None:
            return

        self._last_grad_norm = _grad_l2_norm(heads)

        if (
            self._every_n_steps is not None
            and trainer.global_step % self._every_n_steps == 0
            and trainer.is_global_zero
        ):
            for logger_ in _get_wandb_loggers(pl_module):
                logger_.log_metrics(
                    {"train/grad_norm/total": self._last_grad_norm},
                    step=trainer.global_step,
                )

    @override
    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        heads = self._heads(pl_module)
        if heads is None:
            return

        metrics = compute_weight_norms(heads)
        if self._last_grad_norm is not None:
            metrics["train/grad_norm/total"] = self._last_grad_norm

        for logger_ in _get_wandb_loggers(pl_module):
            logger_.log_metrics(metrics, step=trainer.global_step)
