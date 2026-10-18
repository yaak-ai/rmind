"""Opt-in Lightning training boundary for explicitly labeled offline transitions."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import pytorch_lightning as pl
import torch
from torch import Tensor

from rmind.components.fql import FQLConfig, FQLNoise, FQLPolicy

if TYPE_CHECKING:
    from pytorch_lightning.core.optimizer import LightningOptimizer


class FQLModel(pl.LightningModule):
    """Single-transition FQL; rewards and bootstrap masks are supplied, never inferred.

    Public observations/actions are physical units. Normalization is checkpointed;
    the component operates on standardized observations and bounded canonical actions.
    This model deliberately does not implement the D12/RTC serving interface.
    """

    observation_mean: Tensor
    observation_std: Tensor
    action_low: Tensor
    action_high: Tensor

    def __init__(
        self,
        config: dict[str, Any],
        *,
        learning_rate: float = 3e-4,
        normalization: dict[str, list[float]] | None = None,
    ) -> None:
        super().__init__()
        policy_config = FQLConfig(**config)
        self.policy = FQLPolicy(policy_config)
        if not 0 < learning_rate < float("inf"):
            message = "Learning rate must be finite and positive"
            raise ValueError(message)
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        normalization = normalization or {}
        defaults = {
            "observation_mean": (policy_config.observation_dim, 0.0),
            "observation_std": (policy_config.observation_dim, 1.0),
            "action_low": (policy_config.action_dim, -1.0),
            "action_high": (policy_config.action_dim, 1.0),
        }
        if normalization.keys() - defaults.keys():
            message = "Unknown normalization field"
            raise ValueError(message)
        for name, (size, default) in defaults.items():
            tensor = torch.tensor(
                normalization.get(name, [default] * size), dtype=torch.float32
            )
            if tensor.shape != (size,) or not torch.isfinite(tensor).all():
                message = f"Invalid {name} normalization"
                raise ValueError(message)
            self.register_buffer(name, tensor)
        if (self.observation_std <= 0).any() or (
            self.action_high <= self.action_low
        ).any():
            message = "Normalization scales and action ranges must be positive"
            raise ValueError(message)
        self.save_hyperparameters({
            "config": asdict(policy_config),
            "learning_rate": learning_rate,
            "normalization": {name: getattr(self, name).tolist() for name in defaults},
        })

    def normalized_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "observations": (batch["observations"] - self.observation_mean)
            / self.observation_std,
            "next_observations": (batch["next_observations"] - self.observation_mean)
            / self.observation_std,
            "actions": 2
            * (batch["actions"] - self.action_low)
            / (self.action_high - self.action_low)
            - 1,
            "rewards": batch["rewards"],
            "masks": batch["masks"],
        }

    def forward(self, observations: Tensor, noise: Tensor) -> Tensor:
        canonical = self.policy.sample_actions(
            (observations - self.observation_mean) / self.observation_std, noise
        )
        return (
            self.action_low + (canonical + 1) * (self.action_high - self.action_low) / 2
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        parameters = (
            parameter
            for parameter in self.policy.parameters()
            if parameter.requires_grad
        )
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
        )

    def on_fit_start(self) -> None:
        if str(self.trainer.precision) not in {"32", "32-true"}:
            message = (
                "FQL currently requires FP32 training; mixed precision is not validated"
            )
            raise ValueError(message)
        if (
            self.trainer.accumulate_grad_batches != 1
            or self.trainer.gradient_clip_val not in {None, 0}
        ):
            message = "FQL manual updates require accumulation1 and no external gradient clipping"
            raise ValueError(message)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        normalized = self.normalized_batch(batch)
        optimizer = cast("LightningOptimizer", self.optimizers())
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.policy.losses(
            normalized, FQLNoise.draw(normalized["actions"])
        )
        if not torch.isfinite(loss):
            message = "Nonfinite FQL loss; refusing optimizer/target update"
            raise ValueError(message)
        self.manual_backward(loss)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in self.policy.parameters()
        ):
            message = "Nonfinite FQL gradient; refusing optimizer/target update"
            raise ValueError(message)
        self.policy.update_targets()
        optimizer.step()
        self.log_dict(
            {f"train/{name}": value.detach() for name, value in metrics.items()},
            batch_size=len(normalized["actions"]),
        )
        return loss.detach()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        normalized = self.normalized_batch(batch)
        loss, metrics = self.policy.losses(
            normalized, FQLNoise.draw(normalized["actions"])
        )
        self.log_dict(
            {f"validation/{name}": value for name, value in metrics.items()},
            batch_size=len(normalized["actions"]),
        )
        return loss

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["fql_contract"] = {
            "schema": "rmind_fql_v1",
            "architecture": "official_low_dim_raw_time",
            "config": asdict(self.policy.config),
            "action_horizon": 1,
            "rtc_compatible": False,
            "observation_space": "physical",
            "output_action_space": "physical",
            "noise_space": "standard_normal",
            "target_update": "pre_optimizer_critic",
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        expected: dict[str, Any] = {}
        self.on_save_checkpoint(expected)
        if checkpoint.get("fql_contract") != expected["fql_contract"]:
            message = "Missing or incompatible FQL checkpoint contract"
            raise ValueError(message)
        for name in (
            "observation_mean",
            "observation_std",
            "action_low",
            "action_high",
        ):
            saved = checkpoint["state_dict"].get(name)
            if not isinstance(saved, Tensor) or not torch.equal(
                saved.cpu(), getattr(self, name).cpu()
            ):
                message = "Checkpoint normalization differs from constructor metadata"
                raise ValueError(message)
