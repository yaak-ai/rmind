"""Horizon-matched deterministic regression baseline for the flow policy.

The decisive "is flow earning its complexity?" control. The existing baseline
(PolicyObjective) regresses only the last observed step, so it was never
comparable to the flow expert (which predicts the future action_horizon chunk).
This objective predicts the SAME 6-slot future chunk from the SAME condition
tokens (observation summaries + waypoints), trained in the SAME transformed
model space (gas/brake merge + Gaussianize), with the SAME raw-space metrics
(`sample_l1`, `maneuver_l1/*`) — a single deterministic forward instead of an
integrated flow. Decoder capacity mirrors the flow decoder (same dim/layers/
heads cross-attention stack; learned slot queries instead of noised actions).

What each outcome means:
  - regression ~= flow's mean-of-K and >> flow single-draw: the flow's gap is
    sampling variance; a deterministic readout (or draw aggregation) is the
    cheaper equivalent, and flow's value rests on multimodality alone.
  - regression << flow single-draw everywhere: flow is not earning its
    complexity on this data.
  - flow single-draw beats regression on maneuvers: the distributional model
    captures structure the conditional mean cannot (multimodal maneuvers) —
    the strongest pro-flow evidence available open-loop.

Deliberately mirrors FlowPolicyObjective's helpers/metrics so wandb curves and
predict parquets are directly comparable.
"""

from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.action_transform import GaussianizeActionTransform
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)
from rmind.components.objectives.flow_policy import (
    DEFAULT_ACTION_KEYS,
    DEFAULT_BATCH_ACTION_PATHS,
    POLICY_CONDITION_TOKENS,
)
from rmind.components.objectives.maneuver_weights import ManeuverLossWeights
from rmind.components.transformer.decoder import CrossAttentionDecoder

logger = get_logger(__name__)


@final
class RegressionActionDecoder(torch.nn.Module):
    """Learned slot queries cross-attending to condition tokens -> (H, A) chunk.

    Capacity-matched to FlowActionDecoder: same condition projection, the same
    CrossAttentionDecoder stack (cross-attn + self-attn + MLP per block), and a
    linear output head. Slot identity enters through the queries (nn.Embedding,
    like the flow decoder's position_embedding, so SelectiveAdamW classifies it
    and weight-decay-excludes it).
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        condition_dim: int,
        dim_model: int | None = None,
        action_dim: int = 3,
        action_horizon: int = 6,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
    ) -> None:
        super().__init__()
        dim_model = dim_model or condition_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.condition_projection: torch.nn.Module = (
            torch.nn.Identity()
            if condition_dim == dim_model
            else torch.nn.Linear(condition_dim, dim_model)
        )
        self.slot_queries = torch.nn.Embedding(action_horizon, dim_model)
        torch.nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)
        self.decoder = CrossAttentionDecoder(
            dim_model=dim_model,
            num_layers=num_layers,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            mlp_dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )
        self.output_projection = torch.nn.Linear(dim_model, action_dim)

    @override
    def forward(self, *, condition_tokens: Tensor) -> Tensor:
        context = self.condition_projection(condition_tokens)
        batch_size = context.shape[0]
        indices = torch.arange(self.action_horizon, device=context.device)
        x = (
            self.slot_queries(indices)
            .to(context.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return self.output_projection(self.decoder(x, context))


@final
class RegressionPolicyObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: InstanceOf[RegressionActionDecoder],
        history_steps: int = 1,
        action_keys: tuple[str, ...] = DEFAULT_ACTION_KEYS,
        batch_action_paths: tuple[tuple[str, ...], ...] = DEFAULT_BATCH_ACTION_PATHS,
        maneuver_thresholds: tuple[float, ...] | None = None,
        action_transform_stats: str | None = None,
        lds_stats: str | None = None,
        lds_alpha: float = 0.5,
        lds_cap: float = 15.0,
    ) -> None:
        super().__init__()
        if history_steps <= 0:
            msg = f"history_steps must be positive, got {history_steps}"
            raise ValueError(msg)

        if not action_transform_stats:
            action_transform = None
            raw_dim = decoder.action_dim
        else:
            action_transform = GaussianizeActionTransform.from_stats_file(
                action_transform_stats
            )
            if action_transform.action_keys != action_keys:
                msg = (
                    "action_transform stats action_keys "
                    f"{action_transform.action_keys} != objective action_keys {action_keys}"
                )
                raise ValueError(msg)
            if action_transform.model_dim != decoder.action_dim:
                msg = (
                    f"action_transform.model_dim {action_transform.model_dim} != "
                    f"decoder.action_dim {decoder.action_dim}"
                )
                raise ValueError(msg)
            raw_dim = action_transform.raw_dim

        if len(action_keys) != raw_dim:
            msg = f"action_keys must have {raw_dim} entries, got {len(action_keys)}"
            raise ValueError(msg)
        if len(batch_action_paths) != raw_dim:
            msg = f"batch_action_paths must have {raw_dim} entries, got {len(batch_action_paths)}"
            raise ValueError(msg)
        if maneuver_thresholds is not None and len(maneuver_thresholds) != raw_dim:
            msg = (
                f"maneuver_thresholds must have {raw_dim} entries, "
                f"got {len(maneuver_thresholds)}"
            )
            raise ValueError(msg)

        self.decoder = decoder
        self.history_steps = history_steps
        self.action_keys = action_keys
        self.batch_action_paths = batch_action_paths
        self.action_transform = action_transform
        if maneuver_thresholds is None:
            self.maneuver_thresholds = None
        else:
            self.register_buffer(
                "maneuver_thresholds",
                torch.tensor(maneuver_thresholds, dtype=torch.float32),
                persistent=False,
            )
        if not lds_stats:
            self.maneuver_weights = None
        else:
            self.maneuver_weights = ManeuverLossWeights.from_stats_file(
                lds_stats, alpha=lds_alpha, cap=lds_cap
            )
            if self.maneuver_weights.model_dim != decoder.action_dim:
                msg = (
                    f"lds model_dim {self.maneuver_weights.model_dim} != "
                    f"decoder.action_dim {decoder.action_dim}"
                )
                raise ValueError(msg)

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> TensorDict:
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        return self._trajectory_to_tensordict(
            self._to_raw_space(self.decoder(condition_tokens=condition_tokens))
        )

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, batch: Any, **_kwargs: Any
    ) -> Metrics:
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        target_actions = self._target_actions(batch)

        # Same NaN guard as the flow objective (NaN waypoints poison the
        # condition for the whole sample).
        finite = (
            condition_tokens.flatten(1).isfinite().all(dim=1)
            & target_actions.flatten(1).isfinite().all(dim=1)
        )
        if not bool(finite.all()):
            logger.warning(
                "dropping non-finite samples from regression loss",
                dropped=int((~finite).sum()),
                batch=int(finite.numel()),
            )
            if bool(finite.any()):
                condition_tokens = condition_tokens[finite]
                target_actions = target_actions[finite]

        return self.compute_metrics_from(
            condition_tokens=condition_tokens, target_actions=target_actions
        )

    def compute_metrics_from(
        self, *, condition_tokens: Tensor, target_actions: Tensor
    ) -> Metrics:
        """Metrics body downstream of condition/target extraction (feature-cached
        training entry point, mirroring FlowPolicyObjective.compute_metrics_from)."""
        finite = (
            condition_tokens.flatten(1).isfinite().all(dim=1)
            & target_actions.flatten(1).isfinite().all(dim=1)
        )
        if not bool(finite.all()) and bool(finite.any()):
            condition_tokens = condition_tokens[finite]
            target_actions = target_actions[finite]

        target_model = self._to_model_space(target_actions)
        pred_model = self.decoder(condition_tokens=condition_tokens)

        weight = self._maneuver_weight(target_actions)
        se = (pred_model - target_model).pow(2)
        mse = se.mean()
        loss = mse if weight is None else (weight * se).mean()
        metrics: Metrics = {"loss": loss, "flow_mse": mse.detach()}

        if not self.training:
            pred_raw = self._to_raw_space(pred_model)
            err = (pred_raw - target_actions.to(pred_raw.dtype)).abs()
            metrics["sample_l1"] = err.mean()
            if self.maneuver_thresholds is not None:
                thr = self.maneuver_thresholds.to(target_actions.dtype)
                active = target_actions.abs() > thr
                for c, key in enumerate(self.action_keys):
                    frames = active[..., c]
                    metrics[f"maneuver_frac/{key}"] = frames.float().mean().detach()
                    if bool(frames.any()):
                        metrics[f"maneuver_l1/{key}"] = (
                            err[..., c][frames].mean().detach()
                        )
        return metrics

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        batch: Any,
        **_kwargs: Any,
    ) -> TensorDict:
        del tokenizers
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        target_actions: Tensor | None = None
        prediction_actions: Tensor | None = None

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            target_actions = self._target_actions(batch)
            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(target_actions),
                timestep_indices=self._target_slice(),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.SCORE_L1,
        }:
            condition_tokens = self._condition_tokens(
                episode=episode, embedding=embedding
            )
            prediction_actions = self._to_raw_space(
                self.decoder(condition_tokens=condition_tokens)
            )

        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            if prediction_actions is None:
                msg = "prediction_actions were not computed"
                raise RuntimeError(msg)
            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(prediction_actions),
                timestep_indices=self._target_slice(),
            )

        if (key := ObjectivePredictionKey.SCORE_L1) in keys:
            if target_actions is None:
                target_actions = self._target_actions(batch)
            if prediction_actions is None:
                msg = "prediction_actions were not computed"
                raise RuntimeError(msg)
            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(
                    F.l1_loss(prediction_actions, target_actions, reduction="none")
                ),
                timestep_indices=self._target_slice(),
            )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    def _maneuver_weight(self, target_actions: Tensor) -> Tensor | None:
        if self.maneuver_weights is None:
            return None
        phys = (
            self.action_transform.physical_model(target_actions)
            if self.action_transform is not None
            else target_actions
        )
        label = phys.abs().amax(dim=1)
        return self.maneuver_weights(label).unsqueeze(1)

    def _condition_tokens(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        embeddings = (
            episode
            .index[self.history_steps - 1]
            .select(*POLICY_CONDITION_TOKENS)
            .parse(embedding)
        )
        return torch.cat(
            [embeddings.get(path) for path in POLICY_CONDITION_TOKENS], dim=1
        )

    def _target_actions(self, batch: Any) -> Tensor:
        actions: list[Tensor] = []
        for path in self.batch_action_paths:
            value: Any = batch
            for key in path:
                value = value[key]
            actions.append(value)
        return torch.stack(actions, dim=-1)

    def _to_model_space(self, raw: Tensor) -> Tensor:
        return raw if self.action_transform is None else self.action_transform(raw)

    def _to_raw_space(self, model: Tensor) -> Tensor:
        if self.action_transform is None:
            return model
        return self.action_transform.inverse(model)

    def _target_slice(self) -> slice:
        start = self.history_steps
        return slice(start, start + self.decoder.action_horizon)

    def _trajectory_to_tensordict(self, trajectory: Tensor) -> TensorDict:
        return TensorDict(
            {
                Modality.CONTINUOUS: {
                    key: trajectory[..., idx]
                    for idx, key in enumerate(self.action_keys)
                }
            },  # ty:ignore[invalid-argument-type]
            batch_size=list(trajectory.shape[:2]),
            device=trajectory.device,
        )
