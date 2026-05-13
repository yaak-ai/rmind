from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)
from rmind.components.transformer import FlowActionDecoder

POLICY_CONDITION_TOKENS: tuple[tuple[Modality, str], ...] = (
    (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
    (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
)

DEFAULT_ACTION_KEYS = ("gas_pedal", "brake_pedal", "steering_angle")
DEFAULT_BATCH_ACTION_PATHS: tuple[tuple[str, ...], ...] = (
    ("data", "meta/VehicleMotion/gas_pedal_normalized_target"),
    ("data", "meta/VehicleMotion/brake_pedal_normalized_target"),
    ("data", "meta/VehicleMotion/steering_angle_normalized_target"),
)
FLOW_TIME_LOGIT_NORMAL_MEAN = 0.0
FLOW_TIME_LOGIT_NORMAL_STD = 1.0
FLOW_TIME_SAMPLING_METHODS = {"uniform", "logit-normal"}


@final
class FlowPolicyObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: InstanceOf[FlowActionDecoder],
        loss: InstanceOf[Module],
        history_steps: int = 1,
        action_keys: tuple[str, ...] = DEFAULT_ACTION_KEYS,
        batch_action_paths: tuple[tuple[str, ...], ...] = DEFAULT_BATCH_ACTION_PATHS,
        flow_time_sampling: str = "uniform",
        validation_seed: int = 0,
        validation_sample_metrics: bool = True,
    ) -> None:
        super().__init__()

        if history_steps <= 0:
            msg = f"history_steps must be positive, got {history_steps}"
            raise ValueError(msg)

        if len(action_keys) != decoder.action_dim:
            msg = (
                "action_keys must match decoder.action_dim, "
                f"got {len(action_keys)} keys for action_dim={decoder.action_dim}"
            )
            raise ValueError(msg)

        if len(batch_action_paths) != decoder.action_dim:
            msg = (
                "batch_action_paths must match decoder.action_dim, "
                f"got {len(batch_action_paths)} paths for action_dim={decoder.action_dim}"
            )
            raise ValueError(msg)

        if flow_time_sampling not in FLOW_TIME_SAMPLING_METHODS:
            msg = f"Invalid flow_time_sampling method: {flow_time_sampling}"
            raise ValueError(msg)

        self.decoder = decoder
        self.history_steps = history_steps
        self.action_keys = action_keys
        self.batch_action_paths = batch_action_paths
        self.flow_time_sampling = flow_time_sampling
        self.validation_seed = validation_seed
        self.validation_sample_metrics = validation_sample_metrics
        self.loss: Module = loss

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> TensorDict:
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        return self._trajectory_to_tensordict(
            self.decoder.sample(condition_tokens=condition_tokens)
        )

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, batch: Any, **_kwargs: Any
    ) -> Metrics:
        # Linear interpolant / action-flow target follows Flow Matching:
        # https://arxiv.org/abs/2210.02747
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        target_actions = self._target_actions(batch)
        generator = self._validation_generator(device=target_actions.device)
        noise = torch.randn(
            target_actions.shape,
            dtype=target_actions.dtype,
            device=target_actions.device,
            generator=generator,
        )
        flow_time = self.sample_flow_time(
            self.flow_time_sampling,
            target_actions.shape[0],
            dtype=target_actions.dtype,
            device=target_actions.device,
            generator=generator,
        )
        flow_time_broadcast = flow_time.view(-1, 1, 1)
        noised_actions = (
            1.0 - flow_time_broadcast
        ) * noise + flow_time_broadcast * target_actions
        target_action_flow = target_actions - noise

        predicted_action_flow = self.decoder(
            condition_tokens=condition_tokens,
            noised_actions=noised_actions,
            flow_time=flow_time,
        )

        metrics: Metrics = {
            "loss": self.loss(predicted_action_flow, target_action_flow)
        }
        if not self.training and self.validation_sample_metrics:
            sample_noise = torch.randn(
                target_actions.shape,
                dtype=target_actions.dtype,
                device=target_actions.device,
                generator=generator,
            )
            sample_actions = self.decoder.sample(
                condition_tokens=condition_tokens, noise=sample_noise
            )
            metrics["sample_l1"] = F.l1_loss(sample_actions, target_actions)

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
            prediction_actions = self.decoder.sample(condition_tokens=condition_tokens)

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

    @staticmethod
    def sample_flow_time(
        flow_time_sampling: str,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        match flow_time_sampling:
            case "uniform":
                return torch.rand(
                    batch_size, dtype=dtype, device=device, generator=generator
                )
            case "logit-normal":
                # Logit-normal timestep sampling follows SiT:
                # https://arxiv.org/abs/2401.08740
                # Official code: https://github.com/willisma/SiT
                z = torch.randn(
                    batch_size, dtype=dtype, device=device, generator=generator
                )
                return torch.sigmoid(
                    z * FLOW_TIME_LOGIT_NORMAL_STD + FLOW_TIME_LOGIT_NORMAL_MEAN
                )
            case _:
                msg_0 = f"Invalid flow_time_sampling method: {flow_time_sampling}"
                raise ValueError(msg_0)

    def _validation_generator(self, *, device: torch.device) -> torch.Generator | None:
        if self.training:
            return None

        generator = torch.Generator(device=device)
        generator.manual_seed(self.validation_seed)
        return generator
