from collections.abc import Set as AbstractSet
from typing import Any, cast, final, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # ruff: ignore[import-private-name]

from rmind.components.base import Modality, SummaryToken, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)
from rmind.utils.functional import gauss_prob, non_zero_signal_with_threshold

# (modality, name) of the last-timestep tokens concatenated into the head input,
# in order. Tokens spanning more than one position (e.g. waypoints) are averaged.
DEFAULT_FEATURES: tuple[tuple[str, str], ...] = (
    (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
    (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
    (Modality.CONTEXT, "waypoints"),
)


@final
class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        norm: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        features: tuple[tuple[str, str], ...] = DEFAULT_FEATURES,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.features: tuple[tuple[str, str], ...] = features

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits = self._compute_logits(episode=episode, embedding=embedding)

        def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:
            match nk:
                case (Modality.CONTINUOUS, _):
                    return x[..., 0]
                case (Modality.DISCRETE, _):
                    return non_zero_signal_with_threshold(x).class_idx
                case _:
                    raise NotImplementedError

        return TensorDict(logits).named_apply(fn, nested_keys=True)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def _build_features(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        embeddings = episode.index[-1].select(*self.features).parse(embedding)

        return rearrange(
            [embeddings.get(key).mean(dim=-2, keepdim=True) for key in self.features],
            "i b 1 d -> b 1 (i d)",
        )

    def _compute_logits(self, *, episode: Episode, embedding: Tensor) -> TensorTree:
        return self.heads(self._build_features(episode=episode, embedding=embedding))

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits = self._compute_logits(episode=episode, embedding=embedding)
        targets = tree_map(
            lambda k: episode.get(k)[:, -1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b 1 d -> b d"), logits),
            tree_map(Rearrange("b 1 -> b"), targets),
        )  # ty:ignore[call-non-callable]

        return {"loss": losses}

    @override
    def predict(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
        self,
        episode: Episode,
        *,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        b, t = episode.input.batch_size

        if self.norm is not None:
            embedding = self.norm(embedding)

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).squeeze(-1)
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_STD,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
            ObjectivePredictionKey.SCORE_L1_REL,
            ObjectivePredictionKey.PREDICTION_DIFF_PREV,
            ObjectivePredictionKey.GROUND_TRUTH_DIFF_PREV,
            ObjectivePredictionKey.PREDICTION_DIFF_HIST,
            ObjectivePredictionKey.GROUND_TRUTH_DIFF_HIST,
            ObjectivePredictionKey.SCORE_SIGNED_ERROR,
        }:
            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

            features = self._build_features(episode=episode, embedding=embedding)
            logits = TensorDict(self.heads(features), batch_size=[b, 1])

            timestep_index = slice(-1, None)
            time_index = torch.arange(t).expand(b, -1)[:, timestep_index]

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0]
                        case (Modality.DISCRETE, _):
                            return non_zero_signal_with_threshold(x).class_idx
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_STD) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return torch.sqrt(torch.exp(x[..., 1]))

                        case (Modality.DISCRETE, _):
                            return torch.zeros_like(x[..., 0])  # placeholder
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            return gauss_prob(mean, mean=mean, std=std)

                        case (Modality.DISCRETE, _):
                            return non_zero_signal_with_threshold(x).prob

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SCORE_LOGPROB) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            return -torch.log(gauss_prob(mean, mean=mean, std=std))

                        case (Modality.DISCRETE, _):
                            gt = episode.input[action_type][:, -1]
                            return F.cross_entropy(
                                x.squeeze(1),
                                gt.squeeze(1).long(),  # ty:ignore[unresolved-attribute]
                                reduction="none",
                            ).unsqueeze(1)

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return F.l1_loss(x[..., 0], gt, reduction="none")  # ty:ignore[invalid-argument-type]

                        case (Modality.DISCRETE, _):
                            return F.l1_loss(
                                non_zero_signal_with_threshold(x).class_idx.float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SCORE_L1_REL) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt = gt_episode[:, -1]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            prediction = x[..., 0]  # (b, 1)
                            return (prediction - gt).abs() / (gt.abs() + 1e-4)

                        case (Modality.DISCRETE, _):
                            return F.l1_loss(
                                non_zero_signal_with_threshold(x).class_idx.float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SCORE_SIGNED_ERROR) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0] - gt  # ty:ignore[unsupported-operator]

                        case (Modality.DISCRETE, _):
                            return (
                                non_zero_signal_with_threshold(x).class_idx.float()
                                - gt.float()
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_DIFF_PREV) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt_prev = gt_episode[:, -2]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            prediction = x[..., 0]  # (b, 1)
                            return prediction - gt_prev

                        case (Modality.DISCRETE, _):
                            prediction = non_zero_signal_with_threshold(
                                x
                            ).class_idx  # (b, 1)
                            return (prediction != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.GROUND_TRUTH_DIFF_PREV) in keys:

                def fn(
                    action_type: tuple[Modality, str], _x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt_prev = gt_episode[:, -2]  # (b, 1)
                    gt = gt_episode[:, -1]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return gt - gt_prev

                        case (Modality.DISCRETE, _):
                            return (gt != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_DIFF_HIST) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            gt_hist = gt_episode.mean(dim=1)  # (b, 1)
                            prediction = x[..., 0]  # (b, 1)
                            return prediction - gt_hist

                        case (Modality.DISCRETE, _):
                            gt_hist = torch.mode(gt_episode, dim=1).values  # (b, 1)
                            prediction = non_zero_signal_with_threshold(
                                x
                            ).class_idx  # (b, 1)
                            return (prediction != gt_hist).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.GROUND_TRUTH_DIFF_HIST) in keys:

                def fn(
                    action_type: tuple[Modality, str], _x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt = gt_episode[:, -1]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            gt_hist = gt_episode.mean(dim=1)  # (b, 1)
                            return gt - gt_hist

                        case (Modality.DISCRETE, _):
                            gt_hist = torch.mode(gt_episode, dim=1).values
                            return (gt != gt_hist).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    time_index=time_index,
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
