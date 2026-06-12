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
from torch.utils._pytree import tree_map  # noqa: PLC2701

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


@final
class PolicyObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        norm: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict] | None = None,
        cross_attn_heads: InstanceOf[ModuleDict] | None = None,
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        waypoint_pooler: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        if (heads is None) == (cross_attn_heads is None):
            msg = "Exactly one of `heads` or `cross_attn_heads` must be set."
            raise ValueError(msg)

        self.norm: Module | None = norm
        # Default path: concat[obs_summary, obs_history, pooled_waypoints] -> MLP.
        self.heads: ModuleDict | None = heads
        # Cross-attention path ("Approach A2"): each action head is a
        # ``CrossAttentionPolicyHead`` whose learned query cross-attends over the
        # full token context cat([waypoints(n), obs_summary, obs_history]); this
        # REPLACES the concat->MLP path when set. Mirrors the same tree structure
        # (continuous/{gas,brake,steer}, discrete/turn_signal) so targets,
        # ``tree_paths`` and ``named_apply`` are unchanged.
        self.cross_attn_heads: ModuleDict | None = cross_attn_heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        # How the per-waypoint tokens are reduced to the single token the head
        # consumes. ``None`` => mean (centroid; cannot represent curvature).
        # A pooler (e.g. WaypointTransformerPooler) is a drop-in [b,n,d]->[b,1,d]
        # replacement that can read path shape; it warm-starts to mean at step 0.
        # Only used by the MLP path; the cross-attn path consumes raw waypoints.
        self.waypoint_pooler: Module | None = waypoint_pooler

    @property
    def _active_heads(self) -> ModuleDict:
        """The ModuleDict actually producing logits (MLP or cross-attn)."""
        if self.heads is not None:
            return self.heads
        return cast("ModuleDict", self.cross_attn_heads)

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits = self._compute_logits(episode=episode, embedding=embedding)

        def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:
            match nk:
                case (Modality.CONTINUOUS, _):
                    return x[..., 0]
                case (Modality.DISCRETE, "turn_signal"):
                    return non_zero_signal_with_threshold(x).class_idx
                case _:
                    raise NotImplementedError

        return TensorDict(logits).named_apply(fn, nested_keys=True)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def _aggregate_waypoints(self, waypoints: Tensor) -> Tensor:
        """Reduce per-waypoint tokens ``[b, n, d]`` to one token ``[b, 1, d]``."""
        if self.waypoint_pooler is not None:
            return self.waypoint_pooler(waypoints)
        return waypoints.mean(dim=1, keepdim=True)

    def _extract_features(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        """Assemble the head input from the encoder output.

        Single source of truth shared by ``forward`` / ``compute_metrics`` /
        ``predict`` so every head path consumes waypoints identically.
        """
        embeddings = (
            episode
            .index[-1]
            .select(
                (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
                (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
                (Modality.CONTEXT, "waypoints"),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_HISTORY,
        ))

        observation_summary = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_SUMMARY,
        ))

        waypoints_raw = embeddings.get((Modality.CONTEXT, "waypoints"))

        if self.cross_attn_heads is not None:
            # Cross-attention context: the full token sequence (no pooling).
            # cat([waypoints(n), obs_summary(1), obs_history(1)]) -> [b, n+2, d].
            # The selected tokens are already normed upstream (forward /
            # compute_metrics / predict apply ``self.norm`` to ``embedding``
            # before this call), so no extra per-token norm here -- consistent
            # with the MLP path.
            return torch.cat(
                [waypoints_raw, observation_summary, observation_history], dim=1
            )

        waypoints = self._aggregate_waypoints(waypoints_raw)

        return rearrange(
            [observation_summary, observation_history, waypoints],
            "i b 1 d -> b 1 (i d)",
        )

    def _compute_logits(self, *, episode: Episode, embedding: Tensor) -> TensorTree:
        features = self._extract_features(episode=episode, embedding=embedding)
        return self._active_heads(features)

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
    def predict(  # noqa: C901, PLR0912, PLR0915
        self,
        episode: Episode,
        *,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        b, _t = episode.input.batch_size

        if self.norm is not None:
            embedding = self.norm(embedding)

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self._active_heads.tree_paths()).squeeze(
                    -1
                ),
                timestep_indices=slice(None),
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

            logits = TensorDict(
                self._compute_logits(episode=episode, embedding=embedding),  # ty:ignore[invalid-argument-type]
                batch_size=[b, 1],
            )

            timestep_indices = slice(-1, None)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0]
                        case (Modality.DISCRETE, "turn_signal"):
                            return non_zero_signal_with_threshold(x).class_idx
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_STD) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return torch.sqrt(torch.exp(x[..., 1]))

                        case (Modality.DISCRETE, "turn_signal"):
                            return torch.zeros_like(x[..., 0])  # placeholder
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
                            return non_zero_signal_with_threshold(x).prob

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
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
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return F.l1_loss(x[..., 0], gt, reduction="none")  # ty:ignore[invalid-argument-type]

                        case (Modality.DISCRETE, "turn_signal"):
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
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
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
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_SIGNED_ERROR) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0] - gt  # ty:ignore[unsupported-operator]

                        case (Modality.DISCRETE, "turn_signal"):
                            return (
                                non_zero_signal_with_threshold(x).class_idx.float()
                                - gt.float()
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
                            prediction = non_zero_signal_with_threshold(
                                x
                            ).class_idx  # (b, 1)
                            return (prediction != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
                            return (gt != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
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
                    timestep_indices=timestep_indices,
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

                        case (Modality.DISCRETE, "turn_signal"):
                            gt_hist = torch.mode(gt_episode, dim=1).values
                            return (gt != gt_hist).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
