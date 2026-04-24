from collections.abc import Set as AbstractSet
from typing import final, overload, override

import torch
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as F
from torch.utils._pytree import tree_map, tree_map_with_path  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.distributions import categorical_expected_value, categorical_std
from rmind.components.episode import Episode, EpisodeExport
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)
from rmind.utils.functional import non_zero_signal_with_threshold


@final
class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        output_tokenizers: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.heads: ModuleDict = heads
        self.output_tokenizers: ModuleDict = output_tokenizers
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

    @overload
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict: ...

    @overload
    def forward(self, episode: EpisodeExport, embedding: Tensor) -> TensorTree: ...

    @override
    def forward(
        self, episode: Episode | EpisodeExport, embedding: Tensor
    ) -> TensorDict | TensorTree:
        logits = self._compute_logits(episode=episode, embedding=embedding)

        if isinstance(episode, Episode):
            return TensorDict(logits).named_apply(self._decode_value, nested_keys=True)  # ty:ignore[invalid-return-type, invalid-argument-type]

        return tree_map_with_path(
            lambda kp, v: self._decode_value((kp[0].key, kp[1].key), v), logits
        )

    def _compute_logits(
        self, *, episode: Episode | EpisodeExport, embedding: Tensor
    ) -> TensorTree:
        if isinstance(episode, Episode):
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

            waypoints = embeddings.get((Modality.CONTEXT, "waypoints")).mean(
                dim=1, keepdim=True
            )

        else:
            observation_summary = embedding[
                :,
                episode.index[Modality.SUMMARY.value][  # ty:ignore[invalid-argument-type]
                    SummaryToken.OBSERVATION_SUMMARY.value
                ][-1],
            ]

            observation_history = embedding[
                :,
                episode.index[Modality.SUMMARY.value][  # ty:ignore[invalid-argument-type]
                    SummaryToken.OBSERVATION_HISTORY.value
                ][-1],
            ]

            waypoints = embedding[
                :, episode.index[Modality.CONTEXT.value]["waypoints"][-1]  # ty:ignore[invalid-argument-type]
            ].mean(dim=1, keepdim=True)

        features = rearrange(
            [observation_summary, observation_history.detach(), waypoints],
            "i b 1 d -> b 1 (i d)",
        )

        return self.heads(features)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        b, _t = episode.input.batch_size
        logits = TensorDict(
            self._compute_logits(episode=episode, embedding=embedding),  # ty:ignore[invalid-argument-type]
            batch_size=[b, 1],
        )
        targets = TensorDict(
            tree_map(
                lambda k: episode.get(k)[:, -1],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            ),
            batch_size=[b, 1],
        )

        losses = self.losses(
            tree_map(lambda x: rearrange(x, "b t d -> (b t) d"), logits.to_dict()),
            tree_map(lambda x: rearrange(x, "b t -> (b t)"), targets.to_dict()),
        )  # ty:ignore[call-non-callable]

        return {"loss": losses}

    @override
    def predict(
        self,
        episode: Episode,
        *,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        b, _t = episode.input.batch_size

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).squeeze(-1),
                timestep_indices=slice(None),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_STD,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
        }:
            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

            logits = TensorDict(
                self._compute_logits(episode=episode, embedding=embedding),  # ty:ignore[invalid-argument-type]
                batch_size=[b, 1],
            )
            target_tokens = TensorDict(
                tree_map(
                    lambda k: episode.get(k)[:, -1],
                    self.targets,
                    is_leaf=lambda x: isinstance(x, tuple),
                ),
                batch_size=[b, 1],
            )
            ground_truth = episode.input.select(*self.heads.tree_paths())[:, -1]
            timestep_indices = slice(-1, None)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.named_apply(self._decode_value, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_STD) in keys:
                predictions[key] = Prediction(
                    value=logits.named_apply(
                        lambda k, v: (
                            categorical_std(v, self.output_tokenizers.get_deepest(k))
                            if k[0] == Modality.CONTINUOUS
                            else torch.zeros_like(v[..., 0])
                        ),
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.softmax(dim=-1)),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_LOGPROB) in keys:
                predictions[key] = Prediction(
                    value=logits.named_apply(
                        lambda k, v: F.cross_entropy(
                            rearrange(v, "... d -> (...) d"),
                            rearrange(target_tokens.get(k).long(), "... -> (...)"),
                            reduction="none",
                        ).reshape(v.shape[:-1]),
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=logits.named_apply(
                        lambda k, v: F.l1_loss(
                            self._decode_value(k, v).float(),
                            ground_truth.get(k).float(),  # ty:ignore[unresolved-attribute]
                            reduction="none",
                        ),
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    def _decode_value(
        self, action_type: tuple[str, ...] | tuple[Modality, str], logits: Tensor
    ) -> Tensor:
        if action_type[0] == Modality.CONTINUOUS:
            return categorical_expected_value(
                logits, self.output_tokenizers.get_deepest(action_type)
            )

        if action_type == (Modality.DISCRETE, "turn_signal"):
            return non_zero_signal_with_threshold(logits).class_idx

        if action_type[0] == Modality.DISCRETE:
            return logits.argmax(dim=-1)

        msg = f"invalid action type: {action_type}"
        raise NotImplementedError(msg)
