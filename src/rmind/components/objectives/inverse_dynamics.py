from collections.abc import Set as AbstractSet
from typing import final, override

import torch.nn.functional as F
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.distributions import categorical_expected_value
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)
from rmind.utils.functional import non_zero_signal_with_threshold


@final
class InverseDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        observation_summaries = (
            episode.index
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        # order: (o0, o1), (o1, o2), (o2, o3), ...
        features = rearrange(
            [observation_summaries[:, :-1], observation_summaries[:, 1:]],
            "i ... d -> ... (i d)",
        )

        b, t = episode.input.batch_size
        logits = TensorDict(self.heads(features), batch_size=[b, t - 1])
        targets = TensorDict(
            tree_map(
                lambda k: episode.get(k)[:, :-1],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            ),
            batch_size=[b, t - 1],
        )

        losses = self.losses(
            tree_map(lambda x: rearrange(x, "b t 1 d -> (b t) d"), logits.to_dict()),
            tree_map(lambda x: rearrange(x, "b t 1 -> (b t)"), targets.to_dict()),
        )  # ty:ignore[call-non-callable]

        return {"loss": losses}

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        if tokenizers is None:
            msg = "tokenizers must be provided for inverse dynamics decoding"
            raise ValueError(msg)

        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        b, t = episode.input.batch_size

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()),
                timestep_indices=slice(None),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
        }:
            observation_summaries = (
                episode.index
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            # order: (o0, o1), (o1, o2), (o2, o3), ...
            features = rearrange(
                [observation_summaries[:, :-1], observation_summaries[:, 1:]],
                "i b t 1 d -> b t 1 (i d)",
            )

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])
            target_tokens = TensorDict(
                tree_map(
                    lambda k: episode.get(k)[:, :-1],
                    self.targets,
                    is_leaf=lambda x: isinstance(x, tuple),
                ),
                batch_size=[b, t - 1],
            )
            ground_truth = episode.input.select(*self.heads.tree_paths())[:, : t - 1]

            # all but last
            timestep_indices = slice(t - 1)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.named_apply(
                        lambda k, v: self._decode_value(k, v, tokenizers),
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
                            self._decode_value(k, v, tokenizers).float(),
                            ground_truth.get(k).float(),  # ty:ignore[unresolved-attribute]
                            reduction="none",
                        ),
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    @staticmethod
    def _decode_value(
        action_type: tuple[Modality, str], logits: Tensor, tokenizers: ModuleDict
    ) -> Tensor:
        if action_type[0] == Modality.CONTINUOUS:
            return categorical_expected_value(
                logits, tokenizers.get_deepest(action_type)
            )

        if action_type == (Modality.DISCRETE, "turn_signal"):
            return non_zero_signal_with_threshold(logits).class_idx

        if action_type[0] == Modality.DISCRETE:
            return logits.argmax(dim=-1)

        msg = f"invalid action type: {action_type}"
        raise NotImplementedError(msg)
