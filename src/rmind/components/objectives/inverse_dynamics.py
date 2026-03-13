from collections.abc import Set as AbstractSet
from typing import final, override

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)


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

        logits = self.heads(features)

        targets = tree_map(
            lambda k: episode.get(k)[:, :-1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t 1 d -> (b t) d"), logits),
            tree_map(Rearrange("b t 1 -> (b t)"), targets),
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

            # all but last
            timestep_indices = slice(t - 1)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # ty:ignore[unresolved-attribute]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, unresolved-attribute]
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
                    value=(
                        logits
                        .apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))  # ty:ignore[unresolved-attribute]
                        .apply(  # ty:ignore[unresolved-attribute]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                        )
                        .apply(lambda x: -torch.log(x))  # ty:ignore[unresolved-attribute]
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[unresolved-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, unresolved-attribute]
                            nested_keys=True,
                        )
                        .apply(  # ty:ignore[unresolved-attribute]
                            lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                            episode.input[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                            nested_keys=True,
                        )
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
