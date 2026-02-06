from collections.abc import Set as AbstractSet
from typing import final, override

from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, Modality, SummaryToken
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    Prediction,
    PredictionKey,
    Targets,
)


@final
class MemoryExtractionObjective(Objective):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)."""

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
    def compute_metrics(self, episode: Episode, *, embedding: Tensor) -> Metrics:
        features = (
            episode
            .index[1:]
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = self.heads(features)

        _, t = episode.input.batch_size
        targets = tree_map(
            lambda k: episode.get(k)[:, : t - 1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t 1 d -> (b t) d"), logits),
            tree_map(Rearrange("b t -> (b t)"), targets),
        )  # ty:ignore[call-non-callable]

        return {"loss": losses}

    @override
    def predict(
        self,
        episode: Episode,
        *,
        keys: AbstractSet[PredictionKey],
        tokenizers: ModuleDict | None = None,
        embedding: Tensor,
        attention_rollout: Tensor | None = None,
    ) -> TensorDict:
        del attention_rollout
        predictions: dict[PredictionKey, Prediction] = {}
        b, t = episode.input.batch_size

        timestep_indices = slice(1, None)

        if (key := PredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).apply(
                    lambda x: x.diff(dim=1), batch_size=[b, t - 1]
                ),
                timestep_indices=timestep_indices,
            )

        if keys & {
            PredictionKey.PREDICTION_VALUE,
            PredictionKey.PREDICTION_PROBS,
            PredictionKey.SUMMARY_EMBEDDINGS,
        }:
            features = (
                episode
                .index[1:]
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY))
                .parse(embedding)
                .get(k)
            )

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # ty:ignore[possibly-missing-attribute]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[possibly-missing-attribute, call-non-callable]
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.softmax(dim=-1)),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
