from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    PATCHES,
    CodeTargets,
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)


@final
class MemoryObjective(Objective):
    """Read L[τ], predict the forward residual a_{τ+1}-a_τ (per-channel diff codes)."""

    @validate_call
    def __init__(
        self,
        *,
        decoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: CodeTargets,
        value_norm: InstanceOf[Module] | None = None,
        readout: int = 0,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.heads = heads
        self.losses = losses
        self.targets: CodeTargets = targets
        self.value_norm: Module | None = value_norm
        self.readout: int = readout

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, latent: Tensor | None = None
    ) -> Metrics:
        assert latent is not None
        patches = episode.get(PATCHES)
        value = self.value_norm(patches) if self.value_norm is not None else patches
        queries = episode.embeddings.get((Modality.UTILITY, "mem"))
        features = self.decoder({"query": queries, "key": latent, "value": value})
        features = features[:, :, [self.readout]][:, :-1]

        _, t = episode.input.batch_size
        target = tree_map(
            lambda k: episode.get(k)[:, : t - 1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        losses = self.losses(
            tree_map(Rearrange("b t 1 d -> (b t) d"), self.heads(features)),
            tree_map(Rearrange("b t -> (b t)"), target),
        )
        return {"loss": losses}

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        latent: Tensor | None = None,
    ) -> TensorDict:
        return TensorDict({}, batch_size=[])


@final
class MemoryExtractionObjective(Objective):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)."""

    @validate_call
    def __init__(
        self,
        *,
        condition: InstanceOf[Module],
        query: tuple[str, ...],
        query_norm: InstanceOf[Module] | None = None,
        decoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.condition: Module = condition
        self.query: tuple[str, ...] = query
        self.query_norm: Module | None = query_norm
        self.decoder = decoder
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

    def _features(self, *, episode: Episode, embedding: Tensor) -> Tensor:

        obs_history = (
            episode
            .index[1:]
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )
        query = episode.get(self.query)[:, 1:]  # (b, t-1, p, d)
        if self.query_norm is not None:
            query = self.query_norm(query)
        # observation_history routes attention over patches without entering the values
        key_mem = self.condition({
            "query": query,
            "key": obs_history,
            "value": obs_history,
        })
        mask = episode.embeddings.get((Modality.UTILITY, "mask"))[:, 1:, [2]]
        return self.decoder({"query": mask, "key": key_mem, "value": query})

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode=episode, embedding=embedding)

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
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        b, t = episode.input.batch_size

        timestep_index = slice(1, None)
        time_index = torch.arange(t).expand(b, -1)[:, timestep_index]

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).apply(
                    lambda x: x.diff(dim=1), batch_size=[b, t - 1]
                ),
                time_index=time_index,
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
        }:
            features = self._features(episode=episode, embedding=embedding)

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # ty:ignore[unresolved-attribute]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[unresolved-attribute, call-non-callable]
                        nested_keys=True,
                    ),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.softmax(dim=-1)),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
