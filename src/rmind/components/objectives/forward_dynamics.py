from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from einops import rearrange, repeat
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
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)

type Features = dict[str, dict[str, dict[str, Tensor]]]


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        condition: InstanceOf[Module],
        query: tuple[str, ...],
        query_norm: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.condition: Module = condition
        self.query: tuple[str, ...] = query
        self.query_norm: Module | None = query_norm
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.patch_pos_embed: Module | None = patch_pos_embed

    def _features(self, *, episode: Episode, embedding: Tensor) -> Features:

        summary = (
            episode
            .index[:-1]  # all but last timestep
            .select(
                k_os := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
                k_as := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY),
            )
            .parse(embedding)
        )
        context = torch.cat(
            [summary.get(k_os), summary.get(k_as)], dim=-2
        )  # (b, t-1, 65, d)

        patches = episode.get(self.query)[:, :-1]  # (b, t-1, p, d)
        query = patches.clone()

        if self.query_norm is not None:
            query = self.query_norm(query)

        mask_tokens = repeat(
            episode.embeddings.get((Modality.UTILITY, "mask"))[:, 1:, [0]],
            "b t 1 d -> b t n d",
            n=patches.shape[-2],
        )
        if self.patch_pos_embed is not None:
            mask_tokens = self.patch_pos_embed(mask_tokens)

        return {
            Modality.SUMMARY: {
                SummaryToken.OBSERVATION_SUMMARY: {
                    "query": mask_tokens,
                    "key": self.condition({
                        "query": query,
                        "key": context,
                        "value": context,
                    }),
                    "value": query,
                }
            }
        }

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        logits = self.heads(
            self._features(episode=episode, embedding=embedding),
            is_leaf=lambda x: isinstance(x, dict) and "query" in x,
        )

        targets = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
        )  # ty:ignore[call-non-callable]

        return {
            "loss": losses,
            "_artifacts": {"last_embeddings": logits, "last_targets": targets},
        }

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
            predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(embedding)

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]


@final
class ForwardDynamicsObjective(Objective):
    """Predict the next frame from the latent and gram-anchor it (W0: direct, W1: via decoder)."""

    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: Targets,
        decoder: InstanceOf[Module] | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.losses = losses
        self.targets: Targets = targets
        self.decoder = decoder
        self.patch_pos_embed = patch_pos_embed

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, latent: Tensor | None = None
    ) -> Metrics:
        assert latent is not None
        if self.decoder is not None:
            query = repeat(
                episode.embeddings.get((Modality.UTILITY, "latent")),
                "b t 1 d -> b t p d",
                p=latent.shape[-2],
            )
            query = self.patch_pos_embed(query)
            pred = self.decoder({"query": query, "key": latent, "value": latent})
        else:
            pred = latent

        logits = self.heads(pred)
        target = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        losses = self.losses(
            tree_map(lambda lg: rearrange(lg[:, :-1], "b t s d -> (b t s) d"), logits),
            tree_map(lambda t: rearrange(t, "b t s ... -> (b t s) ..."), target),
        )
        # nest under the heads structure ({summary: {observation_summary: ...}}) so the
        # patch-similarity logger can index both predict and target by [summary, observation_summary]
        return {
            "loss": losses,
            "_artifacts": {"last_embeddings": logits, "last_targets": target},
        }

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
