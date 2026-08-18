from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import final, override

import torch
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)

type CodeTargets = Mapping[str, Mapping[str, tuple[str, ...]]]


@final
class JointInverseDynamicsObjective(Objective):
    """Inverse dynamics predicting a frozen tokenizer's joint residual-VQ codes."""

    @validate_call
    def __init__(
        self,
        *,
        tokenizer: InstanceOf[ModuleDict],
        condition: InstanceOf[Module],
        query: tuple[str, ...],
        decoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: CodeTargets,
        norm: InstanceOf[Module] | None = None,
        query_norm: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.condition = condition
        self.decoder = decoder
        self.heads = heads
        self.losses = losses
        self.targets: CodeTargets = targets
        self.query: tuple[str, ...] = query
        self.norm: Module | None = norm
        self.query_norm: Module | None = query_norm

    def _features(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)
        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        obs_summary = episode.index.select(k).parse(embedding).get(k)  # (b, t, 64, d)
        query = episode.get(self.query)  # (b, t, p, d)
        if self.query_norm is not None:
            query = self.query_norm(query)
        key_inv = self.condition({"query": query, "key": obs_summary, "value": obs_summary})
        mask = episode.embeddings.get((Modality.UTILITY, "mask"))[
            :, :, [1]
        ]  # (b, t, 1, d)
        features = self.decoder({
            "query": mask,
            "key": key_inv,
            "value": query,
        })
        return features.squeeze(-2)[:, :-1]

    @override
    def train(self, mode: bool = True) -> "JointInverseDynamicsObjective":
        super().train(mode)
        self.tokenizer.eval()  # keep the frozen tokenizer's VQ EMA from updating
        return self

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode=episode, embedding=embedding)
        (tokenizer,) = tree_leaves(self.tokenizer)
        g = tokenizer.quantizer.num_quantizers

        with torch.no_grad():
            codes = tree_map(
                lambda path: episode.get(path)[:, 1:],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )

        logits = tree_map(
            lambda lg: rearrange(lg, "b t (g c) -> b t g c", g=g), self.heads(features)
        )

        losses: dict[str, Tensor] = {}
        for q in range(g):
            (losses[f"quantizer_{q}"],) = tree_leaves(
                self.losses(
                    tree_map(lambda lg, q=q: lg[..., q, :].flatten(0, 1), logits),
                    tree_map(lambda c, q=q: c[..., q].flatten(), codes),
                )
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
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            gt = tree_map(
                episode.get, self.targets, is_leaf=lambda x: isinstance(x, tuple)
            )
            predictions[key] = Prediction(
                value=TensorDict(gt), timestep_indices=slice(None)
            )

        if ObjectivePredictionKey.PREDICTION_VALUE in keys:
            features = self._features(episode=episode, embedding=embedding)
            (tokenizer,) = tree_leaves(self.tokenizer)
            g = tokenizer.quantizer.num_quantizers

            # joint head: argmax each quantizer's categorical -> codes (B, T-1, G)
            value = tree_map(
                lambda lg: rearrange(lg, "b t (g c) -> b t g c", g=g).argmax(dim=-1),
                self.heads(features),
            )
            predictions[ObjectivePredictionKey.PREDICTION_VALUE] = Prediction(
                value=TensorDict(value), timestep_indices=slice(1, None)
            )

        return TensorDict(predictions).auto_batch_size_(2)
