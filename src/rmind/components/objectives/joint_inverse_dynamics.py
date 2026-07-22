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
        decoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: CodeTargets,
        norm: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.decoder = decoder
        self.heads = heads
        self.losses = losses
        self.targets: CodeTargets = targets
        self.norm: Module | None = norm

    @override
    def train(self, mode: bool = True) -> "JointInverseDynamicsObjective":
        super().train(mode)
        self.tokenizer.eval()  # keep the frozen tokenizer's VQ EMA from updating
        return self

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        if self.norm is not None:
            embedding = self.norm(embedding)
        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        obs_summary = episode.index.select(k).parse(embedding).get(k)  # (b, t, 64, d)
        mask = episode.embeddings.get((Modality.UTILITY, "mask"))  # (b, t, 1, d)
        # single mask query cross-attends the observation_summary latents
        features = self.decoder({"query": mask, "context": obs_summary}).squeeze(-2)[
            :, :-1
        ]
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
            if self.norm is not None:
                embedding = self.norm(embedding)
            k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
            obs_summary = episode.index.select(k).parse(embedding).get(k)
            mask = episode.embeddings.get((Modality.UTILITY, "mask"))
            features = self.decoder({"query": mask, "context": obs_summary}).squeeze(
                -2
            )[:, :-1]
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
