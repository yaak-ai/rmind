from collections.abc import Set as AbstractSet
from typing import Literal, final, override

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
    PATCHES,
    CodeTargets,
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)


def _flatten_named(tree: object, prefix: str = "") -> dict[str, Tensor]:
    if isinstance(tree, Tensor):
        return {prefix: tree}
    assert isinstance(tree, dict)
    out: dict[str, Tensor] = {}
    for k, v in tree.items():
        out.update(_flatten_named(v, f"{prefix}/{k}" if prefix else str(k)))
    return out


def joint_vq_loss(
    *, features: Tensor, heads: ModuleDict, losses: ModuleDict, codes: object, g: int
) -> dict[str, Tensor]:
    """Per-quantizer categorical loss for joint residual-VQ action codes."""
    logits = tree_map(
        lambda lg: rearrange(lg, "b t (g c) -> b t g c", g=g), heads(features)
    )
    out: dict[str, Tensor] = {}
    for q in range(g):
        (out[f"quantizer_{q}"],) = tree_leaves(
            losses(
                tree_map(lambda lg, q=q: lg[..., q, :].flatten(0, 1), logits),
                tree_map(lambda c, q=q: c[..., q].flatten(), codes),
            )
        )
    return out


@final
class InverseDynamicsObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: InstanceOf[ModuleDict],
        decoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: CodeTargets,
        value_norm: InstanceOf[Module] | None = None,
        key_norm: InstanceOf[Module] | None = None,
        latent_decoder: InstanceOf[Module] | None = None,
        key_source: Literal["latent", "image"] = "latent",
        value_source: Literal["image", "latent"] = "image",
        aux_heads: InstanceOf[ModuleDict] | None = None,
        aux_losses: InstanceOf[ModuleDict] | None = None,
        aux_targets: Targets | None = None,
        readout: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.decoder = decoder
        # value_source="latent": the second read is the NEXT latent L(t+1) instead of I(t+1)
        self.value_source: Literal["image", "latent"] = value_source
        # auxiliary per-step targets at t+1 (e.g. speed), read from the same features
        self.aux_heads: ModuleDict | None = aux_heads
        self.aux_losses: ModuleDict | None = aux_losses
        self.aux_targets: Targets | None = aux_targets
        # W1: queries read L(t) first (K=V=L), then `decoder` reads I(t+1) with K=V=I
        self.latent_decoder: Module | None = latent_decoder
        self.key_source: Literal["latent", "image"] = key_source
        self.heads = heads
        self.losses = losses
        self.targets: CodeTargets = targets
        self.value_norm: Module | None = value_norm
        self.key_norm: Module | None = key_norm
        self.readout: int = readout

    @override
    def train(self, mode: bool = True) -> "InverseDynamicsObjective":
        super().train(mode)
        self.tokenizer.eval()  # keep the frozen tokenizer's VQ EMA from updating
        return self

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, latent: Tensor | None = None
    ) -> Metrics:
        assert latent is not None
        source = latent if self.value_source == "latent" else episode.get(PATCHES)
        value = self.value_norm(source) if self.value_norm is not None else source
        lat = self.key_norm(latent) if self.key_norm is not None else latent
        lat, value = lat[:, :-1], value[:, 1:]  # L(t), next frame I(t+1) or next latent L(t+1)
        queries = episode.embeddings.get((Modality.UTILITY, "inv"))[:, :-1]  # t = 0..T-2
        if self.latent_decoder is not None:
            queries = self.latent_decoder({"query": queries, "key": lat, "value": lat})
        key = lat if self.key_source == "latent" else value
        features = self.decoder({"query": queries, "key": key, "value": value})
        features = features[:, :, self.readout]

        (tokenizer,) = tree_leaves(self.tokenizer)
        g = tokenizer.quantizer.num_quantizers
        with torch.no_grad():
            codes = tree_map(
                lambda path: episode.get(path)[:, :-1],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
        losses = joint_vq_loss(
            features=features, heads=self.heads, losses=self.losses, codes=codes, g=g
        )
        if self.aux_heads is not None:
            assert self.aux_losses is not None
            assert self.aux_targets is not None
            aux_logits = self.aux_heads(features)
            aux_targets = tree_map(
                lambda path: episode.get(path)[:, 1:],
                self.aux_targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
            aux = self.aux_losses(
                tree_map(lambda lg: lg.flatten(0, 1), aux_logits),
                tree_map(lambda tg: tg.flatten(), aux_targets),
            )
            losses.update(_flatten_named(aux))
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
        self.query_norm: Module | None = query_norm

    def _features(self, *, episode: Episode, embedding: Tensor) -> Tensor:

        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        obs_summary = episode.index.select(k).parse(embedding).get(k)  # (b, t, 64, d)
        query = episode.get(self.query)  # (b, t, p, d)
        if self.query_norm is not None:
            query = self.query_norm(query)
        key_inv = self.condition({
            "query": query,
            "key": obs_summary,
            "value": obs_summary,
        })
        mask = episode.embeddings.get((Modality.UTILITY, "mask"))[
            :, :, [1]
        ]  # (b, t, 1, d)
        features = self.decoder({"query": mask, "key": key_inv, "value": query})
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
