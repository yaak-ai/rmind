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
    """Predict the next frame from the latent and gram-anchor it (W0: direct, W1: via decoder).

    With an ``action_tokenizer``, the action a_t taken at each step enters as one extra key
    token e_t = proj(z_q(codes_t)), the frozen tokenizer's quantized embedding, so the
    decoder predicts I(t+1) from (L_t, a_t).
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: Targets,
        decoder: InstanceOf[Module] | None = None,
        action_tokenizer: InstanceOf[Module] | None = None,
        action_codes: tuple[str, ...] | None = None,
        action_projection: InstanceOf[Module] | None = None,
        aux_heads: InstanceOf[ModuleDict] | None = None,
        aux_losses: InstanceOf[ModuleDict] | None = None,
        aux_targets: Targets | None = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.losses = losses
        self.targets: Targets = targets
        self.decoder = decoder
        self.action_tokenizer: Module | None = (
            action_tokenizer.requires_grad_(False).eval()  # noqa: FBT003
            if action_tokenizer is not None
            else None
        )
        self.action_codes: tuple[str, ...] | None = action_codes
        self.action_projection: Module | None = action_projection
        # auxiliary per-step targets at t+1 (e.g. speed): a separate head on the latent,
        # [mean_n L_t ; e_t] (or mean_n L_t without an action token), not on the decoder
        self.aux_heads: ModuleDict | None = aux_heads
        self.aux_losses: ModuleDict | None = aux_losses
        self.aux_targets: Targets | None = aux_targets

    @override
    def train(self, mode: bool = True) -> "ForwardDynamicsObjective":
        super().train(mode)
        if self.action_tokenizer is not None:
            self.action_tokenizer.eval()  # keep the frozen tokenizer's VQ EMA fixed
        return self

    def _action_token(self, episode: Episode) -> Tensor:
        """(b, t, 1, d): projected quantized embedding of the action taken at each step."""
        assert self.action_tokenizer is not None
        assert self.action_codes is not None
        assert self.action_projection is not None
        quantizer = self.action_tokenizer.quantizer
        codes = episode.get(self.action_codes)
        b, t = codes.shape[:2]
        codes = codes.reshape(b, t, -1)[..., -quantizer.num_quantizers :]
        with torch.no_grad():
            z_q = quantizer.lookup(codes.reshape(b * t, -1))
        return self.action_projection(z_q).reshape(b, t, 1, -1)

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, latent: Tensor | None = None
    ) -> Metrics:
        assert latent is not None
        if self.decoder is not None:
            query = episode.embeddings.get((Modality.UTILITY, "fwd"))
            context = latent
            action = None
            if self.action_tokenizer is not None:
                action = self._action_token(episode).to(latent.dtype)
                context = torch.cat([latent, action], dim=-2)
            pred = self.decoder({"query": query, "key": context, "value": context})
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
        if self.aux_heads is not None:
            assert self.aux_losses is not None
            assert self.aux_targets is not None
            aux_in = latent.mean(dim=-2)  # (b, t, d)
            if action is not None:
                aux_in = torch.cat([aux_in, action.squeeze(-2)], dim=-1)  # (b, t, 2d)
            aux_logits = self.aux_heads(aux_in)  # from step t
            aux_targets = tree_map(
                lambda k: episode.get(k)[:, 1:],
                self.aux_targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
            aux = self.aux_losses(
                tree_map(lambda lg: lg[:, :-1].flatten(0, 1), aux_logits),
                tree_map(lambda tg: tg.flatten(), aux_targets),
            )
            # same nesting as the main targets -> forward_dynamics/continuous/speed in W&B
            assert not set(aux) & set(losses)
            losses = {**losses, **aux}
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
