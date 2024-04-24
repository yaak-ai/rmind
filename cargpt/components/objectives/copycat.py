from functools import lru_cache

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module
from typing_extensions import override

from cargpt.components.episode import (
    Episode,
    EpisodeBuilder,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from cargpt.utils.containers import ModuleDict


class CopycatObjective(Module):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    def __init__(
        self,
        *,
        memory_extraction: Module | None = None,
        policy: Module | None = None,
    ):
        super().__init__()

        self.streams = ModuleDict(**{
            name: stream
            for name, stream in (
                ("memory_extraction", memory_extraction),
                ("policy", policy),
            )
            if stream is not None
        })

    @override
    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        loss = TensorDict.from_dict({
            name: stream.forward(episode, embedding)
            for name, stream in self.streams.items()
        })

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        return TensorDict.from_dict({
            name: stream.predict(episode, embedding)
            for name, stream in self.streams.items()
        })

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = ForwardDynamicsPredictionObjective._build_attention_mask(
            index, timestep, legend
        ).clone()  # pyright: ignore

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore[reportIndexIssue]
            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            past_actions = past.select(*timestep.keys(TokenType.ACTION))
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_not_attend(
                    current_observations,
                    past_actions,
                )
                ._do_not_attend(
                    current_observations,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_observation_summary,
                    past_actions,
                )
                ._do_not_attend(
                    current_observation_summary,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_observation_history,
                    past_actions,
                )
                ._do_not_attend(
                    current_observation_history,
                    past_action_summary,
                )
            )

        return mask


class MemoryExtractionStream(Module):
    def __init__(
        self,
        *,
        delta_tokenizers: ModuleDict,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        delta_detokenizers: ModuleDict | None = None,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.delta_tokenizers = delta_tokenizers
        self.delta_detokenizers = delta_detokenizers

    @override
    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
        b, t = episode.embedded.batch_size

        features = (
            episode.index[1:]  # pyright: ignore[reportIndexIssue]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[b, t - 1],
        )

        deltas = episode.inputs.select(*logits.keys(True, True)).apply(  # pyright: ignore[reportArgumentType]
            lambda tensor: torch.diff(tensor, n=1, dim=-1),
            batch_size=[b, t - 1],
        )

        labels = deltas.named_apply(
            lambda nested_key, tensor: self.delta_tokenizers.get(nested_key)(tensor),
            nested_keys=True,
        )

        logits = logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t -> (b t)"), batch_size=[])

        loss = logits.named_apply(
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore[reportOptionalMemberAccess]
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
        if self.delta_detokenizers is None:
            msg = "delta_detokenizers missing"
            raise RuntimeError(msg)

        b, t = episode.embedded.batch_size

        features = (
            episode.index[1:]  # pyright: ignore[reportIndexIssue]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[b, t - 1],
        )

        prediction_tokens = logits.apply(
            lambda x: Categorical(logits=x, validate_args=True).sample()
        )

        prediction = prediction_tokens.named_apply(
            lambda nested_key, tensor: self.delta_detokenizers.get(nested_key)(tensor),  # pyright: ignore[reportOptionalMemberAccess]
            nested_keys=True,
        )

        ground_truth = episode.inputs.select(*logits.keys(True, True)).apply(  # pyright: ignore[reportArgumentType]
            lambda tensor: torch.diff(tensor, n=1, dim=-1),
            batch_size=[b, t - 1],
        )

        return TensorDict.from_dict(
            {
                "ground_truth": ground_truth,
                "prediction": prediction.apply(Rearrange("b t 1 -> b t")),
            },
            batch_size=[b, t - 1],
        )


class PolicyStream(Module):
    def __init__(self, heads: ModuleDict, losses: ModuleDict | None = None):
        super().__init__()

        self.heads = heads
        self.losses = losses

    @override
    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
        embeddings = (
            episode.index[-1]  # pyright: ignore[reportIndexIssue]
            .select(
                (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        )).detach()  # NOTE: equivalent to stop gradient layer in paper

        observation_summary = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))

        features = rearrange(
            [observation_summary, observation_history],
            "i b 1 d -> b 1 (i d)",
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, -1]  # pyright: ignore[reportArgumentType]

        logits = logits.apply(Rearrange("b 1 d -> b d"), batch_size=[])
        labels = labels.apply(Rearrange("b 1 -> b"), batch_size=[])

        loss = logits.named_apply(
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore[reportOptionalMemberAccess]
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])
