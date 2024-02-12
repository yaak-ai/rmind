from functools import lru_cache

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module, ModuleDict

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


class CopycatObjective(Module):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    def __init__(self, **streams: Module):
        super().__init__()

        self.streams = ModuleDict(streams)

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        _b, _t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        embeddings = (
            episode.index[-1]  # pyright: ignore
            .select(
                (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            )
            .parse(embedding)
        )

        loss = TensorDict.from_dict({
            name: stream(episode, embeddings) for name, stream in self.streams.items()
        })

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ):
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore
        )

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore
            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            past_observations = past.select(*timestep.keys(TokenType.OBSERVATION))

            mask = (
                mask._do_attend(
                    current_observations,
                    current_observations,
                )
                ._do_attend(
                    current_observation_summary,
                    current_observations,
                )
                ._do_attend(
                    current_observation_summary,
                    current_observation_summary,
                )
                ._do_attend(
                    current_observation_history,
                    current_observations,
                )
                ._do_attend(
                    current_observation_history,
                    current_observation_history,
                )
                ._do_attend(
                    current_actions,
                    current_actions,
                )
                ._do_attend(
                    current_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    current_actions,
                )
                ._do_attend(
                    current_action_summary,
                    current_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    current_action_summary,
                )
                ._do_attend(
                    current_observation_history,
                    past_observations,
                )
            )

        return mask


class MemoryExtractionStream(Module):
    def __init__(
        self,
        delta_tokenizers: ModuleDict,
        heads: ModuleDict,
        loss: Module,
    ):
        super().__init__()

        self.delta_tokenizers = delta_tokenizers
        self.heads = heads
        self.loss = loss

    def forward(self, episode: Episode, embeddings: TensorDict) -> TensorDict:
        b, _t = episode.embedded.batch_size

        features = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        ))

        logits = TensorDict(
            {
                (modality, name): self.heads[modality][name](features)  # pyright: ignore
                for (modality, name) in episode.timestep.keys(TokenType.ACTION)
            },
            batch_size=[b, 1],
        )

        deltas = episode.inputs.select(*logits.keys(True, True))[:, -2:].apply(  # pyright: ignore
            torch.diff, batch_size=[b, 1]
        )

        labels = deltas.named_apply(
            lambda nested_key, tensor: self.delta_tokenizers.get(nested_key)(tensor),
            nested_keys=True,
        )

        logits = logits.flatten(1, 2)
        labels = labels.flatten(0, 1)

        return logits.apply(self.loss, labels, batch_size=[])


class PolicyStream(Module):
    def __init__(self, heads: ModuleDict, loss: Module):
        super().__init__()

        self.heads = heads
        self.loss = loss

    def forward(self, episode: Episode, embeddings: TensorDict) -> TensorDict:
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

        logits = TensorDict(
            {
                (modality, name): self.heads[modality][name](features)  # pyright: ignore
                for (modality, name) in episode.timestep.keys(TokenType.ACTION)
            },
            batch_size=[],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, -1]  # pyright: ignore

        logits = logits.apply(Rearrange("b 1 d -> b d"), batch_size=[])
        labels = labels.apply(Rearrange("b 1 -> b"), batch_size=[])

        return logits.apply(self.loss, labels)
