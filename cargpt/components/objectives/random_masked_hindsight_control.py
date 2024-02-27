from functools import lru_cache

import numpy as np
import torch
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module, ModuleDict

from cargpt.components.episode import (
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


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: ModuleDict, loss: Module) -> None:
        # TODO
        raise NotImplementedError("update for new timestep structure")  # noqa: EM101

        super().__init__()
        self.heads = heads
        self.loss = loss

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        _, t = inputs.batch_size
        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(*episode.timestep.keys(TokenType.ACTION)).exclude(  # pyright: ignore
            Modality.SPECIAL
        )
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = embeddings.named_apply(
            lambda nested_key, tensor: self.heads[nested_key[0]][nested_key[1]](tensor),
            nested_keys=True,
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[
            :, masked_action_timestep_idx
        ]

        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])
        loss = logits.apply(self.loss, labels)

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore
        )

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            current, future = index[step], index[step + 1 :]  # pyright: ignore

            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            future_observations = future.select(*timestep.keys(TokenType.OBSERVATION))
            future_actions = future.select(*timestep.keys(TokenType.ACTION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            future_observation_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            mask = (
                mask._do_attend(
                    current_observations,
                    current_observations,
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
                    future_observations,
                    current_observation_summary,
                )
                ._do_attend(
                    future_actions,
                    current_observation_summary,
                )
                ._do_attend(
                    current_observations,
                    future_observation_summary,
                )
                ._do_attend(
                    current_actions,
                    future_observation_summary,
                )
            )

        return mask
