from functools import lru_cache

import numpy as np
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
from cargpt.components.objectives.copycat import (
    CopycatObjective,
)


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: ModuleDict, losses: ModuleDict | None) -> None:
        super().__init__()
        self.heads = heads
        self.losses = losses

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
        index = episode.index.select(  # pyright: ignore
            *episode.timestep.keys(TokenType.ACTION)
        )
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = embeddings.named_apply(
            lambda nested_key, tensor: self.heads.get(nested_key)(tensor),
            nested_keys=True,
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[
            :, masked_action_timestep_idx
        ]

        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])
        loss = logits.named_apply(
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss, "mask": mask}, batch_size=[])

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = CopycatObjective._build_attention_mask(index, timestep, legend).clone()  # pyright: ignore

        (t,) = index.batch_size  # pyright: ignore
        for step in range(t):
            past, current, future = (
                index[:step],  # pyright: ignore
                index[step],  # pyright: ignore
                index[step + 1 :],  # pyright: ignore
            )

            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            future_observation_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            past_observation_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            mask = (
                mask._do_attend(
                    current_actions,
                    future_observation_summary,
                )
                ._do_attend(
                    current_actions,
                    past_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    future_observation_summary,
                )
                ._do_attend(
                    current_action_summary,
                    past_observation_summary,
                )
            )

        return mask
