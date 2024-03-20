import operator
from functools import lru_cache

import torch
from einops import pack
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


class ForwardDynamicsPredictionObjective(Module):
    def __init__(self, heads: ModuleDict, losses: Module) -> None:
        super().__init__()
        self.heads = heads
        self.losses = losses

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        logit_bias: TensorDict,
    ) -> TensorDict:
        _b, _t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        index = episode.index.select(*episode.timestep.keys(TokenType.OBSERVATION))
        observations = index.parse(embedding)

        index = episode.index.select(  # pyright: ignore
            (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY),
        )
        summaries = index.parse(embedding)
        observations_summary = summaries.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))
        actions_summary = summaries.get((Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))

        # (obs, obs_summary, action_summary), all but last timestep
        observations_action_pairs = observations.apply(
            lambda obs: pack(
                [
                    obs[:, :-1],
                    torch.broadcast_to(observations_summary, obs.shape)[:, :-1],
                    torch.broadcast_to(actions_summary, obs.shape)[:, :-1],
                ],
                "b t p *",
            )[0]
        )

        logits = TensorDict(
            {
                (modality, name): head(observations_action_pairs[modality][name])  # pyright: ignore
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[],
        )

        image_obs = logits.select(Modality.IMAGE)
        image_labels = episode.raw.select(*image_obs.keys(True, True))[:, 1:]  # pyright: ignore
        image_labels = image_labels.apply(lambda x: x.detach(), batch_size=[])  # SG ?

        non_image_obs = logits.select((Modality.CONTINUOUS), (Modality.DISCRETE))
        non_image_labels = episode.tokenized.select(*non_image_obs.keys(True, True))[
            :, 1:
        ]  # pyright: ignore

        non_image_obs = non_image_obs.apply(operator.add, logit_bias, batch_size=[])
        non_image_obs = non_image_obs.apply(
            Rearrange("b t 1 d -> (b t) d"), batch_size=[]
        )
        non_image_labels = non_image_labels.apply(
            Rearrange("b t 1 -> (b t)"), batch_size=[]
        )

        logits = image_obs.update(non_image_obs)
        labels = image_labels.update(non_image_labels)

        loss = TensorDict(
            {
                (modality, name): criteria(
                    logits[modality][name], labels[modality][name]
                )  # pyright: ignore
                for (modality, name), criteria in self.losses.flatten()
            },
            batch_size=[],
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
        return CopycatObjective._build_attention_mask(index, timestep, legend)
