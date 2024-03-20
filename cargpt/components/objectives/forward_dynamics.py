from functools import lru_cache

import torch
from einops import pack
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
        observations_summary = summaries.get(
            (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
        )
        actions_summary = summaries.get((Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))

        # (obs, obs_summary, action_summary)
        observations_action_pairs = observations.apply(
            lambda obs: pack(
                [
                    obs,
                    torch.broadcast_to(observations_summary, obs.shape),
                    torch.broadcast_to(actions_summary, obs.shape),
                ],
                "b t p *",
            )[0]
        )

        # We take all timesteps except last
        observations.apply(lambda x: x[:, :-1])
        breakpoint()

        logits = observations.apply()
        logits = TensorDict(
            {
                (token, name): self.heads[token][name](
                    observations_action_pairs[token][name]
                )  # pyright: ignore
                for (token, name) in episode.timestep.keys(TokenType.OBSERVATION)
            },
            batch_size=[],
        )

        breakpoint()

        labels = episode.raw.select(*logits.keys(True, True))[:, 1:]  # pyright: ignore
        labels = labels.apply(lambda x: x.detach(), batch_size=[])  # SG
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
        return CopycatObjective._build_attention_mask(index, timestep, legend)
