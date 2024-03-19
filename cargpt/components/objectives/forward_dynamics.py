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
    def __init__(self, heads: ModuleDict, loss: Module) -> None:
        super().__init__()
        self.heads = heads
        self.loss = loss

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
        index = episode.index.select(  # pyright: ignore
            (Modality.IMAGE, "cam_front_left"),
            (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY),
        )
        embeddings = index.parse(embedding)
        observations = embeddings.get(
            (
                Modality.IMAGE,
                "cam_front_left",
            )
        )

        observations_summary = embeddings.get(
            (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
        )
        observations_summary = torch.broadcast_to(
            observations_summary, observations.shape
        )

        actions = embeddings.get((Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
        actions = torch.broadcast_to(actions, observations.shape)

        observation_action_pairs, _ = pack(
            [observations[:, :-1], observations_summary[:, :-1], actions[:, :-1]],
            "b t p *",
        )

        logits = TensorDict(
            {
                (token, name): self.heads[token][name](
                    observation_action_pairs
                )  # pyright: ignore
                for (token, name) in episode.timestep.keys(TokenType.OBSERVATION)
                if token in (Modality.IMAGE)
            },
            batch_size=[],
        )

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
