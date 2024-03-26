from functools import lru_cache

from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module

from cargpt.components.episode import (
    EpisodeBuilder,
    Index,
    Modality,
    SpecialToken,
    Timestep,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.components.objectives.copycat import (
    CopycatObjective,
)
from cargpt.utils import ModuleDict


class InverseDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModuleDict,
        losses: ModuleDict,
    ):
        super().__init__()
        self.heads = heads
        self.losses = losses

    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        b, t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        observation_summaries = (
            episode.index.select(  # pyright: ignore
                k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
            )
            .parse(embedding)
            .get(k)
            .detach()
        )

        # order: (o0, o1), (o1, o2), (o2, o3), ...
        features = rearrange(
            [observation_summaries[:, :-1], observation_summaries[:, 1:]],
            "i b t 1 d -> b t (i d)",
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[b, t - 1],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, :-1]  # pyright: ignore

        logits = logits.apply(Rearrange("b t d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t)"), batch_size=[])
        loss = logits.named_apply(
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),
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
        return CopycatObjective._build_attention_mask(index, timestep, legend)
