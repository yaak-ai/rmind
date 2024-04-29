from functools import lru_cache

from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module
from typing_extensions import override

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
from cargpt.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from cargpt.utils import ModuleDict


class InverseDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.losses = losses

    @override
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
            episode.index.select(
                k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
            )
            .parse(embedding)
            .get(k)
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
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore[reportOptionalMemberAccess]
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        b, t = inputs.batch_size
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        attention = encoder.compute_attention_rollout(
            src=episode.packed_embeddings,
            mask=mask.data,
            drop_ratio=0.9,
        )

        attention = (
            # from relevant tokens
            episode.index.select((Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))  # pyright: ignore
            .parse(attention, dim=1)
            # to all tokens
            .apply(lambda x: episode.index.parse(x, dim=3))
            .apply(
                Rearrange("b t_from s_from t_to s_to -> b t_from t_to s_from s_to"),
                batch_size=[b, t, t],
            )
        )

        return TensorDict.from_dict({"attention": attention})

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
