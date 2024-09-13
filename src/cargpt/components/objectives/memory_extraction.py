from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import override

from einops.layers.torch import Rearrange
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.nn import Module
from torch.utils._pytree import tree_map

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
from cargpt.components.objectives.base import Objective, PredictionResultKey
from cargpt.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from cargpt.utils.containers import ModuleDict
from cargpt.utils.functional import nan_padder


class MemoryExtractionObjective(Objective):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    def __init__(
        self,
        *,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        targets: DictConfig | None = None,
        delta_tokenizers: ModuleDict,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.targets = OmegaConf.to_container(targets)
        self.delta_tokenizers = delta_tokenizers

    @override
    def forward(
        self, inputs: TensorDict, episode_builder: EpisodeBuilder, encoder: Module
    ) -> TensorDict:
        if self.losses is None:
            raise RuntimeError

        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        b, t = episode.embedded.batch_size

        features = (
            episode.index[1:]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = self.heads.forward(features, batch_size=[b, t - 1])
        deltas = TensorDict(tree_map(lambda f: f(episode), self.targets)).apply(
            lambda x: x.diff(dim=1), batch_size=[b, t - 1]
        )
        targets = self.delta_tokenizers(deltas)
        loss = self.losses(
            logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[]),
            targets.apply(Rearrange("b t 1 -> (b t)"), batch_size=[]),
        )

        return TensorDict({"loss": loss})

    @override
    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: AbstractSet[PredictionResultKey] | None = None,
    ) -> TensorDict:
        if result_keys is None:
            result_keys = frozenset(PredictionResultKey)

        b, t = inputs.batch_size
        result = TensorDict({}, batch_size=[b, t])

        episode = episode_builder.build_episode(inputs)

        timestep_padder = nan_padder(pad=(1, 0), dim=1)

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = (
                episode.inputs.select(*self.heads.tree_paths())
                .apply(lambda x: x.diff(dim=1), batch_size=[b, t - 1])
                .apply(timestep_padder, batch_size=[b, t])
            )

        if result_keys & {
            PredictionResultKey.PREDICTION,
            PredictionResultKey.PREDICTION_PROBS,
        }:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

            features = (
                episode.index[1:]
                .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
                .parse(embedding)
                .get(k)
            )

            logits = self.heads.forward(features, batch_size=[b, t - 1])

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: self.delta_tokenizers.get(k).invert(v),
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

        return result

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,  # pyright: ignore[reportGeneralTypeIssues]
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:  # pyright: ignore[reportGeneralTypeIssues]
        mask = ForwardDynamicsPredictionObjective._build_attention_mask(
            index, timestep, legend
        ).clone()

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]
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
                mask._do_not_attend(current_observations, past_actions)
                ._do_not_attend(current_observations, past_action_summary)
                ._do_not_attend(current_observation_summary, past_actions)
                ._do_not_attend(current_observation_summary, past_action_summary)
                ._do_not_attend(current_observation_history, past_actions)
                ._do_not_attend(current_observation_history, past_action_summary)
            )

        return mask
