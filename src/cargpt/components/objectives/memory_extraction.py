from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import override

from einops.layers.torch import Rearrange
from optree import tree_map
from pydantic import ConfigDict, validate_call
from tensordict import TensorDict
from torch.nn import Module

from cargpt.components.episode import (
    Episode,
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
from cargpt.components.objectives.base import Objective, PredictionResultKey, Targets
from cargpt.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from cargpt.utils import ModuleDict
from cargpt.utils.functional import nan_padder


class MemoryExtractionObjective(Objective):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        targets: Targets | None = None,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.targets = targets

    @override
    def forward(self, episode: Episode, encoder: Module) -> TensorDict:
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.embeddings_packed, mask=mask.data)

        b, t = episode.input.batch_size

        features = (
            episode.index[1:]  # pyright: ignore[reportIndexIssue]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = self.heads.forward(features, batch_size=[b, t - 1])
        targets = TensorDict.from_dict(
            tree_map(
                episode.get,  # pyright: ignore[reportAttributeAccessIssue]
                self.targets,  # pyright: ignore[reportArgumentType]
                is_leaf=lambda x: isinstance(x, tuple),
            )
        )[:, : t - 1]

        loss = self.losses.forward(  # pyright: ignore[reportOptionalMemberAccess]
            logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[]),
            targets.apply(Rearrange("b t 1 -> (b t)"), batch_size=[]),
        )

        return TensorDict.from_dict({"loss": loss})

    @override
    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        result = TensorDict({}, batch_size=[b, t])

        timestep_padder = nan_padder(pad=(1, 0), dim=1)

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = (
                episode.input.select(*self.heads.tree_paths())
                .apply(lambda x: x.diff(dim=1), batch_size=[b, t - 1])
                .apply(timestep_padder, batch_size=[b, t])
            )

        if result_keys & {
            PredictionResultKey.PREDICTION,
            PredictionResultKey.PREDICTION_PROBS,
        }:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.embeddings_packed, mask=mask.data)

            features = (
                episode.index[1:]  # pyright: ignore[reportIndexIssue]
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
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess]
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
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = ForwardDynamicsPredictionObjective._build_attention_mask(
            index, timestep, legend
        ).clone(recurse=True)  # pyright: ignore[reportAttributeAccessIssue]

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore[reportIndexIssue]
            current_observations = current.select(
                *timestep.keys_by_type[TokenType.OBSERVATION]
            )
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            past_actions = past.select(*timestep.keys_by_type[TokenType.ACTION])
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
