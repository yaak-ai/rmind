from collections.abc import Set as AbstractSet
from functools import lru_cache
from operator import itemgetter
from typing import Any, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from optree import tree_map
from pydantic import ConfigDict, validate_call
from tensordict import TensorDict
from torch.nn import Module
from torch.nn import functional as F

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
from cargpt.utils.functional import gauss_prob, nan_padder


class PolicyObjective(Objective):
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

        embeddings = (
            episode.index[-1]
            .select(
                (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        )).detach()  # NOTE: equivalent to stop gradient layer in paper

        observation_summary = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))

        features = rearrange(
            [observation_summary, observation_history], "i b 1 d -> b 1 (i d)"
        )

        logits = self.heads.forward(features)
        targets = TensorDict(
            tree_map(
                episode.get,
                self.targets,  # pyright: ignore[reportArgumentType]
                is_leaf=lambda x: isinstance(x, tuple),
            )
        ).auto_batch_size_(2)[:, -1]

        loss = self.losses.forward(  # pyright: ignore[reportOptionalMemberAccess]
            logits.apply(Rearrange("b 1 d -> b d"), batch_size=[]),
            targets.apply(Rearrange("b 1 -> b"), batch_size=[]),
        )

        return TensorDict({"loss": loss})  # pyright: ignore[reportArgumentType]

    @override
    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        **kwargs: Any,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        result = {}

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths())

        if result_keys & {
            PredictionResultKey.PREDICTION,
            PredictionResultKey.PREDICTION_STD,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.embeddings_packed, mask=mask.data)

            embeddings = (
                episode.index[[-1]]
                .select(
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
                )
                .parse(embedding)
            )

            observation_history = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            )).detach()  # NOTE: equivalent to stop gradient layer in paper

            observation_summary = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            features = rearrange(
                [observation_summary, observation_history], "i b t 1 d -> b t 1 (i d)"
            )

            logits = self.heads.forward(features, batch_size=[b, 1])

            timestep_padder = nan_padder(pad=(t - 1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                result[result_key] = logits.apply(itemgetter((..., 0))).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.PREDICTION_STD) in result_keys:
                result[result_key] = (
                    logits.apply(itemgetter((..., 1)))
                    .apply(lambda x: torch.sqrt(torch.exp(x)))  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(timestep_padder, batch_size=[b, t])
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(
                    lambda x: gauss_prob(
                        x[..., 0], mean=x[..., 0], std=torch.sqrt(torch.exp(x[..., 1]))
                    )
                ).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )
            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                result[result_key] = (
                    logits.named_apply(
                        lambda k, x: gauss_prob(
                            episode.input[:, -1][k],
                            mean=x[..., 0].squeeze(-1),
                            std=torch.sqrt(torch.exp(x[..., 1].squeeze(-1))),
                        ),
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.apply(itemgetter((..., 0)))
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(
                        lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                        episode.input,
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(
                    embedding
                )

        return TensorDict(result).auto_batch_size_(2)

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
        ).clone(recurse=True)

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]
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
