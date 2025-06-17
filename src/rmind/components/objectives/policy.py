from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import Any, cast, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from optree import tree_map
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch.nn import Module
from torch.nn import functional as F

from rmind.components.episode import (
    Episode,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from rmind.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from rmind.components.objectives.base import Objective, PredictionResultKey, Targets
from rmind.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from rmind.utils import ModuleDict
from rmind.utils.functional import gauss_prob, nan_padder

ActionType = tuple[Modality, str]


class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

    @override
    def forward(self, episode: Episode, encoder: Module) -> TensorDict:
        mask = self.build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.embeddings_packed, mask=mask.mask)

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

        waypoints = episode.input_embeddings[Modality.CONTEXT, "waypoints"][:, -1].mean(
            dim=1, keepdim=True
        )

        features = rearrange(
            [observation_summary, observation_history, waypoints],
            "i b 1 d -> b 1 (i d)",
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
            logits.apply(Rearrange("b 1 d -> b d"), batch_size=[]),  # pyright: ignore[reportArgumentType]
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
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_STD,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            mask = self.build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.embeddings_packed, mask=mask.mask)
            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(
                    embedding
                )

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

            waypoints = episode.input_embeddings[Modality.CONTEXT, "waypoints"][
                :, -1
            ].mean(dim=1, keepdim=True)

            features = rearrange(
                [observation_summary, observation_history, waypoints],
                "i b 1 d -> b 1 (i d)",
            )

            logits = self.heads.forward(features, batch_size=[b, 1])

            timestep_padder = nan_padder(pad=(t - 1, 0), dim=1)


            result.update(
                {
                    result_key: logits.named_apply(
                        lambda action_type, x: PolicyObjective._process_logits(action_type, x, result_key, episode.input),
                        nested_keys=True,
                    ).apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportAttributeAccessIssue]
                    for result_key in result_keys & {
                        PredictionResultKey.PREDICTION_VALUE,
                        PredictionResultKey.PREDICTION_STD,
                        PredictionResultKey.PREDICTION_PROBS,
                        PredictionResultKey.SCORE_LOGPROB,
                        PredictionResultKey.SCORE_L1,
                    }
                }
            )

        return TensorDict(result).auto_batch_size_(2)

    @staticmethod
    def _process_logits(
        action_type: ActionType,
        x: torch.Tensor,
        prediction_result_key: PredictionResultKey,
        episode_input: TensorDict,
    ) -> torch.Tensor:
        gt = episode_input[action_type][:, -1]
        match prediction_result_key:
            case PredictionResultKey.PREDICTION_VALUE:
                match action_type:
                    case (Modality.CONTINUOUS, _):
                        return x[..., 0]
                    case (Modality.DISCRETE, "turn_signal"):
                        return torch.argmax(x, dim=-1)
                    case _:
                        msg = f"Invalid action type: {action_type}"
                        raise ValueError(msg)

            case PredictionResultKey.PREDICTION_STD:
                match action_type:
                    case (Modality.CONTINUOUS, _):
                        return torch.sqrt(torch.exp(x[..., 1]))
                    case (Modality.DISCRETE, "turn_signal"):
                        return torch.ones_like(x[..., 0])
                    case _:
                        msg = f"Invalid action type: {action_type}"
                        raise ValueError(msg)

            case PredictionResultKey.PREDICTION_PROBS:
                match action_type:
                    case (Modality.CONTINUOUS, _):
                        return gauss_prob(
                            x[..., 0],
                            mean=x[..., 0],
                            std=torch.sqrt(torch.exp(x[..., 1])),
                        )
                    case (Modality.DISCRETE, "turn_signal"):
                        return F.softmax(x, dim=-1) # all 3 categories probs
                    case _:
                        msg = f"Invalid action type: {action_type}"
                        raise ValueError(msg)

            case PredictionResultKey.SCORE_LOGPROB:
                match action_type:
                    case (Modality.CONTINUOUS, _):
                        return -torch.log(
                            gauss_prob(
                                gt,
                                mean=x[..., 0],
                                std=torch.sqrt(torch.exp(x[..., 1])),
                            )
                        )
                    case (Modality.DISCRETE, "turn_signal"):
                        return -torch.log(
                            F.softmax(x, dim=-1)[..., gt]
                        )
                    case _:
                        msg = f"Invalid action type: {action_type}"
                        raise ValueError(msg)

            case PredictionResultKey.SCORE_L1:
                match action_type:
                    case (Modality.CONTINUOUS, _):
                        return F.l1_loss(
                            x[..., 0],
                            gt,
                            reduction="none",
                        )
                    case (Modality.DISCRETE, "turn_signal"):
                        return F.l1_loss(
                            torch.argmax(x, dim=-1),
                            gt,
                            reduction="none",
                        )
                    case _:
                        msg = f"Invalid action type: {action_type}"
                        raise ValueError(msg)

            case _:
                msg = f"Invalid prediction result key: {prediction_result_key}"
                raise ValueError(msg)

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = cast(
            "AttentionMask",
            ForwardDynamicsPredictionObjective.build_attention_mask(
                index, timestep, legend
            ).clone(recurse=True),
        )

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
                mask.do_not_attend(current_observations, past_actions)
                .do_not_attend(current_observations, past_action_summary)
                .do_not_attend(current_observation_summary, past_actions)
                .do_not_attend(current_observation_summary, past_action_summary)
                .do_not_attend(current_observation_history, past_actions)
                .do_not_attend(current_observation_history, past_action_summary)
            )

        return mask
