from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import override

import torch
from einops import pack
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
from rmind.utils import ModuleDict
from rmind.utils.functional import nan_padder


class ForwardDynamicsPredictionObjective(Objective):
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

        # all but last timestep
        index = episode.index[:-1]

        observations: TensorDict = index.select(*[
            k
            for k in episode.timestep.keys_by_type[TokenType.OBSERVATION]
            if k != (Modality.CONTEXT, "waypoints")
        ]).parse(embedding)

        observation_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        action_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        features: TensorDict = observations.apply(  # pyright: ignore[reportAssignmentType]
            # pack: (obs[0], obs_summary, action_summary), (obs[1], obs_summary, action_summary), ...
            lambda obs: pack(
                [
                    obs,
                    observation_summary.broadcast_to(obs.shape),
                    action_summary.broadcast_to(obs.shape),
                ],
                "b t p *",
            )[0]
        )

        logits = self.heads.forward(features)

        targets = TensorDict(
            tree_map(
                episode.get,
                self.targets,  # pyright: ignore[reportArgumentType]
                is_leaf=lambda x: isinstance(x, tuple),
            )
        ).auto_batch_size_(2)[:, 1:]  # all but first timestep

        loss = self.losses.forward(  # pyright: ignore[reportOptionalMemberAccess]
            logits.apply(Rearrange("b t s d -> (b t s) d"), batch_size=[]),  # pyright: ignore[reportArgumentType]
            targets.apply(Rearrange("b t s ... -> (b t s) ..."), batch_size=[]),
        )

        return TensorDict({"loss": loss})  # pyright: ignore[reportArgumentType]

    @override
    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        result = {}
        b, t = episode.input.batch_size

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths()).exclude(
                Modality.IMAGE
            )

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            mask = self.build_attention_mask(episode.index, episode.timestep)
            attention = encoder.compute_attention_rollout(
                src=episode.embeddings_packed, mask=mask.mask, drop_ratio=0.9
            )

            result[result_key] = (
                # from relevant tokens
                episode.index.select(
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
                    (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY),
                )
                .parse(attention, dim=1)
                # to all tokens
                .apply(lambda x: episode.index.parse(x, dim=3))
                .apply(
                    Rearrange("b t_from s_from t_to s_to -> b t_from t_to s_from s_to"),
                    batch_size=[b, t, t],
                )
            )

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            mask = self.build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.embeddings_packed, mask=mask.mask)

            # all but last timestep
            index = episode.index[:-1]

            observations: TensorDict = (
                index.select(*episode.timestep.keys_by_type[TokenType.OBSERVATION])
                .exclude(Modality.IMAGE)
                .parse(embedding)
            )

            observation_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            action_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            features: TensorDict = observations.apply(  # pyright: ignore[reportAssignmentType]
                # pack: (obs[0], obs_summary, action_summary), (obs[1], obs_summary, action_summary), ...
                lambda obs: pack(
                    [
                        obs,
                        observation_summary.broadcast_to(obs.shape),
                        action_summary.broadcast_to(obs.shape),
                    ],
                    "b t p *",
                )[0]
            )

            logits = self.heads.forward(features)

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportAttributeAccessIssue]
                    .named_apply(
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess]
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                """Finds log prob of the correct token at each timestep."""
                result[result_key] = (
                    logits.apply(lambda x: x.softmax(dim=-1))
                    .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                        episode.input_tokens,
                    )
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
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
    def build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_NOT_ATTEND),
            legend=legend,
            device=index.device,
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
            current_actions = current.select(*timestep.keys_by_type[TokenType.ACTION])
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_attend(current, current)
                .do_attend(current, past)
                .do_not_attend(current_observations, current_actions)
                .do_not_attend(current_observations, current_action_summary)
                .do_not_attend(current_observation_summary, current_actions)
                .do_not_attend(current_observation_summary, current_action_summary)
                .do_not_attend(current_observation_history, current_actions)
                .do_not_attend(current_observation_history, current_action_summary)
            )

        return mask
