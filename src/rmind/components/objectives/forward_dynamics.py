from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import torch
import itertools as it
from einops import pack
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.containers import ModuleDict
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
    TorchAttentionMaskLegend,
)
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    PredictionResultKey,
    Targets,
)
from rmind.utils.functional import nan_padder


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.heads = heads
        self.losses = losses
        self.targets = targets

        self._build_attention_mask = lru_cache(maxsize=2, typed=True)(
            self.build_attention_mask
        )

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # pyright: ignore[reportOptionalCall]

        index = episode.index[:-1]  # all but last timestep

        observations = index.select(
            *episode.timestep.get(TokenType.OBSERVATION)
            .exclude((Modality.CONTEXT, "waypoints"))
            .keys(include_nested=True, leaves_only=True)
        ).parse(embedding)

        logits = self.heads(observations.to_dict())

        targets = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
        )

        return {"loss": losses}

    @override
    def predict(
        self,
        episode: Episode,
        *,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        result = {}
        b, t = episode.input.batch_size

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths()).exclude(
                Modality.IMAGE
            )

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            )

            embedding = self.encoder(
                src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
            )  # pyright: ignore[reportOptionalCall]

            # all but last timestep
            index = episode.index[:-1]

            observations = index.select(
                *episode.timestep.get(TokenType.OBSERVATION)
                .exclude((Modality.CONTEXT, "waypoints"))
                .keys(include_nested=True, leaves_only=True)
            ).parse(embedding)

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

            features: TensorDict = observations.apply(
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

            logits = TensorDict(self.heads(features.to_dict()), batch_size=[b, t - 1])

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.argmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])
                    .named_apply(
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.softmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                """Finds log prob of the correct token at each timestep."""
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.softmax(dim=-1))
                    .apply(Rearrange("b t 1 d -> b t d"))
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                        episode.input_tokens,
                    )
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.argmax(dim=-1))
                    .named_apply(
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
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
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_NOT_ATTEND.value),
            legend=legend,
            device="cpu",
        )

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]
            current_observations = current.select(
                *timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            past_observations = past.select(
                *timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_attend(current, current)
                .do_attend(current, past)
                .do_not_attend(current_observations, current_actions)
                .do_not_attend(current_observations, current_action_summary)
                .do_not_attend(current_observations, current_observation_summary)
                .do_not_attend(current_observations, current_observation_history)
                .do_not_attend(past_observations, current_observation_summary)
                .do_not_attend(past_observations, current_observation_history)
                .do_not_attend(current_observation_summary, current_actions)
                .do_not_attend(current_observation_summary, current_action_summary)
                .do_not_attend(current_observation_history, current_actions)
                .do_not_attend(current_observation_history, current_action_summary)
            )

            # Only attend to Image modality
            for modality in timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                ):
                if modality is not Modality.Image:
                    mask = mask.do_not_attend(
                        current_observation_summary, index.select(modality)
                    )
                    mask = mask.do_not_attend(
                        current_observation_history, index.select(modality)
                    )

        return mask
