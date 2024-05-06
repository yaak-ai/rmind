from collections.abc import Sequence
from functools import lru_cache, partial

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module
from torch.nn import functional as F
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
from cargpt.components.objectives.common import PredictionResultKey
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
            episode.index.select(  # pyright: ignore[reportAttributeAccessIssue]
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

        loss = logits.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
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
        *,
        result_keys: Sequence[PredictionResultKey] = tuple(PredictionResultKey),  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> TensorDict:
        b, t = inputs.batch_size
        result = TensorDict({}, batch_size=[b, t])

        if not result_keys:
            return result

        episode = episode_builder.build_episode(inputs)

        if (result_key := PredictionResultKey.PREDICTION) in result_keys:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

            observation_summaries = (
                episode.index.select(  # pyright: ignore[reportAttributeAccessIssue]
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

            prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
            prediction = prediction_tokens.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                lambda k, v: episode_builder.detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                nested_keys=True,
            )

            # insert NaN at index -1 to indicate no prediction for t=-1
            padder = partial(F.pad, pad=(0, 1), mode="constant", value=torch.nan)
            prediction = prediction.apply(padder, batch_size=[b, t])

            result[result_key] = prediction

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            ground_truth = episode.inputs.select(*(k for k, _ in self.heads.flatten()))

            result[result_key] = ground_truth

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            attention = encoder.compute_attention_rollout(
                src=episode.packed_embeddings,
                mask=mask.data,
                drop_ratio=0.9,
            )

            attention = (
                # from relevant tokens
                episode.index.select((  # pyright: ignore[reportAttributeAccessIssue]
                    Modality.SPECIAL,
                    SpecialToken.OBSERVATION_SUMMARY,
                ))
                .parse(attention, dim=1)
                # to all tokens
                .apply(lambda x: episode.index.parse(x, dim=3))
                .apply(
                    Rearrange("b t_from s_from t_to s_to -> b t_from t_to s_from s_to"),
                    batch_size=[b, t, t],
                )
            )

            result[result_key] = attention

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
        ).clone()  # pyright: ignore[reportAttributeAccessIssue]

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
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
