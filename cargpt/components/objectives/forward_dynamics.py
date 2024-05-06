from collections.abc import Sequence
from functools import lru_cache, partial
from typing import TYPE_CHECKING

import torch
from einops import pack
from einops.layers.torch import Rearrange
from jaxtyping import Float
from more_itertools import partition
from tensordict import TensorDict, merge_tensordicts
from torch import Tensor
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
from cargpt.utils import ModuleDict

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class ForwardDynamicsPredictionObjective(Module):
    def __init__(
        self,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
    ) -> None:
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
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        # all but last timestep
        index = episode.index[:-1]  # pyright: ignore

        observations: TensorDict = index.select(
            *episode.timestep.keys(TokenType.OBSERVATION)
        ).parse(embedding)

        observation_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        action_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        features = observations.apply(
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

        logits = features.named_apply(  # pyright: ignore[reportOptionalMemberAccess]
            lambda k, v: self.heads.get(k)(v),
            nested_keys=True,
        )

        non_image_keys, image_keys = partition(
            lambda k: k[0] is Modality.IMAGE,
            logits.keys(include_nested=True, leaves_only=True),
        )
        labels = merge_tensordicts(
            episode.embedded_nope.select(*image_keys),
            episode.tokenized.select(*non_image_keys),
        )[:, 1:]  # all but first timestep

        logits = logits.apply(Rearrange("b t s d -> (b t s) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t s ... -> (b t s) ..."), batch_size=[])

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
            # all but last timestep
            index = episode.index[:-1]  # pyright: ignore

            observations: TensorDict = (
                index.select(*episode.timestep.keys(TokenType.OBSERVATION))
                .exclude(Modality.IMAGE)
                .parse(embedding)
            )

            observation_summary: Float[Tensor, "b t 1 d"] = (
                index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            action_summary: Float[Tensor, "b t 1 d"] = (
                index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            features = observations.apply(
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

            logits = features.named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                lambda k, v: self.heads.get(k)(v),
                nested_keys=True,
            )

            prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
            prediction = prediction_tokens.named_apply(
                lambda k, v: episode_builder.detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                nested_keys=True,
            ).apply(Rearrange("b t 1 -> b t"))

            # insert NaN at index 0 to indicate no prediction for t=0
            padder = partial(F.pad, pad=(1, 0), mode="constant", value=torch.nan)
            prediction = prediction.float().apply(padder, batch_size=[b, t])

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
                episode.index.select(  # pyright: ignore
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
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_NOT_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore[reportAttributeAccessIssue]
        )

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
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_attend(
                    current,
                    current,
                )
                ._do_attend(
                    current,
                    past,
                )
                ._do_not_attend(
                    current_observations,
                    current_actions,
                )
                ._do_not_attend(
                    current_observations,
                    current_action_summary,
                )
                ._do_not_attend(
                    current_observation_summary,
                    current_actions,
                )
                ._do_not_attend(
                    current_observation_summary,
                    current_action_summary,
                )
                ._do_not_attend(
                    current_observation_history,
                    current_actions,
                )
                ._do_not_attend(
                    current_observation_history,
                    current_action_summary,
                )
            )

        return mask
