from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import torch
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
    Prediction,
    PredictionKey,
    Targets,
)


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
        ).parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]

        observation_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        action_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
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

        logits = self.heads(features.to_dict())  # pyright: ignore[reportOptionalMemberAccess]

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
        keys: AbstractSet[PredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        predictions: dict[PredictionKey, Prediction] = {}
        b, t = episode.input.batch_size

        if (key := PredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).exclude(
                    Modality.IMAGE
                ),
                timestep_indices=slice(None),
            )

        if keys & {
            PredictionKey.PREDICTION_VALUE,
            PredictionKey.PREDICTION_PROBS,
            PredictionKey.SCORE_LOGPROB,
            PredictionKey.SCORE_L1,
            PredictionKey.SUMMARY_EMBEDDINGS,
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
            ).parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]

            observation_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
                .get(k)
            )

            action_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
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

            logits = TensorDict(self.heads(features.to_dict()), batch_size=[b, t - 1])  # pyright: ignore[reportOptionalMemberAccess]

            # all but first
            timestep_indices = slice(1, None)

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits.exclude(Modality.IMAGE)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                            nested_keys=True,
                        )
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.exclude(Modality.IMAGE).apply(
                        lambda x: x.softmax(dim=-1)
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_LOGPROB) in keys:
                """Finds log prob of the correct token at each timestep."""
                predictions[key] = Prediction(
                    value=(
                        logits.exclude(Modality.IMAGE)
                        .apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportOptionalMemberAccess]
                        .apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, timestep_indices],
                        )
                        .apply(lambda x: -torch.log(x))  # pyright: ignore[reportOptionalMemberAccess]
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits.exclude(Modality.IMAGE)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                            nested_keys=True,
                        )
                        .apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                            episode.input[:, timestep_indices],
                            nested_keys=True,
                        )
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(  # pyright: ignore[reportAttributeAccessIssue]
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)

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
            current_observation_summary = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_action_summary = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_attend(current, current)  # pyright: ignore[reportArgumentType]
                .do_attend(current, past)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, current_action_summary)  # pyright: ignore[reportArgumentType]
            )

        return mask
