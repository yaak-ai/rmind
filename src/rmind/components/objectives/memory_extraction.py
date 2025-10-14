from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch.nn import Module
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
from rmind.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from rmind.utils.functional import nan_padder


@final
class MemoryExtractionObjective(Objective):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)."""

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

        features = (
            episode.index[1:]  # pyright: ignore[reportCallIssue]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        logits = self.heads(features)

        _, t = episode.input.batch_size
        targets = tree_map(
            lambda k: episode.get(k)[:, : t - 1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b t 1 d -> (b t) d"), logits),
            tree_map(Rearrange("b t -> (b t)"), targets),
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
        b, t = episode.input.batch_size
        result = {}

        timestep_padder = nan_padder(pad=(1, 0), dim=1)

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = (
                episode.input.select(*self.heads.tree_paths())
                .apply(lambda x: x.diff(dim=1), batch_size=[b, t - 1])
                .apply(timestep_padder, batch_size=[b, t])
            )

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            )

            embedding = self.encoder(
                src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
            )  # pyright: ignore[reportOptionalCall]

            features = (
                episode.index[1:]  # pyright: ignore[reportCallIssue]
                .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
                .get(k)
            )

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportOptionalMemberAccess]
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportOptionalMemberAccess]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(  # pyright: ignore[reportAttributeAccessIssue]
                    embedding
                )

        return TensorDict(result).auto_batch_size_(2)

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        mask = ForwardDynamicsPredictionObjective.build_attention_mask(
            index, timestep, legend=legend
        ).clone(recurse=True)

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
            past_actions = past.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            past_action_summary = past.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_not_attend(current_observations, past_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, past_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, past_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, past_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, past_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, past_action_summary)  # pyright: ignore[reportArgumentType]
            )

        return mask
