from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import Any, cast, overload, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map, tree_map_with_path  # noqa: PLC2701

from rmind.components.base import TensorDictExport
from rmind.components.containers import ModuleDict
from rmind.components.episode import (
    Episode,
    EpisodeExport,
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
from rmind.utils.functional import gauss_prob, nan_padder


class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        encoder: InstanceOf[Module],
        heads: InstanceOf[ModuleDict],
        mask: InstanceOf[Tensor] | None = None,
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self._encoder: Module | None = encoder
        self._mask: Tensor | None = mask
        self._heads: ModuleDict = heads
        self._losses: ModuleDict | None = losses
        self._targets: Targets | None = targets

    @overload
    def forward(self, episode: Episode) -> TensorDict: ...

    @overload
    def forward(self, episode: EpisodeExport) -> TensorDictExport: ...

    @override
    def forward(
        self, episode: Episode | EpisodeExport
    ) -> TensorDict | TensorDictExport:
        if self._encoder is None:
            raise RuntimeError

        embedding = self._encoder(src=episode.embeddings_packed, mask=self._mask)

        if not (is_exporting := torch.compiler.is_exporting()):
            _b, _ = episode.input.batch_size

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
            ))

            observation_summary = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            waypoints = episode.input_embeddings[Modality.CONTEXT, "waypoints"][
                :, -1
            ].mean(dim=1, keepdim=True)

        else:
            observation_summary = embedding[
                :,
                episode.index[Modality.SPECIAL.value][
                    SpecialToken.OBSERVATION_SUMMARY.value
                ][-1],
            ]

            observation_history = embedding[
                :,
                episode.index[Modality.SPECIAL.value][
                    SpecialToken.OBSERVATION_HISTORY.value
                ][-1],
            ]

            waypoints = episode.input_embeddings[Modality.CONTEXT.value]["waypoints"][
                :, -1
            ].mean(dim=1, keepdim=True)

        features = rearrange(
            [observation_summary, observation_history, waypoints],
            "i b 1 d -> b 1 (i d)",
        )

        logits = self._heads.forward(features)

        if not is_exporting:

            def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:
                match nk:
                    case (Modality.CONTINUOUS, _):
                        return x[..., 0]
                    case (Modality.DISCRETE, "turn_signal"):
                        return torch.argmax(x, dim=-1)
                    case _:
                        raise NotImplementedError

            return logits.named_apply(fn, nested_keys=True)

        def fn(kp: tuple[Any, ...], v: Tensor) -> Tensor:
            if kp[0].key == Modality.CONTINUOUS.value:
                return v[..., 0]

            if kp[0].key == Modality.DISCRETE.value and kp[1].key == "turn_signal":
                return torch.argmax(v, dim=-1)

            raise NotImplementedError

        return tree_map_with_path(fn, logits)

    @override
    def compute_losses(self, episode: Episode, encoder: Module) -> TensorDict:
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

        logits = self._heads.forward(features)
        targets = TensorDict(
            tree_map(episode.get, self._targets, is_leaf=lambda x: isinstance(x, tuple))
        ).auto_batch_size_(2)[:, -1]

        return self._losses.forward(  # pyright: ignore[reportOptionalMemberAccess]
            logits.apply(Rearrange("b 1 d -> b d"), batch_size=[]),
            targets.apply(Rearrange("b 1 -> b"), batch_size=[]),
        )

    @override
    def predict(  # noqa: C901, PLR0915
        self,
        episode: Episode,
        encoder: Module,
        *,
        result_keys: AbstractSet[PredictionResultKey],
        **kwargs: Any,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        result = {}

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self._heads.tree_paths())

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

            logits = self._heads.forward(features, batch_size=[b, 1])

            timestep_padder = nan_padder(pad=(t - 1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0]
                        case (Modality.DISCRETE, "turn_signal"):
                            return torch.argmax(x, dim=-1)
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.PREDICTION_STD) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return torch.sqrt(torch.exp(x[..., 1]))

                        case (Modality.DISCRETE, "turn_signal"):
                            return torch.ones_like(x[..., 0])  # placeholder
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            return gauss_prob(mean, mean=mean, std=std)

                        case (Modality.DISCRETE, "turn_signal"):
                            return F.softmax(x, dim=-1)
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            return -torch.log(gauss_prob(mean, mean=mean, std=std))

                        case (Modality.DISCRETE, "turn_signal"):
                            gt = episode.input[action_type][:, -1]
                            return F.cross_entropy(
                                x.squeeze(1), gt.squeeze(1).long(), reduction="none"
                            ).unsqueeze(1)

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return F.l1_loss(x, gt, reduction="none")

                        case (Modality.DISCRETE, "turn_signal"):
                            return F.l1_loss(
                                torch.argmax(x, dim=-1).float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
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
