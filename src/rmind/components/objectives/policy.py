from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import Any, cast, final, overload, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map, tree_map_with_path  # noqa: PLC2701

from rmind.components.base import TensorTree
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
from rmind.utils.functional import (
    gauss_prob,
    nan_padder,
    non_zero_signal_with_threshold,
)


@final
class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        mask: InstanceOf[Tensor] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        match mask:
            case Tensor():
                self.register_buffer("_mask", mask, persistent=True)

            case None:
                self._mask = None

        self.heads = heads
        self.losses = losses
        self.targets = targets

        self._build_attention_mask = lru_cache(maxsize=2, typed=True)(
            self.build_attention_mask
        )

    @property
    def mask(self) -> Tensor | None:
        return self._mask

    @overload
    def forward(self, episode: Episode) -> TensorDict: ...

    @overload
    def forward(self, episode: EpisodeExport) -> TensorTree: ...

    @override
    def forward(self, episode: Episode | EpisodeExport) -> TensorDict | TensorTree:
        logits = self._compute_logits(episode)

        if isinstance(episode, Episode):

            def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:  # pyright: ignore[reportRedeclaration]
                match nk:
                    case (Modality.CONTINUOUS, _):
                        return x[..., 0]
                    case (Modality.DISCRETE, "turn_signal"):
                        return non_zero_signal_with_threshold(x).class_idx
                    case _:
                        raise NotImplementedError

            return TensorDict(logits).named_apply(fn, nested_keys=True)  # pyright: ignore[reportReturnType, reportArgumentType]

        def fn(kp: tuple[Any, ...], v: Tensor) -> Tensor:
            if kp[0].key == Modality.CONTINUOUS.value:
                return v[..., 0]

            if kp[0].key == Modality.DISCRETE.value and kp[1].key == "turn_signal":
                return non_zero_signal_with_threshold(v).class_idx

            raise NotImplementedError

        return tree_map_with_path(fn, logits)

    def _compute_logits(self, episode: Episode | EpisodeExport) -> TensorTree:
        mask = self.mask
        if mask is None and isinstance(episode, Episode):
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            ).mask.to(device=episode.device)

        embedding = self.encoder(src=episode.embeddings_packed, mask=mask)  # pyright: ignore[reportOptionalCall]

        if isinstance(episode, Episode):
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

        else:
            observation_summary = embedding[
                :,
                episode.index[Modality.SPECIAL.value][  # pyright: ignore[reportArgumentType]
                    SpecialToken.OBSERVATION_SUMMARY.value
                ][-1],
            ]

            observation_history = embedding[
                :,
                episode.index[Modality.SPECIAL.value][  # pyright: ignore[reportArgumentType]
                    SpecialToken.OBSERVATION_HISTORY.value
                ][-1],
            ]

        features = rearrange(
            [observation_summary, observation_history.detach()], "i b 1 d -> b 1 (i d)"
        )

        return self.heads(features)

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        logits = self._compute_logits(episode)
        targets = tree_map(
            lambda k: episode.get(k)[:, -1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b 1 d -> b d"), logits),
            tree_map(Rearrange("b 1 -> b"), targets),
        )

        return {"loss": losses}

    @override
    def predict(  # noqa: C901, PLR0915
        self,
        episode: Episode,
        *,
        result_keys: AbstractSet[PredictionResultKey],
        **kwargs: Any,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        result = {}

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths()).squeeze(
                -1
            )

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_STD,
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

            features = rearrange(
                [observation_summary, observation_history], "i b 1 d -> b 1 (i d)"
            )

            logits = TensorDict(self.heads(features), batch_size=[b, 1])

            timestep_padder = nan_padder(pad=(t - 1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0]
                        case (Modality.DISCRETE, "turn_signal"):
                            return non_zero_signal_with_threshold(x).class_idx
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportAttributeAccessIssue]
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
                            return torch.zeros_like(x[..., 0])  # placeholder
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportAttributeAccessIssue]
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
                            return non_zero_signal_with_threshold(x).prob

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportAttributeAccessIssue]
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

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, -1]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return F.l1_loss(x[..., 0], gt, reduction="none")

                        case (Modality.DISCRETE, "turn_signal"):
                            return F.l1_loss(
                                non_zero_signal_with_threshold(x).class_idx.float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                result[result_key] = logits.named_apply(fn, nested_keys=True).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

        return TensorDict(result).auto_batch_size_(2)

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        mask = cast(
            "AttentionMask",
            ForwardDynamicsPredictionObjective.build_attention_mask(
                index, timestep, legend=legend
            ).clone(recurse=True),
        )

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]
            current_observations = current.select(
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
            past_actions = past.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
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
