from collections.abc import Set as AbstractSet
from functools import lru_cache, partial
from math import sqrt
from typing import final, override

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
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


@final
class InverseDynamicsPredictionObjective(Objective):
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

        observation_summaries = (
            episode.index.select(
                k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
            )
            .parse(embedding)
            .get(k)
        )

        # order: (o0, o1), (o1, o2), (o2, o3), ...
        features = rearrange(
            [observation_summaries[:, :-1], observation_summaries[:, 1:]],
            "i ... d -> ... (i d)",
        )

        logits = self.heads(features)

        targets = tree_map(
            lambda k: episode.get(k)[:, :-1],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b t 1 d -> (b t) d"), logits),
            tree_map(Rearrange("b t 1 -> (b t)"), targets),
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

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths())

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
            PredictionResultKey.ATTENTION_ROLLOUT,
        }:
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            ).to(episode.device)

            embeddings_packed = episode.embeddings_packed
            embedding = self.encoder(src=embeddings_packed, mask=mask.mask)  # pyright: ignore[reportOptionalCall]

            observation_summaries = (
                episode.index.select(
                    k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY)
                )
                .parse(embedding)
                .get(k)
            )

            # order: (o0, o1), (o1, o2), (o2, o3), ...
            features = rearrange(
                [observation_summaries[:, :-1], observation_summaries[:, 1:]],
                "i b t 1 d -> b t 1 (i d)",
            )
            timestep_mask = torch.tensor(
                [True] * (t - 1) + [False], dtype=torch.bool
            ).expand(b, t)

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = {
                    "value": logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    ),
                    "mask": torch.tensor([1] * (t - 1) + [0], dtype=torch.bool).expand(
                        b, t
                    ),
                }

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = {
                    "value": logits.apply(lambda x: x.softmax(dim=-1)),
                    "mask": torch.tensor([1] * (t - 1) + [0], dtype=torch.bool).expand(
                        b, t
                    ),
                }

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                result[result_key] = {
                    "value": (
                        logits.apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportOptionalMemberAccess]
                        .apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, :-1],
                        )
                        .apply(lambda x: -torch.log(x))  # pyright: ignore[reportOptionalMemberAccess]
                    ),
                    "mask": torch.tensor([1] * (t - 1) + [0], dtype=torch.bool).expand(
                        b, t
                    ),
                }

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = {
                    "value": (
                        logits.apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                            nested_keys=True,
                        )
                        .apply(  # pyright: ignore[reportOptionalMemberAccess]
                            lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                            episode.input[:, :-1],
                            nested_keys=True,
                        )
                    ),
                    "mask": torch.tensor([1] * (t - 1) + [0], dtype=torch.bool).expand(
                        b, t
                    ),
                }

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(  # pyright: ignore[reportAttributeAccessIssue]
                    embedding
                )

            if (result_key := PredictionResultKey.ATTENTION_ROLLOUT) in result_keys:
                attention_rollout = self.encoder.compute_attention_rollout(  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                    src=embeddings_packed,
                    mask=mask,
                    head_fusion="max",
                    discard_ratio=0.9,
                )

                observation_keys = episode.timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                )

                attention = (
                    episode.index.parse(attention_rollout, dim=1)
                    .select((Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))[:, -1]
                    .apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda x: episode.index.parse(x, dim=2)
                        .select(*observation_keys)
                        .squeeze(dim=1)
                    )
                    .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                        partial(_expand_attn, input=episode.input), nested_keys=True
                    )
                    .update({"input": episode.input.select(Modality.IMAGE)})
                )

                result[result_key] = {
                    "value": {
                        f"{k}": v
                        for (k, v) in enumerate(
                            attention.auto_batch_size_(2).split(1, dim=1)
                        )
                    },
                    "mask": torch.tensor([0] * (t - 1) + [1], dtype=torch.bool).expand(
                        b, t
                    ),
                }

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


def _expand_attn(path: tuple[str, ...], attn: Tensor, *, input: TensorDict) -> Tensor:
    match path:
        case (*_, Modality.IMAGE, token_to):
            (_b, _t, hw_attn) = attn.shape
            (_b, _t, _c, h_img, w_img) = input.get_item_shape((
                Modality.IMAGE,
                token_to,
            ))
            attn = rearrange(
                attn,
                "... (h_attn w_attn) -> ... h_attn w_attn",
                h_attn=int(sqrt(hw_attn * h_img / w_img)),
            )

            return F.interpolate(attn, size=(h_img, w_img))

        case _:
            return rearrange(attn, "b t d -> b t 1 d")
