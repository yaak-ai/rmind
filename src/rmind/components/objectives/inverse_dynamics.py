from collections.abc import Callable
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
    SummaryToken,
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

        self.encoder: Module | None = encoder
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

        self._build_attention_mask: Callable[..., AttentionMask] = lru_cache(
            maxsize=2, typed=True
        )(self.build_attention_mask)

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # ty:ignore[call-non-callable]

        observation_summaries = (
            episode.index
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
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

        losses = self.losses(
            tree_map(Rearrange("b t 1 d -> (b t) d"), logits),
            tree_map(Rearrange("b t 1 -> (b t)"), targets),
        )  # ty:ignore[call-non-callable]

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
                value=episode.input.select(*self.heads.tree_paths()),
                timestep_indices=slice(None),
            )

        if keys & {
            PredictionKey.PREDICTION_VALUE,
            PredictionKey.PREDICTION_PROBS,
            PredictionKey.SCORE_LOGPROB,
            PredictionKey.SCORE_L1,
            PredictionKey.SUMMARY_EMBEDDINGS,
            PredictionKey.ATTENTION_ROLLOUT,
        }:
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            ).to(episode.device)

            embeddings_packed = episode.embeddings_packed
            embedding = self.encoder(src=embeddings_packed, mask=mask.mask)  # ty:ignore[call-non-callable]

            observation_summaries = (
                episode.index
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            # order: (o0, o1), (o1, o2), (o2, o3), ...
            features = rearrange(
                [observation_summaries[:, :-1], observation_summaries[:, 1:]],
                "i b t 1 d -> b t 1 (i d)",
            )

            logits = TensorDict(self.heads(features), batch_size=[b, t - 1])

            # all but last
            timestep_indices = slice(t - 1)

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # ty:ignore[possibly-missing-attribute]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, possibly-missing-attribute]
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.softmax(dim=-1)),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_LOGPROB) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))  # ty:ignore[possibly-missing-attribute]
                        .apply(  # ty:ignore[possibly-missing-attribute]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                        )
                        .apply(lambda x: -torch.log(x))  # ty:ignore[possibly-missing-attribute]
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[possibly-missing-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, possibly-missing-attribute]
                            nested_keys=True,
                        )
                        .apply(  # ty:ignore[possibly-missing-attribute]
                            lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                            episode.input[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                            nested_keys=True,
                        )
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

            if (key := PredictionKey.ATTENTION_ROLLOUT) in keys:
                attention_rollout = self.encoder.compute_attention_rollout(  # ty:ignore[possibly-missing-attribute, call-non-callable]
                    src=embeddings_packed,
                    mask=mask,
                    head_fusion="max",
                    discard_ratio=0.9,
                )

                observation_keys = episode.timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                )

                attention = (
                    episode.index
                    .parse(attention_rollout, dim=1)
                    .select((Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))[:, -1]
                    .apply(
                        lambda x: (
                            episode.index
                            .parse(x, dim=2)
                            .select(*observation_keys)
                            .squeeze(dim=1)
                        )
                    )
                    .named_apply(
                        partial(_expand_attn, input=episode.input), nested_keys=True
                    )
                    .update({"input": episode.input.select(Modality.IMAGE)})
                )

                predictions[key] = Prediction(
                    value={
                        f"{k}": v
                        for (k, v) in enumerate(
                            attention.auto_batch_size_(2).split(1, dim=1)
                        )
                    },
                    timestep_indices=slice(t - 1, None),
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        return ForwardDynamicsPredictionObjective.build_attention_mask(
            index, timestep, legend=legend
        ).clone(recurse=True)


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
