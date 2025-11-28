from collections.abc import Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import torch
from einops import pack, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from timm.layers.pos_embed_sincos import rope_rotate_half
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
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

REFERENCE_CAMERA = "cam_front_left"


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        position_embedding: InstanceOf[Module] | None = None,
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

        # TODO: should it smh be initialized via .yaml?  # noqa: FIX002
        # https://github.com/huggingface/pytorch-image-models/blob/af3732eebe8c1964e5ba5f2769f955e6e0deb980/timm/layers/pos_embed_sincos.py#L271
        # https://github.com/huggingface/pytorch-image-models/blob/af3732eebe8c1964e5ba5f2769f955e6e0deb980/timm/layers/pos_embed_sincos.py#L1138
        sin_emb, cos_emb = position_embedding.get_embed().tensor_split(2, -1)  # ty:ignore[possibly-missing-attribute, call-non-callable]
        self.register_buffer("_sin_emb", sin_emb, persistent=False)
        self.register_buffer("_cos_emb", cos_emb, persistent=False)

    def _apply_position_embeddings(self, x: Tensor) -> Tensor:
        """Apply rotary position embeddings to input tensor."""
        return rope_rotate_half(x) * self._sin_emb + x * self._cos_emb  # type: ignore[operator]

    @staticmethod
    def _compute_num_mask_tokens(episode: Episode, foresight: TensorDict) -> int:
        num_foresight_tokens = foresight[REFERENCE_CAMERA].shape[-2]
        num_image_tokens = episode.projected_embeddings.get((
            Modality.IMAGE,
            REFERENCE_CAMERA,
        )).shape[-2]
        return num_image_tokens - num_foresight_tokens

    def _extract_foresight_features(
        self, episode: Episode, index: Index, embedding: Tensor
    ) -> TensorDict:
        action_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        foresight = index.select(k := Modality.FORESIGHT).parse(embedding).get(k)

        num_mask_tokens = self._compute_num_mask_tokens(episode, foresight)
        mask_token = episode.projected_embeddings.get((Modality.UTILITY, "mask"))[
            :, :-1
        ]

        return (
            foresight
            .apply(
                # cat mask tokens to foresight tokens
                lambda x: torch.cat(
                    [x, repeat(mask_token, "b t 1 d -> b t n d", n=num_mask_tokens)],
                    dim=-2,
                )
            )
            .apply(self._apply_position_embeddings)
            .apply(
                # foresight, action_summary
                lambda obs: pack(
                    [obs, action_summary.broadcast_to(obs.shape)], "b t p *"
                )[0]
            )
        )

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # ty:ignore[call-non-callable]

        index = episode.index[:-1]  # all but last timestep
        features = self._extract_foresight_features(episode, index, embedding)

        logits = self.heads({Modality.IMAGE: features.to_dict()})
        logits = tree_map(Rearrange("(b t) s d -> b t s d", t=len(index)), logits)

        targets = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
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
            )  # ty:ignore[call-non-callable]

            index = episode.index[:-1]  # all but last timestep
            features = self._extract_foresight_features(episode, index, embedding)

            logits = self.heads({Modality.IMAGE: features.to_dict()})
            logits = tree_map(Rearrange("(b t) s d -> b t s d", t=len(index)), logits)
            logits = TensorDict(logits, batch_size=[b, t - 1])

            # all but first
            timestep_indices = slice(1, None)

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .exclude(Modality.IMAGE)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[possibly-missing-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, possibly-missing-attribute]
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
                        logits
                        .exclude(Modality.IMAGE)
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
                        .exclude(Modality.IMAGE)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[possibly-missing-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[possibly-missing-attribute, call-non-callable]
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

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1
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
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_SUMMARY,
            ))
            current_foresight = current.select(Modality.FORESIGHT)
            current_observation_history = current.select((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_action_summary = current.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            past_action_summary = past.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))

            mask = (
                mask
                .do_attend(current, current)
                .do_attend(current, past)
                .do_not_attend(current_observations, current_foresight)
                .do_not_attend(current_observations, current_observation_summary)
                .do_not_attend(current_observations, current_observation_history)
                .do_not_attend(current_observations, current_actions)
                .do_not_attend(current_observations, current_action_summary)
                .do_not_attend(current_observation_summary, current_actions)
                .do_not_attend(current_observation_summary, current_action_summary)
                .do_not_attend(current_observation_history, current_actions)
                .do_not_attend(current_observation_history, current_action_summary)
                .do_not_attend(current_foresight, current_observation_history)
                .do_not_attend(current_foresight, current_observation_summary)
                .do_not_attend(current_foresight, current_actions)
                .do_not_attend(current_foresight, current_action_summary)
                .do_not_attend(current_foresight, past_actions)
                .do_not_attend(current_foresight, past_action_summary)
                .do_not_attend(current_observation_summary, past_actions)
                .do_not_attend(current_observation_summary, past_action_summary)
                .do_not_attend(current_observation_history, past_actions)
                .do_not_attend(current_observation_history, past_action_summary)
                .do_not_attend(current_observation_summary, current_observations)
                .do_not_attend(current_observation_history, current_observations)
                .do_not_attend(current_observation_summary, past_observations)
                .do_not_attend(current_observation_history, past_observations)
                .do_not_attend(current_observations, past_actions)
                .do_not_attend(current_observations, past_action_summary)
            )

        return mask
