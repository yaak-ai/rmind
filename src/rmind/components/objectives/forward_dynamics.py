from __future__ import annotations

from collections.abc import Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final

from typing_extensions import override

import torch
from einops import pack, repeat
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


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        projections: InstanceOf[ModuleDict] | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.encoder: Module | None = encoder
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections
        self.patch_pos_embed: Module | None = patch_pos_embed

        self._build_attention_mask: Callable[..., AttentionMask] = lru_cache(
            maxsize=2, typed=True
        )(self.build_attention_mask)

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:  # noqa: PLR0914
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # ty:ignore[call-non-callable]

        index = episode.index[:-1]  # all but last timestep

        foresight_keys = (
            tuple(self.heads[Modality.FORESIGHT].keys())  # ty:ignore[call-non-callable]
            if Modality.FORESIGHT in self.heads
            else ()
        )

        observation_keys = [
            *((Modality.FORESIGHT, key) for key in foresight_keys),
            *(
                k
                for m in self.heads
                if m != Modality.FORESIGHT
                for k in ((m, name) for name in self.heads[m])  # ty:ignore[not-iterable]
            ),
        ]

        observations = index.select(*observation_keys).parse(embedding)

        action_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        observation_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )
        features: TensorDict = observations.apply(
            # pack: (obs[0], observation_summary, action_summary), (obs[1], observation_summary, action_summary), ...
            lambda obs: pack(
                [
                    obs,
                    observation_summary.broadcast_to(obs.shape),
                    action_summary.broadcast_to(obs.shape),
                ],
                "b t p *",
            )[0]
        )
        features_projected = self.projections(features.to_dict())  # ty:ignore[call-non-callable]
        _, _, n_patches, _ = episode.embeddings.get((
            Modality.IMAGE,
            "cam_front_left",
        )).shape

        mask_tokens = repeat(
            episode.embeddings.get((Modality.UTILITY, "mask"))[:, 1:],
            "b t 1 d -> b t n d",
            n=n_patches,
        )
        if self.patch_pos_embed is not None:
            mask_tokens = self.patch_pos_embed(mask_tokens)

        features_projected[Modality.FORESIGHT] = tree_map(
            lambda x: {"query": mask_tokens, "context": x},
            features_projected[Modality.FORESIGHT],
        )
        logits = self.heads(
            features_projected,
            is_leaf=lambda x: isinstance(x, dict) and "query" in x and "context" in x,
        )

        targets = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
        )  # ty:ignore[call-non-callable]

        return {
            "loss": losses,
            "_artifacts": {"last_embeddings": logits, "last_targets": targets},
        }

    @override
    def predict(  # noqa: PLR0914
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
                value=episode.input.select(*self.heads.tree_paths(), strict=False),
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

            # Get all observation keys from heads (foresight cameras + other modalities)
            foresight_keys = (
                tuple(self.heads[Modality.FORESIGHT].keys())  # ty:ignore[call-non-callable]
                if Modality.FORESIGHT in self.heads
                else ()
            )

            observation_keys = [
                *((Modality.FORESIGHT, key) for key in foresight_keys),
                *(
                    k
                    for m in self.heads
                    if m != Modality.FORESIGHT
                    for k in ((m, name) for name in self.heads[m])  # ty:ignore[not-iterable]
                ),
            ]
            observations = index.select(*observation_keys).parse(embedding)

            action_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
                .parse(embedding)
                .get(k)
            )
            observation_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            features: TensorDict = observations.apply(
                lambda obs: pack(
                    [
                        obs,
                        observation_summary.broadcast_to(obs.shape),
                        action_summary.broadcast_to(obs.shape),
                    ],
                    "b t p *",
                )[0]
            )

            features_projected = TensorDict(
                self.projections(features.to_dict())  # ty:ignore[call-non-callable]
            )
            _, _, n_patches, _ = episode.embeddings.get((
                Modality.IMAGE,
                "cam_front_left",
            )).shape

            mask_tokens = repeat(
                episode.embeddings.get((Modality.UTILITY, "mask"))[:, 1:],
                "b t 1 d -> b t n d",
                n=n_patches,
            )
            if self.patch_pos_embed is not None:
                mask_tokens = self.patch_pos_embed(mask_tokens)

            # Build uniform input structure for heads (cross-attention heads receive dict)
            features_for_heads = features_projected.to_dict()
            for key in foresight_keys:
                features_for_heads[Modality.FORESIGHT][key] = {
                    "query": mask_tokens,
                    "context": features_projected.get((Modality.FORESIGHT, key)),
                }

            logits = TensorDict(
                self.heads(
                    features_for_heads,
                    is_leaf=lambda x: (
                        isinstance(x, dict) and "query" in x and "context" in x
                    ),
                ),
                batch_size=[b, t - 1],
            )

            # all but first
            timestep_indices = slice(1, None)

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .exclude(Modality.FORESIGHT)
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
                    value=logits.exclude(Modality.FORESIGHT).apply(
                        lambda x: x.softmax(dim=-1)
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_LOGPROB) in keys:
                """Finds log prob of the correct token at each timestep."""
                predictions[key] = Prediction(
                    value=(
                        logits
                        .exclude(Modality.FORESIGHT)
                        .apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))
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
                        .exclude(Modality.FORESIGHT)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[possibly-missing-attribute, call-non-callable]
                            nested_keys=True,
                        )
                        .apply(
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
    def build_attention_mask(  # noqa: PLR0914
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_NOT_ATTEND.value),
            legend=legend,
            device="cpu",
        )

        obs_keys = tuple(
            timestep.get(TokenType.OBSERVATION).keys(
                include_nested=True, leaves_only=True
            )
        )
        action_keys = tuple(
            timestep.get(TokenType.ACTION).keys(include_nested=True, leaves_only=True)
        )

        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]

            # Current timestep tokens
            cur_obs = current.select(*obs_keys)
            cur_foresight = current.select(Modality.FORESIGHT)
            cur_obs_summary = current.select((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_SUMMARY,
            ))
            cur_obs_history = current.select((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_HISTORY,
            ))
            cur_actions = current.select(*action_keys)
            cur_action_summary = current.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))

            # Past timestep tokens
            past_obs = past.select(*obs_keys)
            past_foresight = past.select(Modality.FORESIGHT)
            past_actions = past.select(*action_keys)

            mask = (
                mask
                .do_attend(cur_obs, cur_obs)
                .do_attend(cur_obs, past_obs)
                .do_attend(cur_foresight, cur_obs)
                .do_attend(cur_foresight, cur_foresight)
                .do_attend(cur_foresight, past_obs)
                .do_attend(cur_obs_summary, cur_foresight)
                .do_attend(cur_obs_summary, cur_obs_summary)
                .do_attend(cur_obs_summary, past_foresight)
                .do_attend(cur_obs_history, cur_foresight)
                .do_attend(cur_obs_history, cur_obs_history)
                .do_attend(cur_obs_history, past_foresight)
                .do_attend(cur_actions, cur_actions)
                .do_attend(cur_actions, past_actions)
                .do_attend(cur_action_summary, cur_actions)
                .do_attend(cur_action_summary, cur_action_summary)
                .do_attend(cur_action_summary, past_actions)
            )

        return mask
