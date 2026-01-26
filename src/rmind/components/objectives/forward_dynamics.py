from collections.abc import Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import TYPE_CHECKING, final, override

import torch
from einops import pack, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf
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

if TYPE_CHECKING:
    from rmind.components.position_encoding import PatchPositionEmbedding2D


@final
class ForwardDynamicsPredictionObjective(Objective):
    def __init__(  # noqa: PLR0913
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        projections: ModuleDict | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.encoder: Module | None = encoder
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections
        self.patch_pos_embed: PatchPositionEmbedding2D | None = patch_pos_embed

        self._build_attention_mask: Callable[..., AttentionMask] = lru_cache(
            maxsize=2, typed=True
        )(self.build_attention_mask)
        self._last_embeddings: torch.Tensor | None = None
        self._last_targets: torch.Tensor | None = None

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # ty:ignore[call-non-callable]

        index = episode.index[:-1]  # all but last timestep

        foresight_cameras = (
            tuple(self.heads[Modality.FORESIGHT].keys())  # ty:ignore[call-non-callable]
            if Modality.FORESIGHT in self.heads
            else ()
        )

        observation_keys = [
            *((Modality.FORESIGHT, cam) for cam in foresight_cameras),
            *(
                k
                for m in self.heads
                if m != Modality.FORESIGHT
                for k in ((m, name) for name in self.heads[m])
            ),
        ]

        observations = TensorDict({
            k: index.select(k).parse(embedding).get(k) for k in observation_keys
        })
        action_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        obs_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )
        features: TensorDict = observations.apply(
            # pack: (obs[0], action_summary), (obs[1], action_summary), ...
            lambda obs: pack(
                [
                    obs,
                    obs_summary.broadcast_to(obs.shape),
                    action_summary.broadcast_to(obs.shape),
                ],
                "b t p *",
            )[0]
        )  # ty:ignore[invalid-assignment]

        features_projected = TensorDict(
            self.projections(features.to_dict())  # ty:ignore[call-non-callable]
        )

        mask_tokens = repeat(
            episode.embeddings.get((Modality.UTILITY, "mask"))[:, :-1],
            "b t 1 d -> b t n d",
            n=episode.embeddings.get((Modality.IMAGE, "cam_front_left")).shape[-2],
        )
        if self.patch_pos_embed is not None:
            mask_tokens = mask_tokens + self.patch_pos_embed()  # noqa: PLR6104

        features_for_heads = features_projected.to_dict()
        for cam in foresight_cameras:
            features_for_heads[Modality.FORESIGHT][cam] = {
                "query": mask_tokens,
                "context": features_projected.get((Modality.FORESIGHT, cam)),
            }

        logits = self.heads(
            features_for_heads,
            is_leaf_input=lambda x: (
                isinstance(x, dict) and "query" in x and "context" in x
            ),
        )

        targets = tree_map(
            lambda k: episode.get(tuple(k))[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, (list, tuple)),
        )

        losses = self.losses(
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
        )  # ty:ignore[call-non-callable]

        self._last_embeddings = logits
        self._last_targets = targets
        return {"loss": losses}

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
            # Filter out foresight paths since they're not in episode.input
            non_foresight_paths = [
                p for p in self.heads.tree_paths() if p[0] != Modality.FORESIGHT
            ]
            predictions[key] = Prediction(
                value=episode.input.select(*non_foresight_paths),
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
            foresight_cameras = (
                tuple(self.heads[Modality.FORESIGHT].keys())  # ty:ignore[call-non-callable]
                if Modality.FORESIGHT in self.heads
                else ()
            )

            observation_keys = [
                *((Modality.FORESIGHT, cam) for cam in foresight_cameras),
                *(
                    k
                    for m in self.heads
                    if m != Modality.FORESIGHT
                    for k in ((m, name) for name in self.heads[m])
                ),
            ]
            observations = TensorDict({
                k: index.select(k).parse(embedding).get(k) for k in observation_keys
            })
            action_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
                .parse(embedding)
                .get(k)
            )
            obs_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))
                .parse(embedding)
                .get(k)
            )

            features: TensorDict = observations.apply(
                lambda obs: pack(
                    [
                        obs,
                        obs_summary.broadcast_to(obs.shape),
                        action_summary.broadcast_to(obs.shape),
                    ],
                    "b t p *",
                )[0]
            )  # ty:ignore[invalid-assignment]

            features_projected = TensorDict(
                self.projections(features.to_dict())  # ty:ignore[call-non-callable]
            )

            mask_tokens = repeat(
                episode.embeddings.get((Modality.UTILITY, "mask"))[:, :-1],
                "b t 1 d -> b t n d",
                n=episode.embeddings.get((Modality.IMAGE, "cam_front_left")).shape[-2],
            )
            if self.patch_pos_embed is not None:
                mask_tokens = mask_tokens + self.patch_pos_embed()  # noqa: PLR6104

            # Build uniform input structure for heads (cross-attention heads receive dict)
            features_for_heads = features_projected.to_dict()
            for cam in foresight_cameras:
                features_for_heads[Modality.FORESIGHT][cam] = {
                    "query": mask_tokens,
                    "context": features_projected.get((Modality.FORESIGHT, cam)),
                }

            logits = TensorDict(
                self.heads(
                    features_for_heads,
                    is_leaf_input=lambda x: (
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
                        .exclude(Modality.FORESIGHT)
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

            mask = (
                mask
                .do_attend(current, current)
                .do_attend(current, past)
                .do_not_attend(current_observations, current_actions)
                .do_not_attend(current_observations, current_action_summary)
                .do_not_attend(current_foresight, current_actions)
                .do_not_attend(current_foresight, current_action_summary)
                .do_not_attend(current_observation_summary, current_actions)
                .do_not_attend(current_observation_summary, current_action_summary)
                .do_not_attend(current_observation_history, current_actions)
                .do_not_attend(current_observation_history, current_action_summary)
                .do_not_attend(current_observations, current_foresight)
                .do_not_attend(current_observations, current_observation_summary)
                .do_not_attend(current_observations, current_observation_history)
                .do_not_attend(current_foresight, current_observation_summary)
                .do_not_attend(current_foresight, current_observation_history)
            )
        return mask

    @classmethod
    def mask_observations_from_past_actions(
        cls,
        mask: AttentionMask,
        index: Index,
        timestep: Timestep,
        *,
        include_foresight: bool = True,
    ) -> AttentionMask:
        (t,) = index.batch_size
        for step in range(t):
            past, current = index[:step], index[step]
            current_observations = current.select(
                *timestep.get(TokenType.OBSERVATION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_observation_summary = current.select((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_HISTORY,
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
                .do_not_attend(current_observations, past_actions)
                .do_not_attend(current_observations, past_action_summary)
                .do_not_attend(current_observation_summary, past_actions)
                .do_not_attend(current_observation_summary, past_action_summary)
                .do_not_attend(current_observation_history, past_actions)
                .do_not_attend(current_observation_history, past_action_summary)
            )

            if include_foresight:
                current_foresight = current.select(Modality.FORESIGHT)
                mask = mask.do_not_attend(
                    current_foresight, past_actions
                ).do_not_attend(current_foresight, past_action_summary)

        return mask
