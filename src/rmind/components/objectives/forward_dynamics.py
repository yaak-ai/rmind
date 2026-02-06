from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from einops import pack, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    Prediction,
    PredictionKey,
    Targets,
)
from rmind.components.tokens import Modality, SummaryToken


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        projections: InstanceOf[ModuleDict] | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections
        self.patch_pos_embed: Module | None = patch_pos_embed

    @override
    def compute_metrics(self, episode: Episode, *, embedding: Tensor) -> Metrics:
        index = episode.index[:-1]  # all but last timestep

        observation_keys = self.heads.tree_paths()
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
    def predict(
        self,
        episode: Episode,
        *,
        embedding: Tensor,
        keys: AbstractSet[PredictionKey],
        tokenizers: ModuleDict | None = None,
        **kwargs: Any,
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
            index = episode.index[:-1]  # all but last timestep
            observation_keys = self.heads.tree_paths()
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

            logits = TensorDict(
                self.heads(
                    features_projected,
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
                        .named_apply(  # ty:ignore[unresolved-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, unresolved-attribute]
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
                        .apply(Rearrange("b t 1 d -> b t d"))  # ty:ignore[unresolved-attribute]
                        .apply(  # ty:ignore[unresolved-attribute]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                        )
                        .apply(lambda x: -torch.log(x))  # ty:ignore[unresolved-attribute]
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .exclude(Modality.FORESIGHT)
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[unresolved-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, unresolved-attribute]
                            nested_keys=True,
                        )
                        .apply(  # ty:ignore[unresolved-attribute]
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
