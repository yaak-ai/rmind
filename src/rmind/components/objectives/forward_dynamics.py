from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)
from rmind.components.transformer.decoder import CrossAttentionDecoderHead


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        norm: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        projections: InstanceOf[ModuleDict] | None = None,
        patch_pos_embed: InstanceOf[Module] | None = None,
        ar_steps: int = 0,
        feedback_projection: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections
        self.patch_pos_embed: Module | None = patch_pos_embed
        self.ar_steps: int = ar_steps
        self.feedback_projection: Module | None = feedback_projection

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        if self.norm is not None:
            embedding = self.norm(embedding)

        index = episode.index[:-1]  # all but last timestep

        observation_keys = self.heads.tree_paths()
        observations = index.select(*observation_keys).parse(embedding)
        action_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )
        features: TensorDict = observations.apply(
            lambda obs: pack([obs, action_summary.broadcast_to(obs.shape)], "b t p *")[
                0
            ]
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

        if (
            self.ar_steps > 0
            and self.feedback_projection is not None
            and self.projections is not None
            and self.losses is not None
        ):
            losses = self._ar_losses(
                logits=logits,
                losses=losses,
                action_summary=action_summary,
                mask_tokens=mask_tokens,
                episode=episode,
            )

        return {
            "loss": losses,
            "_artifacts": {"last_embeddings": logits, "last_targets": targets},
        }

    def _ar_losses(
        self,
        *,
        logits: dict,
        losses: dict,
        action_summary: Tensor,
        mask_tokens: Tensor,
        episode: Episode,
    ) -> dict:
        if (
            self.feedback_projection is None
            or self.targets is None
            or self.losses is None
        ):
            return losses

        foresight_key = ("foresight", "cam_front_left")

        # z_hat_tf: b (T-1) n_patches image_embedding_dim
        z_hat_tf = logits[Modality.FORESIGHT]["cam_front_left"]
        t_minus_1 = z_hat_tf.shape[1]

        if t_minus_1 < 2:  # noqa: PLR2004
            return losses

        # Project each patch independently (preserves spatial structure):
        # b (T-1) n_patches image_embedding_dim → b (T-1) n_patches encoder_embedding_dim
        z_hat_proj = self.feedback_projection(z_hat_tf)

        # AR context for predicting z[t+2]:
        # use all predicted patches from z_hat[t] + action_summary[t+1] as one extra token
        z_hat_ctx_ar = z_hat_proj[:, :-1]  # b (T-2) n_patches encoder_embedding_dim
        action_summary_ar = action_summary[:, 1:]  # b (T-2) 1 encoder_embedding_dim
        # b (T-2) (n_patches+1) encoder_embedding_dim — already in decoder dim, skip projections
        ar_context = torch.cat([z_hat_ctx_ar, action_summary_ar], dim=-2)

        ar_logits = self.heads.get(foresight_key)(
            CrossAttentionDecoderHead.Input(
                query=mask_tokens[:, :-1],  # b (T-2) n_patches encoder_embedding_dim
                context=ar_context,
            )
        )  # b (T-2) n_patches image_embedding_dim

        # targets for AR: true embeddings at z[2:]
        foresight_target_key = self.targets[Modality.FORESIGHT]["cam_front_left"]
        ar_targets = episode.get(foresight_target_key)[:, 2:]

        ar_loss = self.losses.get(foresight_key)(
            rearrange(ar_logits, "b t s d -> (b t s) d"),
            rearrange(ar_targets, "b t s d -> (b t s) d"),
        )

        # split leaf loss into tf / ar for separate wandb logging
        tf_loss = losses[Modality.FORESIGHT]["cam_front_left"]
        losses[Modality.FORESIGHT]["cam_front_left"] = {"tf": tf_loss, "ar": ar_loss}

        return losses

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        b, t = episode.input.batch_size

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths(), strict=False),
                timestep_indices=slice(None),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
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

            features: TensorDict = observations.apply(
                lambda obs: pack(
                    [obs, action_summary.broadcast_to(obs.shape)], "b t p *"
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

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
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

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.exclude(Modality.FORESIGHT).apply(
                        lambda x: x.softmax(dim=-1)
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_LOGPROB) in keys:
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

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:
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

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
