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
        action_dependency_dropout: float = 0.0,
        action_dependency_dropout_keys: list[tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections
        self.patch_pos_embed: Module | None = patch_pos_embed
        self.action_dependency_dropout: float = action_dependency_dropout
        self.action_dependency_dropout_keys: list[tuple[str, ...]] | None = (
            action_dependency_dropout_keys
        )

    def _drop_action_dependency_shortcut(
        self, *, observations: TensorDict, episode: Episode
    ) -> TensorDict:
        """Force `forward_dynamics` to route through `action_summary` on a
        fraction of steps, by blanking out the current-speed observation
        embedding (the copy-forward shortcut's input) with the same
        content-free mask placeholder the foresight branch already uses as
        a query. No-op unless training and `action_dependency_dropout > 0`
        (see `pretrain_copycat_interventions.md` #5)."""
        if not self.training or not self.action_dependency_dropout:
            return observations

        keys = self.action_dependency_dropout_keys
        # all but last timestep, matching `index = episode.index[:-1]`
        mask_embedding = episode.embeddings.get((Modality.UTILITY, "mask"))[:, :-1]

        def fn(key: tuple[str, ...], obs: Tensor) -> Tensor:
            if keys is not None and key not in keys:
                return obs

            drop = (
                torch.rand(obs.shape[:-1], device=obs.device)
                < self.action_dependency_dropout
            )
            return torch.where(drop.unsqueeze(-1), mask_embedding.expand_as(obs), obs)

        return observations.named_apply(fn, nested_keys=True)  # ty:ignore[unresolved-attribute]

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        if self.norm is not None:
            embedding = self.norm(embedding)

        index = episode.index[:-1]  # all but last timestep

        observation_keys = self.heads.tree_paths()
        observations = index.select(*observation_keys).parse(embedding)
        observations = self._drop_action_dependency_shortcut(
            observations=observations, episode=episode
        )
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

        return {
            "loss": losses,
            "_artifacts": {"last_embeddings": logits, "last_targets": targets},
        }

    @override
    def predict(  # noqa: PLR0914
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
                value=episode.input.select(*self.heads.tree_paths(), strict=False)
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
            timestep_index = slice(1, None)
            time_index = torch.arange(t).expand(b, -1)[:, timestep_index]

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
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.exclude(Modality.FORESIGHT).apply(
                        lambda x: x.softmax(dim=-1)
                    ),
                    time_index=time_index,
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
                            episode.input_tokens[:, timestep_index],  # ty:ignore[invalid-argument-type]
                        )
                        .apply(lambda x: -torch.log(x))  # ty:ignore[unresolved-attribute]
                    ),
                    time_index=time_index,
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
                            episode.input[:, timestep_index],  # ty:ignore[invalid-argument-type]
                            nested_keys=True,
                        )
                    ),
                    time_index=time_index,
                )

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
