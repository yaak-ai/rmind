from collections.abc import Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import torch
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf
from rerun import Tensor
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
from rmind.components.nn import Embedding
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
    def __init__(  # noqa: PLR0913
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        projections: ModuleDict | None = None,
        patch_grid_size: tuple[int, int] = (16, 16),
        patch_embed_dim: int = 384,
    ) -> None:
        super().__init__()

        self.encoder: Module | None = encoder
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.projections: ModuleDict | None = projections

        self.row_embed = Embedding(patch_grid_size[0], patch_embed_dim)
        self.col_embed = Embedding(patch_grid_size[1], patch_embed_dim)

        self._build_attention_mask: Callable[..., AttentionMask] = lru_cache(
            maxsize=2, typed=True
        )(self.build_attention_mask)
        self._last_embeddings: torch.Tensor | None = None
        self._last_targets: torch.Tensor | None = None

    def get_patch_pos_embed(self) -> torch.Tensor:
        row_pos = self.row_embed.weight  # (H, D)
        col_pos = self.col_embed.weight  # (W, D)
        pos_embed = rearrange(row_pos, "h d -> h 1 d") + rearrange(
            col_pos, "w d -> 1 w d"
        )
        return rearrange(pos_embed, "h w d -> (h w) d")

    @override
    def compute_metrics(
        self, episode: Episode, final_embedding_norm: InstanceOf[Module]
    ) -> Metrics:
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=final_embedding_norm(episode.embeddings_packed),
            mask=mask.mask.to(episode.device),
        )  # ty:ignore[call-non-callable]

        index = episode.index[:-1]  # all but last timestep
        # TODO: make it generalizable to multiple cameras  # noqa: FIX002

        observations = TensorDict({
            k: index.select(k).parse(embedding).get(k)
            for k in [
                (Modality.FORESIGHT, "cam_front_left"),
                (Modality.CONTINUOUS, "speed"),
            ]
        })
        action_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        obs_summary = (
            index
            .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
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
            self.projections(features.to_dict())  # pyright: ignore[reportOptionalCall, reportOptionalMemberAccess]  # ty:ignore[call-non-callable]
        )
        foresight = features_projected.get((Modality.FORESIGHT, "cam_front_left"))

        num_img_tokens = episode.projected_embeddings.get((
            Modality.IMAGE,
            REFERENCE_CAMERA,
        )).shape[-2]

        mask_tokens = repeat(
            episode.embeddings.get((Modality.UTILITY, "mask"))[:, :-1],
            "b t 1 d -> b t n d",
            n=num_img_tokens,
        )
        mask_tokens = mask_tokens + self.get_patch_pos_embed()  # noqa: PLR6104

        logits = {}

        if ("continuous" in self.heads) and ("speed" in self.heads["continuous"]):
            logits["continuous"] = {}
            logits["continuous"]["speed"] = self.heads["continuous"]["speed"](
                features_projected["continuous"]["speed"]
            )

        if ("foresight" in self.heads) and (
            "cam_front_left" in self.heads["foresight"]
        ):
            logits["foresight"] = {}
            logits["foresight"]["cam_front_left"] = self.heads["foresight"][
                "cam_front_left"
            ](query=mask_tokens, context=foresight)

        targets = tree_map(
            lambda k: episode.get(tuple(k))[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, list),
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

            observations = TensorDict({
                k: index.select(k).parse(embedding).get(k)
                for k in [
                    (Modality.FORESIGHT, REFERENCE_CAMERA),
                    (Modality.CONTINUOUS, "speed"),
                ]
            })
            action_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
                .get(k)
            )
            obs_summary = (
                index
                .select(k := (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
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
                self.projections(features.to_dict())  # pyright: ignore[reportOptionalCall, reportOptionalMemberAccess]  # ty:ignore[call-non-callable]
            )
            foresight = features_projected.get((Modality.FORESIGHT, REFERENCE_CAMERA))

            num_img_tokens = episode.projected_embeddings.get((
                Modality.IMAGE,
                REFERENCE_CAMERA,
            )).shape[-2]

            mask_tokens = repeat(
                episode.embeddings.get((Modality.UTILITY, "mask"))[:, :-1],
                "b t 1 d -> b t n d",
                n=num_img_tokens,
            )
            mask_tokens = mask_tokens + self.get_patch_pos_embed()  # noqa: PLR6104

            # Compute standard heads (e.g., speed) with single input
            features_for_standard_heads = {
                k: v
                for k, v in features_projected.items()
                if k != (Modality.FORESIGHT, REFERENCE_CAMERA)
            }
            logits = self.heads(features_for_standard_heads)

            # Compute foresight head with cross-attention
            # Head automatically handles 4D -> 3D flattening and unflattening
            logits_cam = self.heads[(Modality.FORESIGHT, REFERENCE_CAMERA)](
                query=mask_tokens, context=foresight
            )

            # Add foresight results to logits dict
            if Modality.FORESIGHT not in logits:
                logits[Modality.FORESIGHT] = {}
            logits[Modality.FORESIGHT][REFERENCE_CAMERA] = logits_cam

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
