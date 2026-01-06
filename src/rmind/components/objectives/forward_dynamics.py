from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import torch
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from timm.layers.pos_embed_sincos import rope_rotate_half
from torch.nn import Module
from torch.nn import functional as F
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
from rmind.utils.functional import nan_padder


@final
class ForwardDynamicsPredictionObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        position_embedding: InstanceOf[Module] | None = None,
        projections: InstanceOf[ModuleDict] | None = None,
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
        self.position_embedding = position_embedding
        self.projections = projections

        self._build_attention_mask = lru_cache(maxsize=2, typed=True)(
            self.build_attention_mask
        )

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:  # noqa: PLR0914
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # pyright: ignore[reportOptionalCall]

        index = episode.index[:-1]  # all but last timestep

        speed = (
            index.select(k := (Modality.CONTINUOUS, "speed"))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        foresight = (
            index.select(k := (Modality.SPECIAL, SpecialToken.FORESIGHT))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        observations = TensorDict()
        observations = observations.set(
            (Modality.SPECIAL, SpecialToken.FORESIGHT), foresight
        )
        observations = observations.set((Modality.CONTINUOUS, "speed"), speed)

        action_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
            .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
            .get(k)
        )

        obs_summary = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
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
        )

        projected_features = TensorDict(
            self.projections(features.to_dict())  # pyright: ignore[reportOptionalCall, reportOptionalMemberAccess]
        )
        foresight = projected_features.get((Modality.SPECIAL, SpecialToken.FORESIGHT))

        # get foresight embeddings from the encoder
        # add mask tokens to make up for missing image tokens
        foresight_tokens = foresight.shape[-2]
        batch_size = foresight.shape[0]

        mask_embedding = episode.embeddings.get((Modality.SPECIAL, SpecialToken.MASK))[
            :, :-1
        ]
        image_tokens = episode.embeddings.get((Modality.IMAGE, "cam_front_left")).shape[
            -2
        ]

        mask_embedding = repeat(
            mask_embedding, "b t 1 d -> b t n d", n=image_tokens - foresight_tokens
        )
        foresight_embeddings = torch.cat([foresight, mask_embedding], dim=2)

        padded_foresight = TensorDict()
        padded_foresight = padded_foresight.set(
            (Modality.SPECIAL, SpecialToken.FORESIGHT), foresight_embeddings
        )

        pe_sin = TensorDict()
        pe_cos = TensorDict()

        # https://github.com/huggingface/pytorch-image-models/blob/af3732eebe8c1964e5ba5f2769f955e6e0deb980/timm/layers/pos_embed_sincos.py#L271
        rope = self.position_embedding.get_embed()  # pyright: ignore[reportCallIssue,reportOptionalMemberAccess]
        sin_emb, cos_emb = rope.tensor_split(2, -1)

        pe_sin = pe_sin.set((Modality.SPECIAL, SpecialToken.FORESIGHT), sin_emb)

        pe_cos = pe_cos.set((Modality.SPECIAL, SpecialToken.FORESIGHT), cos_emb)

        # foresight_mask = self.build_foresight_attention_mask(legend=TorchAttentionMaskLegend, timestep=foresight.shape[1], tokens=image_tokens  # noqa: ERA001)

        # https://github.com/huggingface/pytorch-image-models/blob/af3732eebe8c1964e5ba5f2769f955e6e0deb980/timm/layers/pos_embed_sincos.py#L1138
        padded_foresight = (
            padded_foresight * pe_cos
            + padded_foresight.apply(rope_rotate_half) * pe_sin
        )
        # over-write foresight with padded foresight
        projected_features = projected_features.set(
            (Modality.SPECIAL, SpecialToken.FORESIGHT),
            padded_foresight.get((Modality.SPECIAL, SpecialToken.FORESIGHT)),
        )

        speed_head = self.heads.get((Modality.CONTINUOUS, "speed"))
        foresight_head = self.heads.get((Modality.SPECIAL, SpecialToken.FORESIGHT))

        speed_input = projected_features.get((Modality.CONTINUOUS, "speed"))
        foresight_input = projected_features.get((
            Modality.SPECIAL,
            SpecialToken.FORESIGHT,
        ))

        # foresight_input = rearrange(foresight_input, "b t s d -> (b t) s d")  # noqa: ERA001

        speed_logits = speed_head(speed_input)
        foresight_logits = foresight_head(foresight_input)

        foresight_logits = rearrange(
            foresight_logits, "(b t) s d -> b t s d", b=batch_size, t=len(index)
        )

        logits = TensorDict()
        logits = logits.set((Modality.CONTINUOUS, "speed"), speed_logits)
        logits = logits.set(
            (Modality.SPECIAL, SpecialToken.FORESIGHT), foresight_logits
        )
        logits = logits.to_dict()

        # logits = self.heads(projected_features.to_dict())  # noqa: ERA001

        targets = tree_map(
            lambda k: episode.get(k)[:, 1:],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b t s d -> (b t s) d"), logits),
            tree_map(Rearrange("b t s ... -> (b t s) ..."), targets),
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
        result = {}
        b, t = episode.input.batch_size

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths()).exclude(
                Modality.IMAGE
            )

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
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

            # all but last timestep
            index = episode.index[:-1]

            observations = index.select(
                *episode.timestep.get(TokenType.OBSERVATION)
                .exclude((Modality.CONTEXT, "waypoints"))
                .keys(include_nested=True, leaves_only=True)
            ).parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]

            observation_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
                .get(k)
            )

            action_summary = (
                index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))  # pyright: ignore[reportCallIssue]
                .parse(embedding)  # pyright: ignore[reportAttributeAccessIssue]
                .get(k)
            )

            features: TensorDict = observations.apply(
                # pack: (obs[0], obs_summary, action_summary), (obs[1], obs_summary, action_summary), ...
                lambda obs: pack(
                    [
                        obs,
                        observation_summary.broadcast_to(obs.shape),
                        action_summary.broadcast_to(obs.shape),
                    ],
                    "b t p *",
                )[0]
            )

            logits = TensorDict(self.heads(features.to_dict()), batch_size=[b, t - 1])  # pyright: ignore[reportOptionalMemberAccess]

            timestep_padder = nan_padder(pad=(1, 0), dim=1)

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.argmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportOptionalMemberAccess]
                    .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.softmax(dim=-1))
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportOptionalMemberAccess]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                """Finds log prob of the correct token at each timestep."""
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.softmax(dim=-1))
                    .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportOptionalMemberAccess]
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportOptionalMemberAccess]
                    .apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                        episode.input_tokens,
                    )
                    .apply(lambda x: -torch.log(x))  # pyright: ignore[reportOptionalMemberAccess]
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.exclude(Modality.IMAGE)
                    .apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])  # pyright: ignore[reportOptionalMemberAccess]
                    .apply(  # pyright: ignore[reportOptionalMemberAccess]
                        lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                        episode.input,
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(  # pyright: ignore[reportAttributeAccessIssue]
                    embedding
                )

        return TensorDict(result).auto_batch_size_(2)

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
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
            current_observation_summary = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_foresight = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.FORESIGHT,
            ))
            current_observation_history = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            current_actions = current.select(
                *timestep.get(TokenType.ACTION).keys(
                    include_nested=True, leaves_only=True
                )
            )
            current_action_summary = current.select((  # pyright: ignore[reportCallIssue]
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_attend(current, current)  # pyright: ignore[reportArgumentType]
                .do_attend(current, past)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_foresight, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_foresight, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_summary, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, current_actions)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observation_history, current_action_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_foresight)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_observation_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_observations, current_observation_history)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_foresight, current_observation_summary)  # pyright: ignore[reportArgumentType]
                .do_not_attend(current_foresight, current_observation_history)  # pyright: ignore[reportArgumentType]
            )

        return mask

    @classmethod
    def build_foresight_attention_mask(
        cls, legend: AttentionMaskLegend, timestep: int, tokens: int
    ) -> AttentionMask:
        mask = AttentionMask(
            mask=torch.full(
                (timestep * tokens, timestep * tokens), legend.DO_NOT_ATTEND.value
            ),
            legend=legend,
            device="cpu",
        )

        for step in range(timestep):
            # current -> current
            mask.mask[
                step * tokens : (step + 1) * tokens, step * tokens : (step + 1) * tokens
            ] = False

            # current -> past
            mask.mask[step * tokens : (step + 1) * tokens, : (step + 1) * tokens] = (
                False
            )

        return mask
