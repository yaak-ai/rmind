from collections.abc import Sequence
from functools import lru_cache

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from typing_extensions import override

from cargpt.components.episode import (
    Episode,
    EpisodeBuilder,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.components.objectives.common import PredictionResultKey
from cargpt.components.objectives.forward_dynamics import (
    ForwardDynamicsPredictionObjective,
)
from cargpt.utils.containers import ModuleDict
from cargpt.utils.padder import nan_padder


class CopycatObjective(Module):
    """Inspired by: Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction (https://arxiv.org/abs/2207.09705)"""

    def __init__(
        self,
        *,
        memory_extraction: Module | None = None,
        policy: Module | None = None,
    ):
        super().__init__()

        self.streams = ModuleDict(**{
            name: stream
            for name, stream in (
                ("memory_extraction", memory_extraction),
                ("policy", policy),
            )
            if stream is not None
        })

    @override
    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        loss = TensorDict.from_dict({
            name: stream.forward(episode, embedding)
            for name, stream in self.streams.items()
        })

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: Sequence[PredictionResultKey] = tuple(PredictionResultKey),  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> TensorDict:
        episode = episode_builder.build_episode(inputs)
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)

        return TensorDict.from_dict({
            name: stream.predict(episode, embedding, result_keys=result_keys)
            for name, stream in self.streams.items()
        })

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = ForwardDynamicsPredictionObjective._build_attention_mask(
            index, timestep, legend
        ).clone()  # pyright: ignore

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        for step in range(t):
            past, current = index[:step], index[step]  # pyright: ignore[reportIndexIssue]
            current_observations = current.select(*timestep.keys(TokenType.OBSERVATION))
            current_observation_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))
            current_observation_history = current.select((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))
            past_actions = past.select(*timestep.keys(TokenType.ACTION))
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_not_attend(
                    current_observations,
                    past_actions,
                )
                ._do_not_attend(
                    current_observations,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_observation_summary,
                    past_actions,
                )
                ._do_not_attend(
                    current_observation_summary,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_observation_history,
                    past_actions,
                )
                ._do_not_attend(
                    current_observation_history,
                    past_action_summary,
                )
            )

        return mask


class MemoryExtractionStream(Module):
    def __init__(
        self,
        *,
        delta_tokenizers: ModuleDict,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        delta_detokenizers: ModuleDict | None = None,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.delta_tokenizers = delta_tokenizers
        self.delta_detokenizers = delta_detokenizers

    @override
    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
        b, t = episode.embedded.batch_size

        features = (
            episode.index[1:]  # pyright: ignore[reportIndexIssue]
            .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[b, t - 1],
        )

        deltas = episode.inputs.select(*logits.keys(True, True)).apply(  # pyright: ignore[reportArgumentType]
            lambda tensor: torch.diff(tensor, n=1, dim=-1),
            batch_size=[b, t - 1],
        )

        labels = deltas.named_apply(
            lambda k, v: self.delta_tokenizers.get(k)(v),
            nested_keys=True,
        )

        logits = logits.apply(Rearrange("b t 1 d -> (b t) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t -> (b t)"), batch_size=[])

        loss = logits.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore[reportOptionalMemberAccess]
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
        *,
        result_keys: Sequence[PredictionResultKey],
    ) -> TensorDict:
        b, t = episode.embedded.batch_size
        result = TensorDict({}, batch_size=[b, t])

        if not result_keys:
            return result

        if any(
            result_key in result_keys
            for result_key in (
                PredictionResultKey.PREDICTION,
                PredictionResultKey.PREDICTION_PROBS,
                PredictionResultKey.SCORE_LOGPROB,
                PredictionResultKey.SCORE_L1,
            )
        ):
            if self.delta_detokenizers is None:
                msg = "delta_detokenizers missing"
                raise RuntimeError(msg)

            features = (
                episode.index[1:]  # pyright: ignore[reportIndexIssue]
                .select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
                .parse(embedding)
                .get(k)
            )

            logits = TensorDict.from_dict(
                {
                    (modality, name): head(features)
                    for (modality, name), head in self.heads.flatten()
                },
                batch_size=[b, t - 1],
            )

            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
                prediction = prediction_tokens.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                    lambda k, v: self.delta_detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                    nested_keys=True,
                ).apply(Rearrange("b t 1 -> b t"))

                # insert NaN at index 0 to indicate no prediction for t=0 b/c deltas
                prediction = prediction.apply(nan_padder((1, 0)), batch_size=[b, t])

                result[result_key] = prediction

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                # TODO: categorical heads only
                prediction_probs = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    Rearrange("b t 1 bin -> b t bin")
                )

                prediction_probs = prediction_probs.apply(
                    nan_padder((0, 0, 1, 0)), batch_size=[b, t]
                )  # pyright: ignore[reportAttributeAccessIssue]

                result[result_key] = prediction_probs

        # TODO: Scores

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            inputs = episode.inputs.select(*(k for k, _ in self.heads.flatten()))
            ground_truth = inputs.apply(
                lambda tensor: torch.diff(tensor, n=1, dim=-1),
                batch_size=[b, t - 1],
            )
            # insert NaN at index 0 to indicate no ground_truth for t=0 b/c deltas
            ground_truth = ground_truth.apply(nan_padder((1, 0)), batch_size=[b, t])

            result[result_key] = ground_truth

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            # TODO
            raise NotImplementedError

        return result


class PolicyStream(Module):
    def __init__(
        self,
        heads: ModuleDict,
        losses: ModuleDict | None = None,
        detokenizers: ModuleDict | None = None,
    ):
        super().__init__()

        self.heads = heads
        self.losses = losses
        self.detokenizers = detokenizers

    @override
    def forward(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
    ) -> TensorDict:
        embeddings = (
            episode.index[-1]  # pyright: ignore[reportIndexIssue]
            .select(
                (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        )).detach()  # NOTE: equivalent to stop gradient layer in paper

        observation_summary = embeddings.get((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))

        features = rearrange(
            [observation_summary, observation_history],
            "i b 1 d -> b 1 (i d)",
        )

        logits = TensorDict.from_dict(
            {
                (modality, name): head(features)
                for (modality, name), head in self.heads.flatten()
            },
            batch_size=[],
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[:, -1]  # pyright: ignore[reportArgumentType]

        logits = logits.apply(Rearrange("b 1 d -> b d"), batch_size=[])
        labels = labels.apply(Rearrange("b 1 -> b"), batch_size=[])

        loss = logits.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore[reportOptionalMemberAccess]
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        episode: Episode,
        embedding: Float[Tensor, "b s d"],
        *,
        result_keys: Sequence[PredictionResultKey],
    ) -> TensorDict:
        b, t = episode.embedded.batch_size
        result = TensorDict({}, batch_size=[b, t])

        if not result_keys:
            return result
        if any(
            result_key in result_keys
            for result_key in (
                PredictionResultKey.PREDICTION,
                PredictionResultKey.PREDICTION_PROBS,
                PredictionResultKey.SCORE_LOGPROB,
                PredictionResultKey.SCORE_L1,
            )
        ):
            if self.detokenizers is None:
                msg = "detokenizers missing"
                raise RuntimeError(msg)

            embeddings = (
                episode.index[-1]  # pyright: ignore[reportIndexIssue]
                .select(
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
                )
                .parse(embedding)
            )

            observation_history = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            )).detach()  # NOTE: equivalent to stop gradient layer in paper

            observation_summary = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            features = rearrange(
                [observation_summary, observation_history],
                "i b 1 d -> b 1 (i d)",
            )

            logits = TensorDict.from_dict(
                {
                    (modality, name): head(features)
                    for (modality, name), head in self.heads.flatten()
                },
                batch_size=[],
            )

            # NOTE: pad w/ NaNs to indicate the prediction is for the last timestep only
            if (result_key := PredictionResultKey.PREDICTION) in result_keys:
                prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
                prediction = prediction_tokens.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                    lambda k, v: self.detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                    nested_keys=True,
                )

                prediction = prediction.apply(nan_padder((t - 1, 0)), batch_size=[b, t])

                result[result_key] = prediction

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(
                    nan_padder((0, 0, t - 1, 0)), batch_size=[b, t]
                )
            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                prediction_probs = logits.apply(lambda x: x.softmax(dim=-1)).apply(
                    nan_padder((0, 0, t - 1, 0)), batch_size=[b, t]
                )  # pyright: ignore[reportAttributeAccessIssue]

                gt_tokens = episode.tokenized.select(
                    *(k for k, _ in self.heads.flatten())
                )
                probs_of_gt = prediction_probs.apply(
                    lambda _probs, _tokens: _probs.gather(index=_tokens, dim=-1),
                    gt_tokens,
                )
                result[result_key] = probs_of_gt.apply(lambda x: -torch.log(x))

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                # TODO: get rid of code duplication
                prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
                prediction = prediction_tokens.named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                    lambda k, v: self.detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                    nested_keys=True,
                )

                prediction = prediction.apply(nan_padder((t - 1, 0)), batch_size=[b, t])
                ground_truth = episode.inputs.select(
                    *(k for k, _ in self.heads.flatten())
                )
                l1 = prediction.named_apply(
                    lambda k, v: F.l1_loss(v, ground_truth[k], reduction="none"),
                    nested_keys=True,
                )
                result[result_key] = l1

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            ground_truth = episode.inputs.select(*(k for k, _ in self.heads.flatten()))

            result[result_key] = ground_truth

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            raise NotImplementedError

        return result
