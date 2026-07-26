from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)

type Path = tuple[str, ...]


@final
class JointPolicyObjective(Objective):
    """VQ-BeT action-chunk policy (https://arxiv.org/pdf/2403.03181).

    From the last timestep's summary features, a joint head predicts the frozen
    action tokenizer's residual-VQ codes (one categorical per quantizer) and a
    code-conditioned continuous offset; the action chunk is `decode(codes) + offset`.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: InstanceOf[Module],
        decoder: InstanceOf[Module],
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        losses: InstanceOf[ModuleDict],
        chunk: Path,
        norm: InstanceOf[Module] | None = None,
        sample_codes: bool = True,
        offset_decoder: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.decoder = decoder  # mask-query cross-attention pooler over latent sets
        self.code_head = code_head  # features -> (G*C) code logits
        self.offset_head = (
            offset_head  # features -> (G*C*action_dim): offset per (quantizer, code)
        )
        # Optional dedicated cross-attention pooler for the offset head. The
        # shared `decoder` feature is dominated by the (much larger) code losses,
        # so a single offset head reading it regresses sub-cell residuals toward
        # their per-code median (elevated offset loss). A separate `offset_decoder`
        # gives the offset its own pooling of the same summary tokens, decoupled
        # from the code-optimized feature. None = original shared-feature behavior.
        self.offset_decoder: Module | None = offset_decoder
        self.losses = losses  # {"code": ..., "offset": ...}
        self.chunk: Path = chunk
        self.sample_codes = sample_codes

    @override
    def train(self, mode: bool = True) -> "JointPolicyObjective":
        super().train(mode)
        self.tokenizer.eval()
        return self

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        features = self._features(episode, embedding)
        offset_features = self._offset_features(episode, embedding)
        _, codes, offset = self._predict(features, offset_features)

        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({"joint_actions": chunk})

    def _context(self, episode: Episode, embedding: Tensor) -> tuple[Tensor, Tensor]:
        """Cross-attention (query, context) shared by the code + offset poolers."""
        if self.norm is not None:
            embedding = self.norm(embedding)

        last = episode.index[-1]
        k_os = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        k_oh = (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY)
        observation_summary = last.select(k_os).parse(embedding).get(k_os)  # (b, 64, d)
        observation_history = last.select(k_oh).parse(embedding).get(k_oh)  # (b, 32, d)

        context = torch.cat(
            [observation_summary, observation_history], dim=-2
        )  # (b, 96, d)
        query = episode.embeddings.get((Modality.UTILITY, "mask"))[
            :, -1, [3]
        ]  # (b, 1, d)
        return query, context

    def _features(self, episode: Episode, embedding: Tensor) -> Tensor:
        query, context = self._context(episode, embedding)
        return self.decoder({"query": query, "context": context}).squeeze(-2)  # (b, d)

    def _offset_features(self, episode: Episode, embedding: Tensor) -> Tensor:
        """Feature the offset head reads: its own pooler if present, else shared."""
        if self.offset_decoder is None:
            return self._features(episode, embedding)
        query, context = self._context(episode, embedding)
        return self.offset_decoder(
            {"query": query, "context": context}
        ).squeeze(-2)  # (b, d)

    def _code_logits(self, features: Tensor) -> Tensor:
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        return rearrange(self.code_head(features), "b (g c) -> b g c", g=g, c=c)

    def _offsets(self, offset_features: Tensor) -> Tensor:
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        return rearrange(
            self.offset_head(offset_features), "b (g c a) -> b g c a", g=g, c=c
        )

    @staticmethod
    def _gather_offset(offsets: Tensor, codes: Tensor) -> Tensor:
        """Select each quantizer's offset at `codes` and sum over quantizers."""
        index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
        # https://arxiv.org/pdf/2403.03181 Figure 2.
        return offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

    def _predict(
        self, features: Tensor, offset_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """VQ-BeT joint code prediction with a code-conditioned offset."""
        code_logits = self._code_logits(features)
        if self.sample_codes:
            _, g, c = code_logits.shape
            codes = rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            )
        else:
            codes = code_logits.argmax(dim=-1)

        offset = self._gather_offset(self._offsets(offset_features), codes)
        return code_logits, codes, offset

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode, embedding)  # (b, feature_dim)
        offset_features = self._offset_features(episode, embedding)  # (b, feature_dim)
        tokenizer = self.tokenizer

        with torch.no_grad():
            chunk = episode.get(self.chunk)[:, -1]  # (b, action_clip, action_space)
            target_codes = tokenizer(chunk)  # (b, num_quantizers) ground-truth codes
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, action_dim): the GT action chunk the policy must reconstruct

        code_logits = self._code_logits(features)

        losses: dict[str, Tensor] = {}

        # per-quantizer classification against the ground-truth codes
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                code_logits[:, q, :], target_codes[:, q]
            )

        # teacher-forced offset: gather at the GROUND-TRUTH codes so the offset
        # learns each cell's true residual (never the sampled-code median). The
        # frozen tokenizer makes invert(codes) gradient-free, so this term trains
        # only the offset head (+ offset_decoder if present).
        predicted_chunk = tokenizer.invert(target_codes) + self._gather_offset(
            self._offsets(offset_features), target_codes
        )
        losses["offset"] = self.losses["offset"](predicted_chunk, target)

        return {"loss": losses}

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        tokenizer = self.tokenizer
        timestep_index = slice(-1, None)

        action_space = tokenizer._action_features  # noqa: SLF001
        b, t = episode.input.batch_size
        time_index = torch.arange(t, device=embedding.device).expand(b, -1)[
            :, timestep_index
        ]

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            chunk = tokenizer._normalize(  # noqa: SLF001
                episode.get(self.chunk)[:, -1].flatten(-2, -1)
            ).unflatten(-1, (-1, action_space))  # (b, action_horizon, action_space)
            actions = TensorDict({
                "continuous": TensorDict({
                    "gas_pedal": chunk[..., 0],
                    "brake_pedal": chunk[..., 1],
                    "steering_angle": chunk[..., 2],
                }),
                "discrete": TensorDict({"turn_signal": chunk[..., 3].long()}),
            })
            predictions[key] = Prediction(value=actions, time_index=time_index)

        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            features = self._features(episode, embedding)
            offset_features = self._offset_features(episode, embedding)
            _, codes, offset = self._predict(features, offset_features)

            chunk = (tokenizer.invert(codes) + offset).unflatten(
                -1, (-1, action_space)
            )  # (b, action_horizon, action_space)
            actions = TensorDict({
                "continuous": TensorDict({
                    "gas_pedal": chunk[..., 0],
                    "brake_pedal": chunk[..., 1],
                    "steering_angle": chunk[..., 2],
                }),
                "discrete": TensorDict({
                    "turn_signal": torch.bucketize(
                        chunk[..., 3] * 2, torch.tensor([0.5, 1.5], device=chunk.device)
                    )
                }),
            })
            predictions[key] = Prediction(value=actions, time_index=time_index)

        return TensorDict(predictions).auto_batch_size_(2)
