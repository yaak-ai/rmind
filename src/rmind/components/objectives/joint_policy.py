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
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        losses: InstanceOf[ModuleDict],
        chunk: Path,
        norm: InstanceOf[Module] | None = None,
        sample_codes: bool = True,
        teacher_force_offset: bool = True,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.code_head = code_head  # features -> (G*C) code logits
        self.offset_head = (
            offset_head  # features -> (G*C*action_dim): offset per (quantizer, code)
        )
        self.losses = losses  # {"code": ..., "offset": ...}
        self.chunk: Path = chunk
        self.sample_codes = sample_codes
        self.teacher_force_offset = teacher_force_offset

    @override
    def train(self, mode: bool = True) -> "JointPolicyObjective":
        super().train(mode)
        self.tokenizer.eval()
        return self

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        features = self._features(episode, embedding)
        _, codes, offset = self._predict(features)

        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({"joint_actions": chunk})

    def _features(self, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)

        embeddings = (
            episode
            .index[-1]
            .select(
                (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
                (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
                (Modality.CONTEXT, "waypoints"),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_HISTORY,
        ))
        observation_summary = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_SUMMARY,
        ))
        waypoints = embeddings.get((Modality.CONTEXT, "waypoints")).mean(
            dim=1, keepdim=True
        )

        return rearrange(
            [observation_summary, observation_history, waypoints], "i b 1 d -> b (i d)"
        )

    def _heads(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Code logits (b, g, c) and the full offset table (b, g, c, action_dim)."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        code_logits = rearrange(self.code_head(features), "b (g c) -> b g c", g=g, c=c)
        offsets = rearrange(
            self.offset_head(features), "b (g c a) -> b g c a", g=g, c=c
        )
        return code_logits, offsets

    @staticmethod
    def _gather_offset(offsets: Tensor, codes: Tensor) -> Tensor:
        """Select each quantizer's offset at `codes` and sum over quantizers."""
        index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
        # https://arxiv.org/pdf/2403.03181 Figure 2.
        return offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

    def _sample_codes(self, code_logits: Tensor) -> Tensor:
        if self.sample_codes:
            _, g, c = code_logits.shape
            return rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            )
        return code_logits.argmax(dim=-1)

    def _predict(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """VQ-BeT joint code prediction with a code-conditioned offset."""
        code_logits, offsets = self._heads(features)
        codes = self._sample_codes(code_logits)
        offset = self._gather_offset(offsets, codes)

        return code_logits, codes, offset

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode, embedding)  # (b, feature_dim)
        tokenizer = self.tokenizer

        with torch.no_grad():
            chunk = episode.get(self.chunk)[:, -1]  # (b, action_clip, action_space)
            target_codes = tokenizer(chunk)  # (b, num_quantizers) ground-truth codes
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, action_dim): the GT action chunk the policy must reconstruct

        code_logits, offsets = self._heads(features)

        losses: dict[str, Tensor] = {}

        # per-quantizer classification against the ground-truth codes
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                code_logits[:, q, :], target_codes[:, q]
            )

        # reconstruction as inference does it (decode sampled/argmax codes + offset
        # gathered at those codes); the frozen tokenizer makes invert() gradient-free
        codes = self._sample_codes(code_logits)
        sampled_chunk = tokenizer.invert(codes) + self._gather_offset(offsets, codes)
        sampled_recon = self.losses["offset"](sampled_chunk.detach(), target)

        if self.teacher_force_offset:
            # offset supervised at the GROUND-TRUTH codes (teacher forcing), so each
            # code's offset entry only sees residuals of actions that quantized to it;
            # supervising at sampled codes lets the offset learn the conditional median
            # regardless of code, cancelling the code selection
            predicted_chunk = tokenizer.invert(target_codes) + self._gather_offset(
                offsets, target_codes
            )
        else:
            predicted_chunk = sampled_chunk

        losses["offset"] = self.losses["offset"](predicted_chunk, target)

        # sampled-code reconstruction error (the pre-fix loss value), logged for
        # comparability across the teacher-forcing A/B; carries no gradient
        return {"loss": losses, "metric": {"offset_sampled_recon": sampled_recon}}

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
            _, codes, offset = self._predict(features)

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
