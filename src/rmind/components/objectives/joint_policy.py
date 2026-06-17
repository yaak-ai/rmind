from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module, ModuleList

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

    From the last timestep's summary features, predicts the frozen action
    tokenizer's residual-VQ codes for the action chunk (per-quantizer classifiers)
    plus a continuous offset; the action chunk is `decode(codes) + offset`.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: InstanceOf[Module],
        projection: InstanceOf[Module],
        heads: InstanceOf[ModuleList],
        offset_head: InstanceOf[Module],
        losses: InstanceOf[ModuleDict],
        chunk: Path,
        norm: InstanceOf[Module] | None = None,
        sample_codes: bool = True,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.projection = projection
        self.heads = heads
        self.offset_head = offset_head
        self.losses = losses  # {"code": ..., "offset": ...}
        self.chunk: Path = chunk
        self.sample_codes = sample_codes

    @override
    def train(self, mode: bool = True) -> "JointPolicyObjective":  # noqa: FBT001, FBT002
        super().train(mode)
        self.tokenizer.eval()
        return self

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

    def _predict_codes(self, features: Tensor) -> Tensor:
        """Autoregressively predict the residual-VQ codes from policy features.

        Each quantizer's code is sampled from its head's logits (VQ-BeT) when
        `sample_codes`, else taken as argmax; the residual is reduced by the
        selected code's codebook vector before the next quantizer.
        """
        tokenizer = self.tokenizer
        residual = self.projection(features)
        codes: list[Tensor] = []
        for q in range(tokenizer.quantizer.num_quantizers):
            logits = self.heads[q](residual)
            code = (
                torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)
                if self.sample_codes
                else logits.argmax(dim=-1)
            )
            codes.append(code)
            residual = residual - F.embedding(code, tokenizer.quantizer.codebook(q))

        return torch.stack(codes, dim=-1)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode, embedding)  # (b, feature_dim)
        tokenizer = self.tokenizer

        with torch.no_grad():
            chunk = episode.get(self.chunk)[:, -1]  # (b, action_clip, action_space)
            codes = tokenizer(chunk)  # (b, num_quantizers) ground-truth codes
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, action_dim): the GT action chunk the policy must reconstruct

        losses: dict[str, Tensor] = {}

        # code heads: teacher-forced classification against the ground-truth codes
        residual = self.projection(features)
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                self.heads[q](residual), codes[:, q]
            )
            residual = residual - F.embedding(
                codes[:, q], tokenizer.quantizer.codebook(q)
            )

        # reconstruct the full action chunk the way inference does
        # (detokenize predicted codes + predicted offset) and train it to match the GT chunk
        with torch.no_grad():
            detokenized = tokenizer.invert(self._predict_codes(features))

        predicted_chunk = detokenized + self.offset_head(features)
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
        timestep_indices = slice(-1, None)

        action_space = tokenizer._action_features  # noqa: SLF001

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            chunk = tokenizer._normalize(  # noqa: SLF001
                episode.get(self.chunk)[:, -1].flatten(-2, -1)
            ).unflatten(-1, (-1, action_space))  # (b, action_horizon, action_space)
            predictions[key] = Prediction(
                value=TensorDict({"joint_actions": chunk}),
                timestep_indices=timestep_indices,
            )

        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            features = self._features(episode, embedding)

            chunk = (
                tokenizer.invert(self._predict_codes(features))
                + self.offset_head(features)
            ).unflatten(-1, (-1, action_space))  # (b, action_horizon, action_space)
            predictions[key] = Prediction(
                value=TensorDict({"joint_actions": chunk}),
                timestep_indices=timestep_indices,
            )

        return TensorDict(predictions).auto_batch_size_(2)
