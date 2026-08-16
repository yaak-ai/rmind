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
        condition: InstanceOf[Module],
        query: Path,
        decoder: InstanceOf[Module],
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        losses: InstanceOf[ModuleDict],
        chunk: Path,
        norm: InstanceOf[Module] | None = None,
        sample_codes: bool = True,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.condition = condition
        self.query: Path = query
        self.decoder = decoder
        self.code_head = code_head
        self.offset_head = offset_head
        self.losses = losses
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
        _, codes, offset = self._predict(features)

        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({"joint_actions": chunk})

    def _features(self, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)

        last = episode.index[-1]
        k_os = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        observation_summary = last.select(k_os).parse(embedding).get(k_os)  # (b, 64, d)

        patches = episode.get(self.query)[:, -1]  # (b, p, d)
        # observation_summary routes attention over patches without entering the values
        key_policy = self.condition({
            "query": patches,
            "key": observation_summary,
            "value": observation_summary,
        })

        mask = episode.embeddings.get((Modality.UTILITY, "mask"))[
            :, -1, [3]
        ]  # (b, 1, d)
        return self.decoder({
            "query": mask,
            "key": key_policy,
            "value": patches,
        }).squeeze(-2)  # (b, d)

    def _predict(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """VQ-BeT joint code prediction with a code-conditioned offset."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        code_logits = rearrange(self.code_head(features), "b (g c) -> b g c", g=g, c=c)
        codes = code_logits.argmax(dim=-1)

        if self.sample_codes:
            codes = rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            )

        offsets = rearrange(
            self.offset_head(features), "b (g c a) -> b g c a", g=g, c=c
        )
        index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
        offset = offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

        return code_logits, codes, offset

    def _gather_offset(self, features: Tensor, codes: Tensor) -> Tensor:
        """Select each quantizer's offset at `codes` and sum over quantizers."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        offsets = rearrange(
            self.offset_head(features), "b (g c a) -> b g c a", g=g, c=c
        )
        index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
        return offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(episode, embedding)
        tokenizer = self.tokenizer
        noise_mask = episode.noise_mask

        with torch.no_grad():
            chunk = episode.get(self.chunk)[:, -1]  # (b, action_clip, action_space)
            target_codes = tokenizer(chunk)  # (b, num_quantizers) ground-truth codes
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, action_dim): the GT action chunk the policy must reconstruct

        if noise_mask is not None:
            random_codes = torch.randint(
                0,
                tokenizer.quantizer.codebook_size,
                target_codes.shape,
                device=target_codes.device,
            )
            target_codes = torch.where(noise_mask[:, None], random_codes, target_codes)

        code_logits, _, _ = self._predict(features)

        losses: dict[str, Tensor] = {}

        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                code_logits[:, q, :], target_codes[:, q]
            )

        if noise_mask is not None:
            clean = ~noise_mask
            features_clean, target_codes_clean, target_clean = (
                features[clean],
                target_codes[clean],
                target[clean],
            )
        else:
            features_clean, target_codes_clean, target_clean = (
                features,
                target_codes,
                target,
            )

        predicted_chunk = tokenizer.invert(target_codes_clean) + self._gather_offset(
            features_clean, target_codes_clean
        )
        losses["offset"] = self.losses["offset"](predicted_chunk, target_clean)

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
            _, codes, offset = self._predict(features)

            chunk = (tokenizer.invert(codes) + offset).unflatten(
                -1, (-1, action_space)
            )  # (b, action_horizon, action_space)
            offset_unflat = offset.unflatten(
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
                "codes": codes.float(),  # (b, num_quantizers)
                "offset": TensorDict({
                    "gas_pedal": offset_unflat[..., 0],
                    "brake_pedal": offset_unflat[..., 1],
                    "steering_angle": offset_unflat[..., 2],
                    "turn_signal": offset_unflat[..., 3],
                }),
            })
            predictions[key] = Prediction(value=actions, time_index=time_index)

        return TensorDict(predictions).auto_batch_size_(2)
