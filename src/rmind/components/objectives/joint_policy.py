from collections.abc import Set as AbstractSet
from typing import Any, Literal, final, override

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
        offset_scale: float | None = None,
        decode_strategy: Literal["argmax", "chain_greedy", "heads"] = "argmax",
        decode_beta: float = 1.0,
        read_waypoints: bool | None = None,
        foresight_attn: InstanceOf[Module] | None = None,
        foresight_maxpool: bool = False,
        foresight_key: str = "cam_front_left",
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
        self.offset_scale = offset_scale

        # Foresight readout for the policy head (spinoffs 1.1 / 2.4). Both are
        # ADDITIVE: their outputs are concatenated onto [observation_summary,
        # observation_history] in _features, never replacing them.
        #   foresight_attn    (1.1): learned multi-query attention pooling over the
        #                            256 foresight slots -> (b, num_queries * d).
        #   foresight_maxpool (2.4): channel-wise max over the foresight slots ->
        #                            (b, d), preserving onset peaks a single
        #                            softmax-attention token averages away.
        # read_waypoints=None keeps the legacy auto-detect (waypoint inclusion
        # inferred from the code head's input width); it must be set explicitly
        # once a foresight branch is enabled, since the head width is then no
        # longer 2*d / 3*d. code/offset head in_channels must equal the assembled
        # feature width (validated on first _features call).
        self.read_waypoints = read_waypoints
        self.foresight_attn: Module | None = foresight_attn
        self.foresight_maxpool = foresight_maxpool
        self.foresight_key = foresight_key
        self._code_head_in: int = next(
            m.in_features
            for m in self.code_head.modules()
            if isinstance(m, torch.nn.Linear)
        )

        # inference-time code decoding (train/eval always argmax-or-sampled via
        # _sample_codes; these govern forward/predict, i.e. the exported graph):
        #   "argmax"       per-quantizer independent argmax (deployment default)
        #   "chain_greedy" greedy coarse->fine argmax over
        #                  log_softmax(logits_g) + decode_beta * log P(c_g | c_<g),
        #                  with the empirical chain conditionals in `chain_log_prior_*`
        #                  buffers (install via scripts.calibrate_decode_luts).
        # Deterministic + export/TensorRT-clean. Stochastic decoding stays host-side
        # off a "heads" export (see scripts/export_onnx --heads).
        self.decode_strategy = decode_strategy
        self.decode_beta = decode_beta
        g = self.tokenizer.quantizer.num_quantizers
        c = self.tokenizer.quantizer.codebook_size
        # one chain-conditional table per quantizer level g: shape (c**g, c),
        # indexed by the packed prefix (c_0..c_{g-1}); level 0 is (1, c). Registered
        # (not trained) so they persist in the checkpoint once calibrated; default
        # all-zeros = uniform prior = decode collapses to plain argmax until filled.
        for level in range(g):
            self.register_buffer(
                f"chain_log_prior_{level}", torch.zeros(c**level, c), persistent=True
            )

    @override
    def train(self, mode: bool = True) -> "JointPolicyObjective":
        super().train(mode)
        self.tokenizer.eval()
        return self

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        features = self._features(episode, embedding)

        if self.decode_strategy == "heads":
            # export the raw heads (deterministic, TensorRT-clean) so decode
            # (incl. stochastic/entropy-gated sampling, which needs RNG that does
            # not belong in a TRT engine) runs host-side in drivr on the small
            # (b,G,C) logits + (b,G,C,action_dim) offset table.
            code_logits, offsets = self._heads(features)
            return TensorDict({"code_logits": code_logits, "offsets": offsets})

        _, codes, offset = self._predict(features)
        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({"joint_actions": chunk})

    def _features(self, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)

        keys = [
            (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
            (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
            (Modality.CONTEXT, "waypoints"),
        ]
        use_foresight = self.foresight_attn is not None or self.foresight_maxpool
        if use_foresight:
            keys.append((Modality.FORESIGHT, self.foresight_key))

        embeddings = episode.index[-1].select(*keys).parse(embedding)

        observation_summary = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_SUMMARY,
        ))  # (b, 1, d)
        observation_history = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_HISTORY,
        ))  # (b, 1, d)
        d = observation_summary.shape[-1]

        # Waypoint inclusion: legacy path (read_waypoints=None, no foresight
        # branch) infers it from the code head's input width — 3 tokens (3*d,
        # waypoint-aware) vs 2 (2*d, "no waypoints in policy head" finetunes,
        # e.g. feat/action-tokenizer @ ad93596) — so one code path serves both
        # families. With a foresight branch the head width is no longer 2*d/3*d,
        # so read_waypoints must be set explicitly in the config.
        if self.read_waypoints is None:
            read_waypoints = self._code_head_in == 3 * d
        else:
            read_waypoints = self.read_waypoints

        parts: list[Tensor] = [
            observation_summary.squeeze(1),
            observation_history.squeeze(1),
        ]  # each (b, d)
        if read_waypoints:
            parts.append(embeddings.get((Modality.CONTEXT, "waypoints")).mean(dim=1))
        if use_foresight:
            foresight = embeddings.get((
                Modality.FORESIGHT,
                self.foresight_key,
            ))  # (b, n, d)
            if self.foresight_attn is not None:
                parts.append(self.foresight_attn(foresight))  # (b, num_queries * d)
            if self.foresight_maxpool:
                parts.append(foresight.amax(dim=1))  # (b, d)

        features = torch.cat(parts, dim=-1)  # (b, W)
        if features.shape[-1] != self._code_head_in:
            msg = (
                f"assembled policy-head width {features.shape[-1]} != code_head "
                f"in_features {self._code_head_in}; set code/offset head "
                f"in_channels to the assembled width (waypoints + foresight "
                f"branches) or fix read_waypoints"
            )
            raise ValueError(msg)
        return features

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

    def _offset(self, offsets: Tensor, codes: Tensor) -> Tensor:
        """Gathered offset, optionally soft-bounded to (-offset_scale, offset_scale).

        The bound is `scale * tanh(x / scale)`: identity for |x| << scale, so
        enabling it on a cleanly-trained model preserves sub-cell offsets while
        capping cross-cell excursions.
        """
        offset = self._gather_offset(offsets, codes)
        if self.offset_scale is None:
            return offset
        return torch.tanh(offset / self.offset_scale) * self.offset_scale

    def _sample_codes(self, code_logits: Tensor) -> Tensor:
        if self.sample_codes:
            _, g, c = code_logits.shape
            return rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            )
        return code_logits.argmax(dim=-1)

    def _chain_greedy(self, code_logits: Tensor) -> Tensor:
        """Greedy coarse->fine decode over the empirical chain conditionals.

        c_g = argmax_c [ log_softmax(logits_g)_c + beta * log P(c | c_0..c_{g-1}) ]
        with the chain priors in the `chain_log_prior_{g}` buffers (all-zeros =
        uniform ⇒ this reduces exactly to per-quantizer argmax until calibrated).
        Deterministic; the loop is over the fixed quantizer count so it unrolls
        under torch.export, and each step is gather + argmax + add (TensorRT-clean).
        """
        _, g, c = code_logits.shape
        logp = code_logits.log_softmax(dim=-1)
        prefix = torch.zeros(
            code_logits.shape[0], dtype=torch.long, device=code_logits.device
        )
        codes: list[Tensor] = []
        for level in range(g):
            prior = getattr(self, f"chain_log_prior_{level}")[prefix]  # (b, c)
            code = (logp[:, level] + self.decode_beta * prior).argmax(dim=-1)
            codes.append(code)
            prefix = prefix * c + code
        return torch.stack(codes, dim=1)

    def _decode_codes(self, code_logits: Tensor) -> Tensor:
        """Inference code decode (forward/predict, i.e. the exported graph)."""
        if self.decode_strategy == "chain_greedy":
            return self._chain_greedy(code_logits)
        return self._sample_codes(code_logits)  # argmax (export) or sampled

    def _predict(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """VQ-BeT joint code prediction with a code-conditioned offset."""
        code_logits, offsets = self._heads(features)
        codes = self._decode_codes(code_logits)
        offset = self._offset(offsets, codes)

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
        sampled_chunk = tokenizer.invert(codes) + self._offset(offsets, codes)
        sampled_recon = self.losses["offset"](sampled_chunk.detach(), target)

        if self.teacher_force_offset:
            # offset supervised at the GROUND-TRUTH codes (teacher forcing), so each
            # code's offset entry only sees residuals of actions that quantized to it;
            # supervising at sampled codes lets the offset learn the conditional median
            # regardless of code, cancelling the code selection
            predicted_chunk = tokenizer.invert(target_codes) + self._offset(
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
