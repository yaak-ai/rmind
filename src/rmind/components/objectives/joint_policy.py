from collections.abc import Mapping
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
from rmind.components.nn import FeatureFusionPool
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)

type Path = tuple[str, ...]

# a raw feature always carries a trailing feature axis, so (b,) becomes (b, 1)
_MIN_RAW_FEATURE_DIMS = 2


@final
class JointPolicyObjective(Objective):
    """VQ-BeT action-chunk policy (https://arxiv.org/pdf/2403.03181) on fused features.

    A joint head predicts the frozen action tokenizer's residual-VQ codes (one
    categorical per quantizer) plus a code-conditioned continuous offset; the
    predicted chunk is `decode(codes) + offset`.

    The features come from `FeatureFusionPool` rather than a mask-query pool over a
    single timestep: one token per `observation_summary` history tick, one per raw
    route waypoint, and one for raw speed, combined by the pool's learned queries.
    This is the path for checkpoints whose pretraining embedded neither speed nor
    waypoints (the joint PT+FT `no_speed,no_waypoints` lineage) -- those signals can
    only reach the head raw, off `episode.input`.

    The target chunk is likewise not read from `episode.input.joint_actions` (no
    checkpoint in this lineage builds that field); it is stacked here from
    `action_paths`, in the tokenizer's own field order, over the `action_horizon`
    ticks that follow the conditioning tick.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: InstanceOf[Module],
        fusion_pool: InstanceOf[FeatureFusionPool],
        code_head: InstanceOf[Module],
        offset_head: InstanceOf[Module],
        losses: InstanceOf[ModuleDict],
        action_paths: Mapping[str, Path],
        history_steps: int,
        action_horizon: int,
        raw_waypoints_key: Path = ("context", "waypoints"),
        raw_speed_key: Path = ("continuous", "speed"),
        raw_waypoints_dropout: float = 0.0,
        raw_speed_dropout: float = 0.0,
        raw_waypoints_horizon: int | None = None,
        speed_scale: float = 50.0,
        norm: InstanceOf[Module] | None = None,
        sample_codes: bool = True,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.fusion_pool = fusion_pool  # obs-summary history + waypoints + speed
        self.code_head = code_head  # features -> (G*C) code logits
        self.offset_head = (
            offset_head  # features -> (G*C*action_dim): offset per (quantizer, code)
        )
        self.losses = losses  # {"code": ..., "offset": ...}

        # ordered: the stacking order here *is* the action vector layout the
        # tokenizer was trained with, so it must match its `targets` order
        # (gas_pedal, brake_pedal, steering_angle, turn_signal).
        self.action_paths: Mapping[str, Path] = action_paths
        action_features: int = tokenizer._action_features  # noqa: SLF001
        if len(action_paths) != action_features:
            msg = (
                f"action_paths has {len(action_paths)} fields, "
                f"tokenizer expects {action_features}"
            )
            raise ValueError(msg)

        self.history_steps = history_steps
        self.action_horizon = action_horizon
        self.raw_waypoints_key: Path = raw_waypoints_key
        self.raw_speed_key: Path = raw_speed_key
        self.raw_waypoints_dropout = raw_waypoints_dropout
        self.raw_speed_dropout = raw_speed_dropout
        self.raw_waypoints_horizon: int | None = raw_waypoints_horizon
        self.speed_scale = speed_scale
        self.sample_codes = sample_codes

    @override
    def train(self, mode: bool = True) -> "JointPolicyObjective":
        super().train(mode)
        self.tokenizer.eval()
        return self

    @property
    def _conditioning_index(self) -> int:
        """Tick the head conditions on: the last history tick.

        Fixed for both train and validation (unlike `PolicyObjective`, which falls
        back to the last tick when predicting a single action) -- the chunk target
        always spans the `action_horizon` ticks *after* this one, so moving it to -1
        would leak the future into the features. `forward` (export/deployment, where
        the window holds history only) conditions on -1 instead.
        """
        return self.history_steps - 1

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        features = self._features(episode, embedding, idx=-1)
        _, codes, offset = self._predict(features)

        chunk = (self.tokenizer.invert(codes) + offset).unflatten(
            -1,
            (-1, self.tokenizer._action_features),  # noqa: SLF001
        )
        return TensorDict({"joint_actions": chunk})

    def _features(self, episode: Episode, embedding: Tensor, *, idx: int) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)

        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        history = (
            episode
            .index[self._history_window_slice(idx, self.history_steps)]
            .select(k)
            .parse(embedding)
        )
        # (b, t, n, d) -> (b, t*n, d): one token per (tick, summary slot). n == 1 in
        # this lineage (singleton summary tokens), so this is one token per tick.
        obs_summary_history = rearrange(history.get(k), "b t n d -> b (t n) d")

        raw_waypoints = self._raw_feature(
            episode,
            self.raw_waypoints_key,
            idx,
            self.raw_waypoints_dropout,
            horizon=self.raw_waypoints_horizon,
        )
        raw_speed = self._raw_feature(
            episode,
            self.raw_speed_key,
            idx,
            self.raw_speed_dropout,
            scale=self.speed_scale,
        )

        features = self.fusion_pool(
            obs_summary_history=obs_summary_history,
            raw_waypoints=raw_waypoints,
            raw_speed=raw_speed,
        )  # (b, 1, num_queries*d)
        return features.squeeze(-2)  # (b, num_queries*d)

    @staticmethod
    def _history_window_slice(idx: int, window: int) -> slice:
        """Slice selecting the `window` ticks ending at (and including) tick `idx`.

        `idx` is either a small non-negative int (`_conditioning_index`) or -1
        (`forward`). For idx == -1, `idx + 1 == 0` would mean "up to but excluding
        tick 0" (empty) rather than "through the last tick", so stop is None there.
        For idx >= 0 the start is clamped at 0 ourselves -- left negative it would be
        reinterpreted as "N from the end" and silently yield the wrong window.
        """
        if idx >= 0:
            return slice(max(idx - window + 1, 0), idx + 1)

        return slice(idx - window + 1, None if idx == -1 else idx + 1)

    def _raw_feature(  # noqa: PLR0913
        self,
        episode: Episode,
        key: Path,
        idx: int,
        dropout: float,
        *,
        scale: float = 1.0,
        horizon: int | None = None,
    ) -> Tensor:
        """Read an un-embedded feature straight off `episode.input` at tick `idx`.

        `scale` divides raw physical units (e.g. km/h speed) down to roughly the
        network's activation scale; `horizon` keeps only the nearest `horizon`
        waypoints. In training, zeroes the *whole* vector for `dropout` of samples
        (modality dropout, not elementwise) so the head can't just copy the route
        or the speed instead of reading the observation history.
        """
        x = episode.input.get(key)[:, idx] / scale  # (b, ...)
        if horizon is not None:
            x = x[:, :horizon]

        # guarantee a trailing feature axis: scalar-per-tick speed (b,) -> (b, 1)
        out = x if x.dim() >= _MIN_RAW_FEATURE_DIMS else x.unsqueeze(-1)

        if self.training and dropout > 0.0:
            keep_shape = (out.shape[0], *(1,) * (out.dim() - 1))
            keep = (torch.rand(keep_shape, device=out.device) >= dropout).to(out.dtype)
            out *= keep

        return out

    def _action_chunk(self, episode: Episode) -> Tensor:
        """(b, action_horizon, action_features): the chunk the policy must reproduce.

        The ticks after the conditioning tick, stacked in `action_paths` order. Each
        field is (b, h, 1) on this branch (AtLeast3D in the input transform), so a
        trailing cat yields the (chunk, fields) layout `ActionTokenizer` flattens.
        `turn_signal` arrives as an integer code and is rescaled to [0, 1] by the
        tokenizer's own `_normalize`, so no scaling happens here.
        """
        start = self.history_steps
        stop = start + self.action_horizon
        fields = [
            episode.input.get(path)[:, start:stop]
            for path in self.action_paths.values()
        ]
        dtype = fields[0].dtype  # gas_pedal leads and is float; turn_signal is integer

        return torch.cat([field.to(dtype) for field in fields], dim=-1)

    def _predict(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """VQ-BeT joint code prediction with a code-conditioned offset."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size

        code_logits = rearrange(self.code_head(features), "b (g c) -> b g c", g=g, c=c)
        if self.sample_codes:
            codes = rearrange(
                torch.multinomial(code_logits.softmax(dim=-1).reshape(-1, c), 1),
                "(b g) 1 -> b g",
                g=g,
            )
        else:
            codes = code_logits.argmax(dim=-1)

        return code_logits, codes, self._gather_offset(features, codes)

    def _gather_offset(self, features: Tensor, codes: Tensor) -> Tensor:
        """Select each quantizer's offset at `codes` and sum over quantizers."""
        quantizer = self.tokenizer.quantizer
        g, c = quantizer.num_quantizers, quantizer.codebook_size
        offsets = rearrange(
            self.offset_head(features), "b (g c a) -> b g c a", g=g, c=c
        )
        index = codes[..., None, None].expand(-1, -1, 1, offsets.shape[-1])
        # https://arxiv.org/pdf/2403.03181 Figure 2.
        return offsets.gather(2, index).squeeze(2).sum(dim=1)  # (b, action_dim)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._features(
            episode, embedding, idx=self._conditioning_index
        )  # (b, feature_dim)
        tokenizer = self.tokenizer

        with torch.no_grad():
            chunk = self._action_chunk(episode)  # (b, action_horizon, action_space)
            target_codes = tokenizer(chunk)  # (b, num_quantizers) ground-truth codes
            target = tokenizer._normalize(  # noqa: SLF001
                chunk.flatten(-2, -1)
            )  # (b, action_dim): the GT action chunk the policy must reconstruct

        code_logits, _, _ = self._predict(features)

        losses: dict[str, Tensor] = {}

        # per-quantizer classification against the ground-truth codes
        for q in range(tokenizer.quantizer.num_quantizers):
            losses[f"code_{q}"] = self.losses["code"](
                code_logits[:, q, :], target_codes[:, q]
            )

        # reconstruct the chunk as inference does (decode codes + code-conditioned
        # offset); the frozen tokenizer makes invert(codes) gradient-free, so this
        # term trains only the offset head
        predicted_chunk = tokenizer.invert(target_codes) + self._gather_offset(
            features, target_codes
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
        action_space = tokenizer._action_features  # noqa: SLF001

        # the chunk *is* the ticks after the conditioning tick, for both the
        # prediction and the ground truth, so they line up tick-for-tick
        timestep_indices = slice(
            self.history_steps, self.history_steps + self.action_horizon
        )

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            chunk = tokenizer._normalize(  # noqa: SLF001
                self._action_chunk(episode).flatten(-2, -1)
            ).unflatten(-1, (-1, action_space))  # (b, action_horizon, action_space)
            predictions[key] = Prediction(
                value=self._as_actions(chunk, discretize=False),
                timestep_indices=timestep_indices,
            )

        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            features = self._features(episode, embedding, idx=self._conditioning_index)
            _, codes, offset = self._predict(features)

            chunk = (tokenizer.invert(codes) + offset).unflatten(
                -1, (-1, action_space)
            )  # (b, action_horizon, action_space)
            predictions[key] = Prediction(
                value=self._as_actions(chunk, discretize=True),
                timestep_indices=timestep_indices,
            )

        return TensorDict(predictions).auto_batch_size_(2)

    @staticmethod
    def _as_actions(chunk: Tensor, *, discretize: bool) -> TensorDict:
        """Split a normalized chunk back into named action fields.

        `turn_signal` lives on the same [0, 1] scale the tokenizer normalized it to,
        so both paths undo that: ground truth by scaling {0.0, 0.5, 1.0} back to
        {OFF, LEFT, RIGHT} exactly, a prediction by bucketizing to the nearest of
        those three centers.
        """
        turn_signal = (
            torch.bucketize(
                chunk[..., 3] * 2, torch.tensor([0.5, 1.5], device=chunk.device)
            )
            if discretize
            else (chunk[..., 3] * 2).round().long()
        )

        return TensorDict({
            "continuous": TensorDict({
                "gas_pedal": chunk[..., 0],
                "brake_pedal": chunk[..., 1],
                "steering_angle": chunk[..., 2],
            }),
            "discrete": TensorDict({"turn_signal": turn_signal}),
        })
