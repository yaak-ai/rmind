"""Token-time flow matching, ported from the pinned RTC Kinetix policy.

All actions, prefixes and noise in this module use normalized policy units.
The legacy flat-MLP ActionHead is deliberately unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from rmind.components.fm_action_head import ActionHead
from rmind.components.rtc_time_embedding import native_time_frequencies

if TYPE_CHECKING:
    from collections.abc import Mapping


class RTCNorm(nn.Module):
    """Flax LayerNorm's non-affine, fast-variance convention."""

    @override
    def forward(self, value: Tensor) -> Tensor:
        mean = value.mean(dim=-1, keepdim=True)
        variance = (
            value.square().mean(dim=-1, keepdim=True) - mean.square()
        ).clamp_min(0)
        return (value - mean) * torch.rsqrt(variance + 1e-6)


class RTCMixerBlock(nn.Module):
    def __init__(
        self, horizon: int, token_hidden: int, channels: int, channel_hidden: int
    ) -> None:
        super().__init__()
        self.token_mix_in = nn.Linear(horizon, token_hidden, bias=False)
        self.token_mix_out = nn.Linear(token_hidden, horizon, bias=False)
        self.channel_mix_in = nn.Linear(channels, channel_hidden, bias=False)
        self.channel_mix_out = nn.Linear(channel_hidden, channels, bias=False)
        self.norm_1 = RTCNorm()
        self.norm_2 = RTCNorm()
        self.adaln_1 = nn.Linear(channels, 3 * channels)
        self.adaln_2 = nn.Linear(channels, 3 * channels)
        for layer in (self.adaln_1, self.adaln_2):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: Tensor, conditioning: Tensor) -> Tensor:
        scale_1, shift_1, gate_1 = self.adaln_1(conditioning).chunk(3, dim=-1)
        scale_2, shift_2, gate_2 = self.adaln_2(conditioning).chunk(3, dim=-1)
        mixed = (self.norm_1(value) * (1 + scale_1) + shift_1).transpose(1, 2)
        mixed = self.token_mix_out(
            functional.gelu(self.token_mix_in(mixed), approximate="tanh")
        )
        value = torch.add(value, gate_1 * mixed.transpose(1, 2))
        mixed = self.norm_2(value) * (1 + scale_2) + shift_2
        mixed = self.channel_mix_out(
            functional.gelu(self.channel_mix_in(mixed), approximate="tanh")
        )
        return value + gate_2 * mixed


class TokenTimeFlowHead(ActionHead):
    """ActionHead-compatible RTC head with explicit clean-prefix conditioning.

    ``max_prefix`` is the largest trained delay, inclusive. Training samples
    lengths with probability proportional to exp(-length), as the reference.
    Explicit randomness APIs make conversion and cross-runtime tests auditable.

    New heads derive native frequencies analytically, with no external vector.
    Explicit imported ``time_frequency`` selects reference compatibility when
    loading older converted configurations without a mode. ``legacy_dynamic``
    preserves the former per-forward FP32 arithmetic only when requested.
    """

    rtc_variant = "token_time_clean_prefix_v1"

    @override
    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        action_horizon: int,
        channel_dim: int = 256,
        channel_hidden_dim: int = 512,
        token_hidden_dim: int = 64,
        num_layers: int = 4,
        max_prefix: int = 4,
        time_frequency: list[float] | tuple[float, ...] | None = None,
        time_embedding_mode: str | None = None,
        time_min_period: float = 4e-3,
        time_max_period: float = 4.0,
    ) -> None:
        nn.Module.__init__(self)
        if not 0 <= max_prefix < action_horizon:
            msg = "max_prefix must leave at least one predicted action"
            raise ValueError(msg)
        if (
            channel_dim % 2
            or min(feat_dim, action_dim, action_horizon, channel_dim, num_layers) < 1
        ):
            msg = "dimensions must be positive and channel_dim must be even"
            raise ValueError(msg)
        self.kind = "fm"
        self.D = action_horizon * action_dim
        self.K = 1
        self.feat_dim = feat_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.channel_dim = channel_dim
        self.max_prefix = max_prefix
        self.fm_steps = 5
        self.register_buffer("progress", torch.zeros(()))
        if time_embedding_mode is None:
            time_embedding_mode = (
                "reference_compatibility" if time_frequency is not None else "native"
            )
        if time_embedding_mode not in {
            "native",
            "reference_compatibility",
            "legacy_dynamic",
        }:
            msg_0 = "invalid time_embedding_mode"
            raise ValueError(msg_0)
        if (time_embedding_mode == "reference_compatibility") != (
            time_frequency is not None
        ):
            msg_0 = "only reference_compatibility accepts a supplied time_frequency"
            raise ValueError(msg_0)
        self.time_embedding_mode = time_embedding_mode
        self.time_min_period = time_min_period
        self.time_max_period = time_max_period
        frequency = None
        if time_embedding_mode == "native":
            frequency = native_time_frequencies(
                channel_dim, time_min_period, time_max_period
            )
        elif time_frequency is not None:
            frequency = torch.tensor(time_frequency, dtype=torch.float32)
            if frequency.shape != (channel_dim // 2,) or not (
                torch.isfinite(frequency).all() and (frequency > 0).all()
            ):
                msg = "time_frequency must contain channel_dim/2 finite positive values"
                raise ValueError(msg)
        self.register_buffer("time_frequency", frequency)
        self.in_proj = nn.Linear(feat_dim + action_dim, channel_dim)
        self.mlp_stack = nn.ModuleList([
            RTCMixerBlock(
                action_horizon, token_hidden_dim, channel_dim, channel_hidden_dim
            )
            for _ in range(num_layers)
        ])
        self.time_mlp = nn.Sequential(
            nn.Linear(channel_dim, channel_dim),
            nn.SiLU(),
            nn.Linear(channel_dim, channel_dim),
            nn.SiLU(),
        )
        self.final_norm = RTCNorm()
        self.final_adaln = nn.Linear(channel_dim, 2 * channel_dim)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.out_proj = nn.Linear(channel_dim, action_dim)

    @override
    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.time_embedding_mode == "native":
            incoming = state_dict.get(prefix + "time_frequency")
            expected = native_time_frequencies(
                self.channel_dim, self.time_min_period, self.time_max_period
            )
            if incoming is not None and not torch.equal(
                incoming.detach().cpu(), expected
            ):
                error_msgs.append(
                    prefix
                    + "time_frequency does not match the native analytical recipe"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def velocity(self, features: Tensor, state: Tensor, time: Tensor) -> Tensor:
        flattened_dimensions = 2
        flattened = state.ndim == flattened_dimensions
        state = state.reshape(-1, self.action_horizon, self.action_dim)
        if time.ndim == 1:
            time = time[:, None]
        time = time.expand(state.shape[0], self.action_horizon)
        if self.time_frequency is None:
            fraction = torch.linspace(
                0, 1, self.channel_dim // 2, device=time.device, dtype=time.dtype
            )
            period = (
                self.time_min_period
                * (self.time_max_period / self.time_min_period) ** fraction
            )
            frequency = 1.0 / period * (2 * math.pi)
        else:
            frequency = self.time_frequency.to(dtype=time.dtype)
        angle = time[..., None] * frequency
        conditioning = self.time_mlp(torch.cat((angle.sin(), angle.cos()), dim=-1))
        value = self.in_proj(
            torch.cat(
                (state, features[:, None].expand(-1, self.action_horizon, -1)), dim=-1
            )
        )
        for layer in self.mlp_stack:
            value = layer(value, conditioning)
        scale, shift = self.final_adaln(conditioning).chunk(2, dim=-1)
        result = self.out_proj(self.final_norm(value) * (1 + scale) + shift)
        return result.flatten(1) if flattened else result

    def sample(
        self,
        features: Tensor,
        noise: Tensor,
        clean_prefix: Tensor,
        prefix_mask: Tensor,
        steps: int = 5,
    ) -> Tensor:
        """Euler sampler; exported graphs require host-validated prefix inputs.

        Raises:
            ValueError: Steps or eager prefix inputs violate the trained contract.
        """
        if steps < 1:
            msg = "Euler steps must be positive"
            raise ValueError(msg)
        state = noise.reshape(-1, self.action_horizon, self.action_dim)
        clean_prefix = clean_prefix.reshape_as(state)
        raw_mask = prefix_mask.reshape(-1, self.action_horizon)
        if not torch.compiler.is_compiling():
            if not torch.isfinite(clean_prefix).all():
                msg_0 = "RTC prefix actions must be finite"
                raise ValueError(msg_0)
            if not ((raw_mask == 0) | (raw_mask == 1)).all():
                msg_0 = "RTC prefix mask must be binary"
                raise ValueError(msg_0)
            if (raw_mask[:, 1:] > raw_mask[:, :-1]).any():
                msg_0 = "RTC prefix mask must be contiguous from token zero"
                raise ValueError(msg_0)
            if (raw_mask.sum(dim=-1) > self.max_prefix).any():
                msg_0 = "RTC prefix exceeds the trained maximum"
                raise ValueError(msg_0)
        mask = raw_mask.to(torch.bool)
        time = state.new_zeros((state.shape[0], self.action_horizon))
        for _ in range(steps):
            state = torch.where(mask[..., None], clean_prefix, state)
            token_time = torch.where(mask, 1.0, time)
            state = torch.add(
                state, self.velocity(features, state, token_time) * (1.0 / steps)
            )
            time = torch.add(time, 1.0 / steps)
        return torch.where(mask[..., None], clean_prefix, state)

    def loss_explicit(
        self,
        features: Tensor,
        target: Tensor,
        noise: Tensor,
        time: Tensor,
        prefix_lengths: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        target = target.reshape(-1, self.action_horizon, self.action_dim)
        noise = noise.reshape_as(target)
        prefix = (
            torch.arange(self.action_horizon, device=target.device)[None]
            < prefix_lengths[:, None]
        )
        token_time = torch.where(prefix, 1.0, time[:, None])
        state = (1 - token_time[..., None]) * noise + token_time[..., None] * target
        prediction = self.velocity(features, state, token_time)
        weights = (~prefix)[..., None].expand_as(target).to(target.dtype)
        if mask is not None:
            weights = torch.mul(weights, mask.reshape_as(target))
        return (
            (prediction - (target - noise)).square() * weights
        ).sum() / weights.sum().clamp_min(1e-8)

    @override
    def loss(
        self,
        features: Tensor,
        conditioning: Tensor | None,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        probabilities = torch.softmax(
            -torch.arange(
                self.max_prefix + 1, device=target.device, dtype=target.dtype
            ),
            dim=0,
        )
        lengths = torch.multinomial(probabilities, target.shape[0], replacement=True)
        loss = self.loss_explicit(
            features,
            target,
            torch.randn_like(target),
            torch.rand(target.shape[0], device=target.device, dtype=target.dtype),
            lengths,
            mask=mask,
        )
        return loss, {"fm_mse": loss.detach()}

    @torch.no_grad()
    def act(self, features: Tensor, mode: str = "argmax") -> tuple[Tensor, None, None]:
        noise = features.new_zeros((
            features.shape[0],
            self.action_horizon,
            self.action_dim,
        ))
        if mode == "sample":
            noise = torch.randn_like(noise)
        return (
            self.sample(
                features,
                noise,
                torch.zeros_like(noise),
                noise.new_zeros(noise.shape[:2]),
                self.fm_steps,
            ).flatten(1),
            None,
            None,
        )


def load_kinetix_state(head: TokenTimeFlowHead, source: Mapping) -> None:
    """Load a trusted, already-deserialized Flax parameter dictionary exactly.

    Raises:
        ValueError: A source parameter shape differs from the destination.
    """
    converted = {}
    for name, destination in head.state_dict().items():
        if name in {"progress", "time_frequency"}:
            converted[name] = destination
            continue
        path = name.split(".")
        if path[0] == "time_mlp":
            path.insert(1, "layers")
        path[-1] = "kernel" if path[-1] == "weight" else path[-1]
        value = source
        for component in path:
            key = int(component) if component.isdigit() else component
            value = value[key]
        tensor = torch.as_tensor(value.copy())
        if path[-1] == "kernel":
            tensor = tensor.T
        if tensor.shape != destination.shape:
            msg = f"checkpoint shape mismatch at {name}: {tensor.shape} != {destination.shape}"
            raise ValueError(msg)
        converted[name] = tensor
    head.load_state_dict(converted, strict=True)
