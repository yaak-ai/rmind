import math
from typing import Literal, Self, final, override

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, ConfigDict, model_validator, validate_call
from torch import Tensor, nn

from rmind.components.position_encoding import RotaryPositionalEmbeddings
from rmind.components.transformer.feed_forward import MLPGLU
from rmind.components.transformer.utils import run_layer_stack

FlowSamplingMethod = Literal["euler", "midpoint", "heun"]


class CrossAttentionDecoderBlock(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
    ) -> None:
        super().__init__()

        self.cross_attn_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.cross_attn_resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.self_attn_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.self_attn_resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

    @override
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        residual = x
        x_norm = self.cross_attn_norm(x)
        cross_attn_out, _ = self.cross_attn(
            query=x_norm, key=context, value=context, need_weights=False
        )
        x = residual + self.cross_attn_resid_drop(cross_attn_out)

        residual = x
        x_norm = self.self_attn_norm(x)
        self_attn_out, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm, need_weights=False
        )
        x = residual + self.self_attn_resid_drop(self_attn_out)

        residual = x
        mlp_out = self.mlp(self.mlp_norm(x))
        return residual + mlp_out


class CrossAttentionDecoder(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionDecoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                resid_dropout=resid_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])

    @override
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        return run_layer_stack(self.layers, x, context, training=self.training)


@final
class CrossAttentionDecoderHead(nn.Module):
    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        query: Tensor
        context: Tensor

        @model_validator(mode="after")
        def _validate_shapes(self) -> Self:
            if self.query.ndim != self.context.ndim or self.query.ndim not in {3, 4}:
                msg = (
                    "query/context must both be 3D or 4D with matching ndim, "
                    f"got query={self.query.ndim}D, context={self.context.ndim}D"
                )
                raise ValueError(msg)
            return self

    def __init__(
        self, decoder: CrossAttentionDecoder, output_projection: nn.Linear
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.output_projection = output_projection

    @validate_call
    @override
    def forward(self, input: Input) -> Tensor:
        query = input.query
        context = input.context

        if query.ndim == 4:  # noqa: PLR2004
            b, t, sq, d = query.shape
            _, _, sc, _ = context.shape

            query_flat = query.reshape(b * t, sq, d)
            context_flat = context.reshape(b * t, sc, d)

            decoded = self.decoder(query_flat, context_flat)
            output = self.output_projection(decoded)

            return output.reshape(b, t, sq, d)

        decoded = self.decoder(query, context)
        return self.output_projection(decoded)


class SinusoidalTimeEmbedding(nn.Module):
    # Time embedding / AdaLN conditioning follows the SiT design space:
    # https://arxiv.org/abs/2401.08740
    # Official code: https://github.com/willisma/SiT
    def __init__(
        self, dim: int, *, scale: float = 1000.0, logit_scale: float = 0.25
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            msg = f"dim must be even, got {dim}"
            raise ValueError(msg)
        if scale <= 0.0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)
        if logit_scale < 0.0:
            msg = f"logit_scale must be non-negative, got {logit_scale}"
            raise ValueError(msg)
        self.dim = dim
        self.scale = scale
        self.logit_scale = logit_scale

    @override
    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = (t * self.scale).unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        embedding[:, 0] = t

        if self.logit_scale > 0.0:
            eps = torch.finfo(t.dtype).eps
            embedding[:, half] = torch.logit(t.clamp(eps, 1.0 - eps)) * self.logit_scale

        return embedding


class RotarySelfAttention(nn.Module):
    """Self-attention with rotary position embeddings on queries/keys.

    Unlike the additive learned position embedding (applied once at the
    decoder input), rotary phases are injected into the attention logits of
    every layer that uses this module, so slot identity is structurally
    available at full strength and cannot be washed out by the residual
    stream.
    """

    @validate_call
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 64,
        base: int = 10,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            msg = f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            msg = f"head_dim must be even for rotary embeddings, got {head_dim}"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=max_seq_len, base=base
        )

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.qkv_proj(x), "b s (three h d) -> three b s h d", three=3,
            h=self.num_heads,
        )
        q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class AdaLNCrossAttentionDecoderBlock(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        embedding_dim: int,
        num_heads: int,
        time_dim: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        rope: bool = False,
    ) -> None:
        super().__init__()

        self.cross_attn_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.cross_attn_resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.self_attn_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.self_attn: nn.Module = (
            RotarySelfAttention(
                embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout
            )
            if rope
            else nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        )
        self.self_attn_resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.mlp_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.mlp = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

        # 6 outputs: (scale, shift) for each of the 3 norms
        self.adaLN_modulation = nn.Linear(time_dim, 6 * embedding_dim)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    @override
    def forward(self, x: Tensor, context: Tensor, time_emb: Tensor) -> Tensor:
        s_ca, b_ca, s_sa, b_sa, s_mlp, b_mlp = self.adaLN_modulation(time_emb).chunk(
            6, dim=-1
        )
        s_ca, b_ca = s_ca.unsqueeze(1), b_ca.unsqueeze(1)
        s_sa, b_sa = s_sa.unsqueeze(1), b_sa.unsqueeze(1)
        s_mlp, b_mlp = s_mlp.unsqueeze(1), b_mlp.unsqueeze(1)

        residual = x
        x_norm = (1 + s_ca) * self.cross_attn_norm(x) + b_ca
        cross_attn_out, _ = self.cross_attn(
            query=x_norm, key=context, value=context, need_weights=False
        )
        x = residual + self.cross_attn_resid_drop(cross_attn_out)

        residual = x
        x_norm = (1 + s_sa) * self.self_attn_norm(x) + b_sa
        if isinstance(self.self_attn, RotarySelfAttention):
            self_attn_out = self.self_attn(x_norm)
        else:
            self_attn_out, _ = self.self_attn(
                query=x_norm, key=x_norm, value=x_norm, need_weights=False
            )
        x = residual + self.self_attn_resid_drop(self_attn_out)

        residual = x
        mlp_out = self.mlp((1 + s_mlp) * self.mlp_norm(x) + b_mlp)
        return residual + mlp_out


class AdaLNCrossAttentionDecoder(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        time_dim: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        rope: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AdaLNCrossAttentionDecoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                time_dim=time_dim,
                rope=rope,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                mlp_dropout=mlp_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])

    @override
    def forward(self, x: Tensor, context: Tensor, time_emb: Tensor) -> Tensor:
        return run_layer_stack(
            self.layers, x, context, time_emb, training=self.training
        )


@final
class FlowActionDecoder(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        condition_dim: int,
        dim_model: int | None = None,
        action_dim: int = 3,
        action_horizon: int = 6,
        flow_sampling_steps: int = 8,
        flow_sampling_method: FlowSamplingMethod = "euler",
        time_embedding_scale: float = 1000.0,
        time_logit_scale: float = 0.25,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        rope: bool = False,
    ) -> None:
        super().__init__()

        if action_dim <= 0:
            msg = f"action_dim must be positive, got {action_dim}"
            raise ValueError(msg)

        if action_horizon <= 0:
            msg = f"action_horizon must be positive, got {action_horizon}"
            raise ValueError(msg)

        if flow_sampling_steps <= 0:
            msg = f"flow_sampling_steps must be positive, got {flow_sampling_steps}"
            raise ValueError(msg)

        self._validate_sampling_method(flow_sampling_method)

        dim_model = dim_model or condition_dim
        if dim_model % 2 != 0:
            msg = f"dim_model must be even for sinusoidal time embeddings, got {dim_model}"
            raise ValueError(msg)

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.flow_sampling_steps = flow_sampling_steps
        self.flow_sampling_method = flow_sampling_method

        self.condition_projection: nn.Module = (
            nn.Identity()
            if condition_dim == dim_model
            else nn.Linear(condition_dim, dim_model)
        )
        self.action_projection = nn.Linear(action_dim, dim_model)
        self.time_embedding = SinusoidalTimeEmbedding(
            dim_model, scale=time_embedding_scale, logit_scale=time_logit_scale
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.SiLU(),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.position_embedding = nn.Embedding(action_horizon, dim_model)
        self.decoder = AdaLNCrossAttentionDecoder(
            dim_model=dim_model,
            num_layers=num_layers,
            num_heads=num_heads,
            time_dim=dim_model,
            rope=rope,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            mlp_dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )
        self.output_projection = nn.Linear(dim_model, action_dim)

        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)

    @override
    def forward(
        self, *, condition_tokens: Tensor, noised_actions: Tensor, flow_time: Tensor
    ) -> Tensor:
        self._validate_inputs(
            condition_tokens=condition_tokens, noised_actions=noised_actions
        )

        condition_tokens = self.condition_projection(condition_tokens)
        action_tokens = self.action_projection(noised_actions)

        flow_time_1d = self._prepare_flow_time(
            flow_time=flow_time,
            batch_size=noised_actions.shape[0],
            dtype=noised_actions.dtype,
            device=noised_actions.device,
        )
        time_emb = self.time_mlp(self.time_embedding(flow_time_1d))

        position_indices = torch.arange(
            self.action_horizon, device=noised_actions.device
        )
        position_tokens = self.position_embedding(position_indices).to(
            dtype=action_tokens.dtype
        )
        x = action_tokens + position_tokens
        return self.output_projection(
            self.decoder(x=x, context=condition_tokens, time_emb=time_emb)
        )

    def sample(
        self,
        *,
        condition_tokens: Tensor,
        noise: Tensor | None = None,
        steps: int | None = None,
        method: FlowSamplingMethod | None = None,
    ) -> Tensor:
        # Used in inference
        steps = self.flow_sampling_steps if steps is None else steps
        if steps <= 0:
            msg = f"steps must be positive, got {steps}"
            raise ValueError(msg)

        method = self.flow_sampling_method if method is None else method
        self._validate_sampling_method(method)

        if condition_tokens.ndim != 3:  # noqa: PLR2004
            msg = (
                "condition_tokens must have shape (batch, sequence, dim), "
                f"got {tuple(condition_tokens.shape)}"
            )
            raise ValueError(msg)

        batch_size = condition_tokens.shape[0]
        if noise is None:
            x = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                dtype=condition_tokens.dtype,
                device=condition_tokens.device,
            )
        else:
            x = noise
            self._validate_noised_actions(noised_actions=x, batch_size=batch_size)

        dt = 1.0 / steps
        for step in range(steps):
            t = step * dt
            velocity = self._velocity_at(
                condition_tokens=condition_tokens, x=x, flow_time=t
            )

            # Higher-order ODE sampling is based on EDM's Heun sampler:
            # https://arxiv.org/abs/2206.00364
            match method:
                case "euler":
                    x = torch.add(x, velocity, alpha=dt)
                case "midpoint":
                    midpoint = torch.add(x, velocity, alpha=0.5 * dt)
                    midpoint_velocity = self._velocity_at(
                        condition_tokens=condition_tokens,
                        x=midpoint,
                        flow_time=t + 0.5 * dt,
                    )
                    x = torch.add(x, midpoint_velocity, alpha=dt)
                case "heun":
                    predictor = torch.add(x, velocity, alpha=dt)
                    corrected_velocity = self._velocity_at(
                        condition_tokens=condition_tokens, x=predictor, flow_time=t + dt
                    )
                    x += 0.5 * dt * (velocity + corrected_velocity)

        return x

    def _velocity_at(
        self, *, condition_tokens: Tensor, x: Tensor, flow_time: float
    ) -> Tensor:
        batch_size = condition_tokens.shape[0]
        flow_time_tensor = torch.full(
            (batch_size,), flow_time, dtype=x.dtype, device=x.device
        )
        return self(
            condition_tokens=condition_tokens,
            noised_actions=x,
            flow_time=flow_time_tensor,
        )

    def _validate_inputs(
        self, *, condition_tokens: Tensor, noised_actions: Tensor
    ) -> None:
        if condition_tokens.ndim != 3:  # noqa: PLR2004
            msg = (
                "condition_tokens must have shape (batch, sequence, dim), "
                f"got {tuple(condition_tokens.shape)}"
            )
            raise ValueError(msg)

        self._validate_noised_actions(
            noised_actions=noised_actions, batch_size=condition_tokens.shape[0]
        )

    def _validate_noised_actions(
        self, *, noised_actions: Tensor, batch_size: int
    ) -> None:
        expected = (batch_size, self.action_horizon, self.action_dim)
        if noised_actions.shape != expected:
            msg = (
                "noised_actions must have shape "
                f"{expected}, got {tuple(noised_actions.shape)}"
            )
            raise ValueError(msg)

    @staticmethod
    def _validate_sampling_method(method: str) -> None:
        if method not in {"euler", "midpoint", "heun"}:
            msg = (
                "flow sampling method must be one of "
                "('euler', 'midpoint', 'heun'), "
                f"got {method!r}"
            )
            raise ValueError(msg)

    @staticmethod
    def _prepare_flow_time(
        *, flow_time: Tensor, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        flow_time = flow_time.to(device=device, dtype=dtype)

        if flow_time.ndim == 0:
            flow_time = flow_time.expand(batch_size)
        elif flow_time.ndim == 1:
            if flow_time.shape[0] == 1:
                flow_time = flow_time.expand(batch_size)
            elif flow_time.shape[0] != batch_size:
                msg = (
                    "flow_time must be scalar or have one value per batch item, "
                    f"got {tuple(flow_time.shape)} for batch size {batch_size}"
                )
                raise ValueError(msg)
        else:
            flow_time = flow_time.reshape(batch_size, -1)
            if flow_time.shape[1] != 1:
                msg = (
                    "flow_time must be scalar or have one value per batch item, "
                    f"got {tuple(flow_time.shape)}"
                )
                raise ValueError(msg)
            flow_time = flow_time.squeeze(1)

        return flow_time  # (batch,)
