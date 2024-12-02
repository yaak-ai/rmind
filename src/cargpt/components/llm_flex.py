from typing import Literal, Optional, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import FlexAttention
from xformers.components import Activation, ResidualNormStyle, build_activation
from xformers.components.feedforward import register_feedforward
from xformers.components.feedforward.mlp import Feedforward, MlpConfig
from xformers.factory.model_factory import (
    get_weight_init_fn,
    xFormerEncoderConfig,
    xFormerWeightInit,
)


@register_feedforward("MLPGLU", MlpConfig)
class MLPGLU(Feedforward):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: str,
        hidden_layer_multiplier: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim_model
        self.l1 = nn.Linear(in_features=dim_model, out_features=dim_mlp * 2, bias=bias)
        self.a1 = build_activation(Activation(activation))
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=bias)
        self.d2 = nn.Dropout(dropout)

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.l1(inputs)
        xW, xV = x.chunk(2, dim=-1)
        geglu = self.a1(xW) * xV
        return self.l2(self.d1(geglu))


class xFormerEncoder(nn.Module):
    def __init__(
        self,
        *,
        config: xFormerEncoderConfig,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
        freeze: Optional[bool] = None,
    ):
        super().__init__()

        self.config = config
        self.encoders = nn.ModuleList([
            FlexAttentionEncoderBlock(config) for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.dim_model)

        # Weight initialization
        init_fn = get_weight_init_fn(weight_init)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                init_fn(module=module, name=name, gain=1.0)

        if freeze is not None:
            self.requires_grad_(not freeze)

    def forward(
        self,
        src: Tensor,  # [batch_size, sequence_length, dim]
        mask: Optional[Tensor] = None,  # [sequence_length, sequence_length]
    ) -> Tensor:
        x = src
        for encoder_block in self.encoders:
            x = encoder_block(x, mask)

        return self.layer_norm(x)

    def compute_attention_rollout(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        *,
        head_fusion: Literal["max", "mean", "min"] = "max",
        drop_ratio: Optional[float] = None,
    ) -> Tensor:
        """
        Compute attention rollout visualization inspired by:
        [1] Quantifying Attention Flow in Transformers
        [2] Exploring Explainability for Vision Transformers
        """
        batch_size, seq_len, _ = src.shape
        device = src.device

        # Initialize rollout matrix
        rollout = (
            torch.eye(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        x = src
        for encoder_block in self.encoders:
            # Capture attention weights during forward pass
            x, attention_weights = encoder_block(x, mask, capture_attention=True)

            # Process attention weights
            if head_fusion == "max":
                attention_weights = attention_weights.max(dim=1)[0]
            elif head_fusion == "mean":
                attention_weights = attention_weights.mean(dim=1)
            elif head_fusion == "min":
                attention_weights = attention_weights.min(dim=1)

            # Optional: Drop low-importance attention connections
            if drop_ratio is not None:
                flat_attn = attention_weights.view(batch_size, -1)
                drop_count = int(flat_attn.shape[-1] * drop_ratio)
                _, low_indices = torch.topk(
                    flat_attn, drop_count, largest=False, dim=-1
                )

                for b in range(batch_size):
                    attention_weights[b].view(-1)[low_indices[b]] = 0

            # Normalize attention
            attention_weights = F.normalize(attention_weights, p=1, dim=-1)

            # Update rollout matrix
            rollout = torch.bmm(attention_weights, rollout)

        return rollout


class FlexAttentionEncoderBlock(nn.Module):
    def __init__(self, config: xFormerEncoderConfig):
        super().__init__()

        self.flex_attention = FlexAttention(
            embed_dim=config.dim_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim_model, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.dim_model),
        )

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, capture_attention: bool = False
    ) -> Tensor:
        # Self-attention with optional attention weight capture
        attn_output = self.flex_attention(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            key_padding_mask=mask,
        )

        if capture_attention:
            # Return both output and attention weights
            attn_output, attention_weights = attn_output
            x = x + attn_output
            return x, attention_weights

        x = x + attn_output

        # Feed-forward
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output

        return x
