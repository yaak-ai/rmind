from typing import override

from einops import rearrange
from pydantic import InstanceOf, validate_call
from torch import Tensor, nn
from torch.nn.modules.module import Module

from rmind.components.mask import FactorizedAttentionMask
from rmind.components.nn import default_weight_init_fn
from rmind.components.transformer.attention import MaskedSelfAttention
from rmind.components.transformer.feed_forward import MLPGLU
from rmind.components.transformer.utils import run_layer_stack


class TransformerEncoder(nn.Module):
    @validate_call
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        emb_norm: InstanceOf[nn.Module] | None = None,
        rope: InstanceOf[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedTransformerEncoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                resid_dropout=resid_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
                rope=rope,
            )
            for _ in range(num_layers)
        ])
        self.emb_norm: nn.Module | None = emb_norm

    @override
    def forward(self, *, src: Tensor, mask: FactorizedAttentionMask) -> Tensor:
        x = self.emb_norm(src) if self.emb_norm is not None else src
        out = run_layer_stack(
            self.layers,
            x,
            mask.spatial.mask_tensor,
            mask.temporal.mask_tensor,
            training=self.training,
        )
        return rearrange(out, "b t s d -> b (t s) d")


class FactorizedTransformerEncoderBlock(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        rope: Module | None = None,
    ) -> None:
        super().__init__()

        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.mlp_norm = nn.LayerNorm(embedding_dim)

        self.temporal_mha: MaskedSelfAttention = MaskedSelfAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            rope=rope,
            attn_dropout=attn_dropout,
        )

        self.spatial_mha: MaskedSelfAttention = MaskedSelfAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            rope=None,
            attn_dropout=attn_dropout,
        )

        self.resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.mlp = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            default_weight_init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_attention(
        self,
        *,
        attention: MaskedSelfAttention,
        norm: nn.LayerNorm,
        x: Tensor,
        mask: Tensor,
    ) -> Tensor:
        return x + self.resid_drop(attention(norm(x), mask))

    @override
    def forward(self, x: Tensor, spatial_mask: Tensor, temporal_mask: Tensor) -> Tensor:
        _, t, s, _ = x.shape

        # Temporal attention: each spatial slot attends over timesteps independently.
        x = self._apply_attention(
            attention=self.temporal_mha,
            norm=self.temporal_norm,
            x=rearrange(x, "b t s d -> (b s) t d"),
            mask=temporal_mask,
        )
        x = rearrange(x, "(b s) t d -> b t s d", s=s)

        # Spatial attention: each timestep attends over within-step tokens independently.
        x = self._apply_attention(
            attention=self.spatial_mha,
            norm=self.spatial_norm,
            x=rearrange(x, "b t s d -> (b t) s d"),
            mask=spatial_mask,
        )
        x = rearrange(x, "(b t) s d -> b t s d", t=t)

        return x + self.mlp(self.mlp_norm(x))
