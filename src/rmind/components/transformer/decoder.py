from typing import final, override

from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from rmind.components.transformer.mlp import MLPGLU


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()

        self.norm1: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout1: nn.Dropout = nn.Dropout(resid_dropout)

        self.norm2: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout2: nn.Dropout = nn.Dropout(resid_dropout)

        self.norm3: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.mlp: MLPGLU = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            activation="gelu",
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

    @override
    def forward(
        self,
        x: Tensor,  # query
        context: Tensor,  # key/value
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=tgt_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(attn_out)

        residual = x
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(
            query=x_norm,
            key=context,
            value=context,
            attn_mask=memory_mask,
            need_weights=False,
        )
        x = residual + self.dropout2(attn_out)

        residual = x
        x_norm = self.norm3(x)
        mlp_out = self.mlp(x_norm)
        return residual + mlp_out


class TransformerDecoder(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
        freeze: bool | None = None,  # noqa: FBT001
    ) -> None:
        super().__init__()

        self.layers: nn.ModuleList = nn.ModuleList([
            TransformerDecoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                mlp_dropout=mlp_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])

        self.norm: nn.LayerNorm = nn.LayerNorm(dim_model)

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(
        self,
        x: Tensor,
        context: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: query
            context: key/value
        """

        if self.training:

            def run_layer(
                layer: TransformerDecoderBlock, inp: Tensor, ctx: Tensor
            ) -> Tensor:
                return checkpoint(
                    layer, inp, ctx, tgt_mask, memory_mask, use_reentrant=False
                )

        else:

            def run_layer(
                layer: TransformerDecoderBlock, inp: Tensor, ctx: Tensor
            ) -> Tensor:
                return layer(inp, ctx, tgt_mask, memory_mask)

        for layer in self.layers:
            x = run_layer(layer, x, context)

        return self.norm(x)


@final
class TransformerDecoderWithProjectors(nn.Module):
    def __init__(
        self,
        img_embedding_dim: int,
        encoder_embedding_dim: int,
        decoder: TransformerDecoder,
    ) -> None:
        super().__init__()
        self.pre_projector = nn.Linear(encoder_embedding_dim, img_embedding_dim)
        self.decoder = decoder
        self.post_projector = nn.Linear(img_embedding_dim, img_embedding_dim)

    @override
    def forward(
        self,
        query: Tensor,
        context: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        query = self.pre_projector(query)
        query = rearrange(query, "b t s d -> (b t) s d")
        context = self.pre_projector(context)
        context = rearrange(context, "b t s d -> (b t) s d")
        x = self.decoder(query, context, tgt_mask, memory_mask)
        return self.post_projector(x)
