from typing import Self, final, override

from pydantic import BaseModel, ConfigDict, model_validator, validate_call
from torch import Tensor, nn

from rmind.components.transformer.feed_forward import MLPGLU
from rmind.components.transformer.utils import run_layer_stack


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
    def decode(self, input: Input) -> Tensor:
        """Decoded per-query embeddings, BEFORE `output_projection`: `(b, sq, d)`
        for 3D input, `(b, t, sq, d)` for 4D -- `d` is the decoder's `dim_model`.
        Exposed separately from `forward` so a downstream head can read the
        decoder's residual stream (e.g. `DrivoR.score_head`) without changing
        `forward`'s `Tensor` contract, which
        `rmind.components.objectives.forward_dynamics` relies on (it feeds head
        outputs straight into `tree_map`/`TensorDict`).
        """
        query = input.query
        context = input.context

        if query.ndim == 4:  # noqa: PLR2004
            b, t, sq, d = query.shape
            sc = context.shape[-2]
            decoded = self.decoder(
                query.reshape(b * t, sq, d), context.reshape(b * t, sc, d)
            )
            return decoded.reshape(b, t, sq, d)

        return self.decoder(query, context)

    @validate_call
    @override
    def forward(self, input: Input) -> Tensor:
        # NOTE: the 4D reshape now happens on `decoded` (whose last dim IS
        # `dim_model`) rather than on the projected output -- the previous
        # `output.reshape(b, t, sq, d)` silently assumed
        # `output_projection.out_features == dim_model`, true only for the single
        # existing 4D caller (control_transformer's foresight head).
        return self.output_projection(self.decode(input))
