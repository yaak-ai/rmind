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
        self_attn: bool = True,
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

        # self-attn is a no-op mixing step for a single query token; skip it there
        self.use_self_attn = self_attn
        if self_attn:
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
    def forward(
        self, x: Tensor, key: Tensor, value: Tensor | None = None
    ) -> Tensor:
        residual = x
        x_norm = self.cross_attn_norm(x)
        cross_attn_out, _ = self.cross_attn(
            query=x_norm,
            key=key,
            value=key if value is None else value,
            need_weights=False,
        )
        x = residual + self.cross_attn_resid_drop(cross_attn_out)

        if self.use_self_attn:
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
        self_attn: bool = True,
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
                self_attn=self_attn,
            )
            for _ in range(num_layers)
        ])

    @override
    def forward(
        self, query: Tensor, key: Tensor, value: Tensor | None = None
    ) -> Tensor:
        return run_layer_stack(
            self.layers, query, key, value, training=self.training
        )


@final
class CrossAttentionDecoderHead(nn.Module):
    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        query: Tensor
        key: Tensor
        value: Tensor | None = None

        @model_validator(mode="after")
        def _validate_shapes(self) -> Self:
            if self.query.ndim != self.key.ndim or self.query.ndim not in {3, 4}:
                msg = (
                    "query/key must both be 3D or 4D with matching ndim, "
                    f"got query={self.query.ndim}D, key={self.key.ndim}D"
                )
                raise ValueError(msg)
            return self

    def __init__(
        self, decoder: CrossAttentionDecoder, output_projection: nn.Linear
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.output_projection = output_projection

    @override
    def forward(self, input: Input | dict[str, Tensor]) -> Tensor:
        if isinstance(input, dict):
            query, key = input["query"], input["key"]
            value = input.get("value")
        else:
            query, key, value = input.query, input.key, input.value

        if query.ndim == 4:  # noqa: PLR2004
            b, t, sq, d = query.shape

            def flatten(x: Tensor | None) -> Tensor | None:
                return None if x is None else x.reshape(b * t, x.shape[2], x.shape[3])

            decoded = self.decoder(
                query.reshape(b * t, sq, d), flatten(key), flatten(value)
            )
            output = self.output_projection(decoded)

            return output.reshape(b, t, sq, -1)

        decoded = self.decoder(query, key, value)
        return self.output_projection(decoded)
