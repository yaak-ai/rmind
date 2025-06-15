from typing import Any, Literal, final, override

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, attn_dropout=0.1, resid_dropout=0.1, ffn_dropout=0.1, activation="gelu", hidden_layer_multiplier=1):
        super().__init__()

        # f
        self.pre_norm = nn.LayerNorm(embedding_dim)  # pre-norm

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )

        self.resid_drop = nn.Dropout(resid_dropout, inplace=False)

        # g
        self.post_norm = nn.LayerNorm(embedding_dim)  # ffn

        self.mlp = MLPGLU(
                dim_model=embedding_dim,
                dropout=ffn_dropout,
                activation="gelu",
                hidden_layer_multiplier=hidden_layer_multiplier,
            )

    def forward(self, x, attn_mask=None):

        # f
        residual = x
        x_norm = self.pre_norm(x)
        mha, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.resid_drop(mha)

        # g
        residual = x
        y = residual + self.mlp(self.post_norm(x))

        return y


class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, num_layers, num_heads, attn_dropout=0.1, resid_dropout=0.1, ffn_dropout=0.1, hidden_layer_multiplier=1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                resid_dropout=resid_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier
            ) for _ in range(num_layers)
        ])
        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L182
        self.layer_norm = nn.LayerNorm(dim_model)


    def forward(self, x, attn_mask=None):

        for layer in self.layers:
            if self.training:
                x = checkpoint(layer, x, attn_mask)
            else:
                x = layer(x, attn_mask)

        return self.layer_norm(x)


class MLPGLU(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        hidden_layer_multiplier: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim_model
        self.l1 = nn.Linear(in_features=dim_model, out_features=dim_mlp * 2, bias=bias)
        self.a1 = nn.GELU()
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=bias)

    @override
    def forward(self, input: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.l1(input)
        xw, xv = x.chunk(2, dim=-1)
        geglu = self.a1(xw) * xv
        return self.l2(self.d1(geglu))
