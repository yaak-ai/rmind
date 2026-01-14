from typing import TYPE_CHECKING, Any, Literal, final

from typing_extensions import override

import torch
from pydantic import InstanceOf, NonNegativeFloat, validate_call
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint

from rmind.components.mask import AttentionMask
from rmind.components.nn import default_weight_init_fn

if TYPE_CHECKING:
    from collections.abc import Callable


class TransformerEncoderBlock(nn.Module):
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

        # self.pre_norm and self.mha mimic f in
        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/reversible.py#L72
        self.pre_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)  # pre-norm

        self.mha: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/multi_head_dispatch.py#L258
        self.resid_drop: nn.Dropout = nn.Dropout(resid_dropout, inplace=False)

        # self.post_norm and self.mlp mimic g in
        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/reversible.py#L72

        self.post_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)  # ffn

        self.mlp: MLPGLU = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            activation="gelu",
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

    @staticmethod
    def _init_weights(module: Module) -> None:
        match module:
            case nn.Linear():
                default_weight_init_fn(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            case _:
                pass

    @override
    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> tuple[Tensor, Tensor | None] | Tensor:
        # f
        residual = x
        x_norm = self.pre_norm(x)
        mha, attn_weights = self.mha.forward(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = residual + self.resid_drop(mha)

        # g
        residual = x
        mlp = self.mlp(self.post_norm(x))
        out = residual + mlp

        return (out, attn_weights) if need_weights else out


class TransformerEncoder(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        freeze: bool | None = None,  # noqa: FBT001
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                resid_dropout=resid_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])
        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L182
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(dim_model)

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(self, *, src: Tensor, mask: Tensor) -> Tensor:
        x = src

        if self.training:

            def run_layer(layer: Module, layer_input: Tensor, mask: Tensor) -> Any:
                return checkpoint(layer, layer_input, mask, use_reentrant=False)

        else:

            def run_layer(layer: Module, layer_input: Tensor, mask: Tensor) -> Any:
                return layer(layer_input, mask)

        for layer in self.layers:
            x = run_layer(layer, x, mask)

        return self.layer_norm(x)

    @validate_call
    def compute_attention_rollout(
        self,
        *,
        src: InstanceOf[Tensor],
        mask: InstanceOf[AttentionMask],
        head_fusion: Literal["mean", "max", "min"] = "mean",
        discard_ratio: NonNegativeFloat | None = None,
    ) -> Tensor:
        fuse_heads: Callable[[Tensor], Tensor]
        match head_fusion:
            case "mean":
                fuse_heads = lambda x: x.mean(axis=1)  # noqa: E731
            case "max":
                fuse_heads = lambda x: x.max(axis=1).values  # noqa: E731
            case "min":
                fuse_heads = lambda x: x.min(axis=1).values  # noqa: E731

        _, s, _ = src.shape
        identity = torch.eye(s, s, device=src.device)
        attn_rollout = identity.clone()

        x = src
        for layer in self.layers:
            x, attn = layer(x, mask.mask, need_weights=True, average_attn_weights=False)
            attn_fused = fuse_heads(attn)
            attn_discarded = self._discard_attention(attn_fused, mask, discard_ratio)
            attn_residual = (attn_discarded + identity) * 0.5
            attn_norm = attn_residual / attn_residual.sum(dim=-1, keepdim=True)
            attn_rollout = attn_norm @ attn_rollout

        return attn_rollout

    @staticmethod
    def _discard_attention(
        attn: Tensor, mask: AttentionMask, discard_ratio: float | None
    ) -> Tensor:
        """Set `discard_ratio` of non-masked-out values (per-row) in `attn` to zero."""
        if not discard_ratio:
            return attn

        attn_mask = mask.mask == mask.legend.DO_ATTEND.value
        discard_counts = (attn_mask.count_nonzero(dim=1) * discard_ratio).int().tolist()

        # NOTE: done per-row b/c masks and the k in topk may differ
        # NOTE: could likely be optimized for blocky masks
        for i, (row_mask, discard_count) in enumerate(
            zip(attn_mask, discard_counts, strict=True)
        ):
            row = attn[:, i]
            row_masked = row[:, row_mask]
            discard_indices = row_masked.topk(
                k=discard_count, dim=-1, largest=False
            ).indices
            row_masked_discarded = row_masked.scatter(
                dim=1, index=discard_indices, value=0.0
            )
            attn[:, i] = row.masked_scatter(row_mask, row_masked_discarded)

        return attn


@final
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
