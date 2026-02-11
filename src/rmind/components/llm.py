from typing import TYPE_CHECKING, Any, Literal, final, override

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, InstanceOf, NonNegativeFloat, validate_call
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint

from rmind.components.mask import AttentionMask
from rmind.components.nn import default_weight_init_fn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "MLPGLU",
    "CrossAttentionDecoder",
    "CrossAttentionDecoderBlock",
    "CrossAttentionDecoderHead",
    "TransformerEncoder",
    "TransformerEncoderBlock",
]


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.B(self.A(x))


class LoRAMultiheadAttention(nn.Module):
    def __init__(
        self, base_mha: nn.MultiheadAttention, rank: int, alpha: float | None = None
    ) -> None:
        super().__init__()
        self.embed_dim = base_mha.embed_dim
        self.num_heads = base_mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_dropout = base_mha.dropout
        self.lora_scale = (alpha / rank) if alpha is not None else 1.0

        self.in_proj_weight = base_mha.in_proj_weight
        self.in_proj_bias = base_mha.in_proj_bias
        self.out_proj = base_mha.out_proj

        self.lora_q = LoRALinear(self.embed_dim, self.embed_dim, rank)
        self.lora_v = LoRALinear(self.embed_dim, self.embed_dim, rank)

    @override
    def forward(  # noqa: PLR0914
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        **_kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        E = self.embed_dim  # noqa: N806
        w_q, w_k, w_v = (
            self.in_proj_weight[:E],
            self.in_proj_weight[E : 2 * E],
            self.in_proj_weight[2 * E :],
        )
        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = (
                self.in_proj_bias[:E],
                self.in_proj_bias[E : 2 * E],
                self.in_proj_bias[2 * E :],
            )

        q = F.linear(query, w_q, b_q) + self.lora_q(query) * self.lora_scale
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v) + self.lora_v(value) * self.lora_scale

        B, S_q, _ = q.shape  # noqa: N806
        S_k = k.shape[1]  # noqa: N806

        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_mask is not None and attn_mask.ndim == 3:  # noqa: PLR2004
            attn_mask = attn_mask.view(B, self.num_heads, S_q, S_k)

        if need_weights:
            scale = self.head_dim**-0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
                else:
                    attn_weights = attn_weights + attn_mask  # noqa: PLR6104
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = F.dropout(
                attn_probs, p=self.attn_dropout, training=self.training
            )
            attn_output = attn_probs @ v
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, E)
            attn_output = self.out_proj(attn_output)
            if average_attn_weights:
                attn_probs = attn_probs.mean(dim=1)
            return attn_output, attn_probs

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = ~attn_mask
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, E)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


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
        freeze: bool | None = None,  # noqa: FBT001
        lora_rank: int | None = None,
        lora_num_layers: int | None = None,
        lora_alpha: float | None = None,
        emb_norm: InstanceOf[nn.Module] | None = None,
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
        self.emb_norm: nn.Module | None = emb_norm
        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L182
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(dim_model)

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)

        if lora_rank is not None and lora_num_layers is not None:
            self._apply_lora(lora_rank, lora_num_layers, lora_alpha)

    def _apply_lora(
        self, rank: int, num_layers: int, alpha: float | None = None
    ) -> None:
        for block in self.layers[-num_layers:]:  # ty:ignore[not-iterable]
            block.mha = LoRAMultiheadAttention(block.mha, rank, alpha)

    @override
    def forward(self, *, src: Tensor, mask: Tensor) -> Tensor:
        x = self.emb_norm(src) if self.emb_norm is not None else src

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
        freeze: bool | None = None,  # noqa: FBT001
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
        self.layer_norm = nn.LayerNorm(dim_model)

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        if self.training:

            def run_layer(
                layer: Module, layer_input: Tensor, layer_context: Tensor
            ) -> Any:
                return checkpoint(
                    layer, layer_input, layer_context, use_reentrant=False
                )

        else:

            def run_layer(
                layer: Module, layer_input: Tensor, layer_context: Tensor
            ) -> Any:
                return layer(layer_input, layer_context)

        for layer in self.layers:
            x = run_layer(layer, x, context)

        return self.layer_norm(x)


@final
class CrossAttentionDecoderHead(nn.Module):
    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        query: Tensor
        context: Tensor

    def __init__(
        self, decoder: CrossAttentionDecoder, output_projection: nn.Linear
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.output_projection = output_projection

    @staticmethod
    def _validate_shapes(query: Tensor, context: Tensor) -> None:
        if query.ndim != context.ndim or query.ndim not in {3, 4}:
            msg = (
                "query/context must both be 3D or 4D with matching ndim, "
                f"got query={query.ndim}D, context={context.ndim}D"
            )
            raise ValueError(msg)

    @validate_call
    @override
    def forward(self, input: Input) -> Tensor:
        query = input.query
        context = input.context

        self._validate_shapes(query, context)

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
