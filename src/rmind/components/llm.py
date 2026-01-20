from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, final

from typing_extensions import override

import torch
from pydantic import InstanceOf, NonNegativeFloat, validate_call
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint

from rmind.components.mask import AttentionMask
from rmind.components.nn import default_weight_init_fn

if TYPE_CHECKING:
    from collections.abc import Callable


class KVCache(NamedTuple):
    """Key-Value cache for a single transformer layer.

    Stores the projected key and value tensors from previous positions
    to avoid recomputation during incremental inference.
    """

    key: Tensor  # [B, S_cached, D]
    value: Tensor  # [B, S_cached, D]


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

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

    def _compute_qkv(self, x_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, K, V projections using the MHA's internal weights.

        This allows us to access the projected K, V for caching while
        reusing the MHA's learned weights.
        """
        # nn.MultiheadAttention uses in_proj_weight when q, k, v have same dim
        # Shape: [3 * embed_dim, embed_dim]
        if self.mha._qkv_same_embed_dim:  # noqa: SLF001
            # Combined projection
            qkv = F.linear(x_norm, self.mha.in_proj_weight, self.mha.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Separate projections
            q = F.linear(
                x_norm,
                self.mha.q_proj_weight,
                self.mha.in_proj_bias[: self.embedding_dim]
                if self.mha.in_proj_bias is not None
                else None,
            )
            k = F.linear(
                x_norm,
                self.mha.k_proj_weight,
                self.mha.in_proj_bias[self.embedding_dim : 2 * self.embedding_dim]
                if self.mha.in_proj_bias is not None
                else None,
            )
            v = F.linear(
                x_norm,
                self.mha.v_proj_weight,
                self.mha.in_proj_bias[2 * self.embedding_dim :]
                if self.mha.in_proj_bias is not None
                else None,
            )
        return q, k, v

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute scaled dot-product attention with optional mask.

        Args:
            q: Query tensor [B, num_heads, S_q, head_dim]
            k: Key tensor [B, num_heads, S_kv, head_dim]
            v: Value tensor [B, num_heads, S_kv, head_dim]
            mask: Optional attention mask [S_q, S_kv] or [B, num_heads, S_q, S_kv]
            dropout_p: Dropout probability

        Returns:
            Tuple of (attention output, attention weights if requested)
        """
        # Use PyTorch's optimized implementation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p if self.training else 0.0,
            is_causal=False,  # We handle masking explicitly
        )
        return attn_output, None  # Weights not computed with SDPA

    def forward_with_cache(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        """Forward pass with optional KV caching for incremental inference.

        Args:
            x: Input tensor [B, S_new, D] - can be full sequence or just new tokens
            mask: Attention mask [S_q, S_kv] where S_kv = S_cached + S_new
            past_kv: Cached key/value from previous forward passes
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output tensor, updated KV cache if use_cache else None)
        """
        residual = x
        x_norm = self.pre_norm(x)

        # Compute Q, K, V for new positions
        q_new, k_new, v_new = self._compute_qkv(x_norm)

        # Concatenate with cached K, V if available
        if past_kv is not None:
            k = torch.cat([past_kv.key, k_new], dim=1)
            v = torch.cat([past_kv.value, v_new], dim=1)
        else:
            k = k_new
            v = v_new

        # Reshape for multi-head attention: [B, S, D] -> [B, num_heads, S, head_dim]
        batch_size = q_new.shape[0]
        q = q_new.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_mh = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_mh = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        dropout_p = self.mha.dropout if self.training else 0.0
        attn_output, _ = self._scaled_dot_product_attention(
            q, k_mh, v_mh, mask=mask, dropout_p=dropout_p
        )

        # Reshape back: [B, num_heads, S, head_dim] -> [B, S, D]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        )

        # Apply output projection
        attn_output = F.linear(
            attn_output, self.mha.out_proj.weight, self.mha.out_proj.bias
        )

        x = residual + self.resid_drop(attn_output)

        # MLP block
        residual = x
        mlp = self.mlp(self.post_norm(x))
        out = residual + mlp

        # Return updated cache if requested
        new_cache = KVCache(key=k, value=v) if use_cache else None

        return out, new_cache

    @override
    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, KVCache | None] | Tensor:
        """Forward pass with backward-compatible signature.

        When use_cache is True, uses the caching path. Otherwise uses the
        original nn.MultiheadAttention for full compatibility.

        Args:
            x: Input tensor [B, S, D]
            mask: Attention mask
            need_weights: Whether to return attention weights (incompatible with cache)
            average_attn_weights: Whether to average attention weights across heads
            past_kv: Optional cached key/value from previous pass
            use_cache: Whether to use/return KV cache

        Returns:
            - If use_cache: (output, KVCache | None)
            - If need_weights: (output, attention_weights)
            - Otherwise: output tensor
        """
        # Use caching path if requested or if we have past cache
        if use_cache or past_kv is not None:
            return self.forward_with_cache(
                x, mask, past_kv=past_kv, use_cache=use_cache
            )

        # Original path using nn.MultiheadAttention
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
        self.num_layers = num_layers
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

    def forward_with_cache(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        *,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, list[KVCache] | None]:
        """Forward pass with KV caching for incremental inference.

        Args:
            src: Input embeddings [B, S_new, D] - full sequence or new positions only
            mask: Attention mask. For incremental inference with cache, should be
                  [S_new, S_total] where S_total = S_cached + S_new
            past_key_values: List of KVCache, one per layer, from previous pass
            use_cache: Whether to compute and return updated KV cache

        Returns:
            Tuple of:
            - Output tensor [B, S_new, D]
            - List of KVCache per layer if use_cache, else None
        """
        x = src
        new_key_values: list[KVCache] = [] if use_cache else []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            x, new_kv = layer.forward_with_cache(
                x, mask, past_kv=past_kv, use_cache=use_cache
            )

            if use_cache and new_kv is not None:
                new_key_values.append(new_kv)

        x = self.layer_norm(x)

        return x, new_key_values if use_cache else None

    @override
    def forward(
        self,
        *,
        src: Tensor,
        mask: Tensor,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[KVCache] | None]:
        """Forward pass with optional KV caching.

        This method maintains backward compatibility. When use_cache is False
        and past_key_values is None, it behaves exactly as before.

        Args:
            src: Input embeddings [B, S, D]
            mask: Attention mask [S, S] or appropriate shape for caching
            past_key_values: Optional list of KVCache from previous forward
            use_cache: Whether to return KV cache for subsequent calls

        Returns:
            - If use_cache or past_key_values: tuple of (output, kv_cache)
            - Otherwise: just output tensor (backward compatible)
        """
        # Use caching path if requested
        if use_cache or past_key_values is not None:
            return self.forward_with_cache(
                src, mask, past_key_values=past_key_values, use_cache=use_cache
            )

        # Original path without caching
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
