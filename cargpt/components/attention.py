import logging
from dataclasses import dataclass
from typing import Union

import torch
from xformers import ops as xops
from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)

logger = logging.getLogger("xformers")


@dataclass
class ScaledDotProductConfig(AttentionConfig):
    causal: bool | None
    seq_len: int | None
    to_seq_len: int | None


@register_attention("memory_efficient_scaled_dot_product", ScaledDotProductConfig)
class MemoryEfficientScaledDotProduct(Attention):
    r"""
    A memory efficient version of the
        Implementing the Scaled Dot-Product attention proposed in
        `Attention is all you need`_, Vaswani et al.

        .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
    """

    mask: AttentionMask | None

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,  # noqa: FBT001, FBT002
        seq_len: int | None = None,
        to_seq_len: int | None = None,
        *_args,
        **_kwargs,
    ):
        super().__init__()

        self.dropout = dropout
        self.causal = causal
        self.seq_len = seq_len

        if causal and seq_len is not None:
            self.mask = AttentionMask.make_causal(seq_len, to_seq_len)
        else:
            self.mask = None

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Union[AttentionMask, torch.Tensor],
        *_args,
        **_kwargs,
    ) -> torch.Tensor:
        r"""
        att_mask    A 2D or 3D mask which ignores attention at certain positions.

                    - If the mask is boolean, a value of True will keep the value,
                        while a value of False will mask the value.

                        Key padding masks (dimension: batch x sequence length) and attention masks
                        (dimension: sequence length x sequence length OR batch x sequence length x sequence length)
                        can be combined and passed in here. Method maybe_merge_masks provided in the utils can be
                        used for that merging.

                    - If the mask has the float type, then an additive mask is expected (masked values are -inf)

        """
        # I removed all the handlings of `att_mask = None`

        # Convenience, create an attention mask if a tensor was passed
        att_mask = att_mask.to(q.dtype)  # type: ignore
        if isinstance(att_mask, torch.Tensor):
            # By default we don't know of the causality, and a check would be expensive
            att_mask = (
                AttentionMask.from_bool(att_mask)
                if att_mask.dtype == torch.bool
                else AttentionMask(att_mask, is_causal=False)
            )

        # Handle a possibly deferred causal mask handling
        mask = self.mask
        if self.causal:
            mask = AttentionMask.make_causal(
                seq_len=q.shape[-2],
                to_seq_len=q.shape[-2],
                device=q.device,
                dtype=q.dtype,
            )

        # Merge the optional causal mask and the user-provided mask
        if mask is not None:
            mask = mask.to(dtype=q.dtype, device=q.device)

            att_mask = att_mask + mask

        # Try to handle a case where the sequence is smaller than the mask
        if q.shape[-2] == k.shape[-2] and q.shape[-2] < att_mask.shape[1]:
            att_mask = att_mask.make_crop(seq_len=q.shape[-2])

        att_mask = att_mask.values.expand(q.shape[0], -1, -1)

        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        return xops.memory_efficient_attention(
            query=q,
            key=k,
            value=v,
            p=self.dropout,
            attn_bias=att_mask,
        )
