from typing import Any, Literal, final, override

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from xformers.components import Activation, ResidualNormStyle, build_activation
from xformers.components.attention import ScaledDotProduct
from xformers.components.attention.core import scaled_query_key_softmax
from xformers.components.feedforward import register_feedforward
from xformers.components.feedforward.mlp import Feedforward, MlpConfig
from xformers.components.multi_head_dispatch import (
    _fold_heads,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    _split_heads,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
)
from xformers.components.reversible import ReversibleSequence
from xformers.factory.model_factory import (
    get_weight_init_fn,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
    xFormerWeightInit,
)


@final
@register_feedforward("MLPGLU", MlpConfig)
class MLPGLU(Feedforward):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: str,
        hidden_layer_multiplier: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        dim_mlp = hidden_layer_multiplier * dim_model
        self.l1 = nn.Linear(in_features=dim_model, out_features=dim_mlp * 2, bias=bias)
        self.a1 = build_activation(Activation(activation))
        self.d1 = nn.Dropout(dropout)
        self.l2 = nn.Linear(in_features=dim_mlp, out_features=dim_model, bias=bias)
        self.d2 = nn.Dropout(dropout)

    @override
    def forward(self, input: Tensor) -> Tensor:
        # FFN_GEGLU eq. 6, https://arxiv.org/pdf/2002.05202v1.pdf
        x = self.l1(input)
        xw, xv = x.chunk(2, dim=-1)
        geglu = self.a1(xw) * xv
        return self.l2(self.d1(geglu))


class xFormerEncoder(nn.Module):  # noqa: N801
    def __init__(
        self,
        *,
        config: xFormerEncoderConfig,
        weight_init: xFormerWeightInit = xFormerWeightInit.ViT,
        freeze: bool | None = None,
    ) -> None:
        super().__init__()

        if any((
            not config.reversible,
            config.residual_norm_style is ResidualNormStyle.DeepNorm,
            config.position_encoding_config is not None,
        )):
            raise NotImplementedError

        self.encoders: ReversibleSequence = ReversibleSequence(
            nn.ModuleList(
                nn.ModuleList(xFormerEncoderBlock.get_reversible_layer(config))
                for _ in range(config.num_layers)
            )
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(config.dim_model)

        init_fn = get_weight_init_fn(weight_init)
        for name, module in self.encoders.named_children():
            init_fn(module=module, name=name, gain=1.0)

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(self, src: Tensor, mask: Tensor) -> Tensor:
        x = torch.cat([src, src], dim=-1)
        x = self.encoders(x, att_mask=mask)
        x = torch.stack(x.chunk(2, dim=-1))
        x = x.mean(dim=0)

        return self.layer_norm(x)

    def compute_attention_rollout(
        self,
        src: Tensor,
        mask: Tensor,
        *,
        head_fusion: Literal["max", "min", "mean"] = "max",
        drop_ratio: float | None = None,
    ) -> Tensor:
        """[1] Quantifying Attention Flow in Transformers (https://arxiv.org/abs/2005.00928)
        [2] Exploring Explainability for Vision Transformers (https://jacobgil.github.io/deeplearning/vision-transformer-explainability).
        """
        b, s, _ = src.shape
        identity = torch.eye(s, s, device=mask.device)
        attn_rollout = repeat(identity, "s_from s_to -> b s_from s_to", b=b)

        seq = repeat(src, "b s d -> b s (n d)", n=2)
        for block in self.encoders.blocks:
            match (net := block.f.net).sublayer.attention:
                case ScaledDotProduct():
                    _, x = rearrange(seq, "b s (n d) -> n b s d", n=2)
                    x = net.norm(x)
                    q = net.sublayer.in_proj_container.q_proj(x)
                    k = net.sublayer.in_proj_container.k_proj(x)

                    # https://github.com/facebookresearch/xformers/blob/184b2808a5eed5f03ce381b8e7bd73f6188d8453/xformers/components/multi_head_dispatch.py#L236
                    reshape_fn = (
                        _split_heads
                        if net.sublayer.attention.requires_head_dimension
                        else _fold_heads
                    )

                    q = reshape_fn(
                        q, b, s, net.sublayer.num_heads, net.sublayer.dim_key_head
                    )
                    k = reshape_fn(
                        k, b, s, net.sublayer.num_heads, net.sublayer.dim_key_head
                    )

                    attn = scaled_query_key_softmax(q, k, att_mask=mask)

                    # unfold attention
                    attn = attn.view(b, net.sublayer.num_heads, s, s)

                    match head_fusion:
                        case "mean":
                            attn = attn.mean(axis=1)  # pyright: ignore[reportCallIssue]
                        case "max":
                            attn = attn.max(axis=1)[0]  # pyright: ignore[reportCallIssue]
                        case "min":
                            attn = attn.min(axis=1)[0]  # pyright: ignore[reportCallIssue]

                    # TODO: exclude certain indices from getting dropped?  # noqa: FIX002
                    if drop_ratio is not None:
                        attn_flat = rearrange(attn, "b s_from s_to -> (b s_from) s_to)")
                        drop_count = int(attn_flat.shape[-1] * drop_ratio)
                        drop_indices = attn_flat.topk(
                            k=drop_count, dim=-1, largest=False
                        ).indices
                        for batch_idx, batch_drop_indices in enumerate(drop_indices):
                            attn_flat[batch_idx, batch_drop_indices] = 0

                        attn = rearrange(
                            attn_flat,
                            "(b s_from) s_to -> b s_from s_to",
                            s_from=s,
                            s_to=s,
                        )

                    attn = (attn + identity) * 0.5
                    # [2] > We also have to normalize the rows, to keep the total attention flow 1.
                    attn /= attn.sum(dim=-1, keepdim=True)
                    attn_rollout = torch.matmul(attn, attn_rollout)

                case _:
                    raise NotImplementedError

            seq = block(seq, f_args={"att_mask": mask})

        return attn_rollout
