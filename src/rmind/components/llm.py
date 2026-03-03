from functools import partial
from math import sqrt
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Self,
    final,
    override,
)

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    NonNegativeFloat,
    model_validator,
    validate_call,
)
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint

from rmind.components.base import Modality, SummaryToken, TokenType
from rmind.components.episode import Episode
from rmind.components.mask import AttentionMask
from rmind.components.nn import default_weight_init_fn
from rmind.components.objectives.base import Prediction

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


class AttentionRolloutPredictionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    head_fusion: Literal["mean", "max", "min"] = "max"
    discard_ratio: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.9


class EncoderPredictionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    attention_rollout: AttentionRolloutPredictionConfig | None = None


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

    @validate_call
    def predict(
        self,
        *,
        src: InstanceOf[Tensor],
        mask: InstanceOf[AttentionMask],
        episode: InstanceOf[Episode] | None = None,
        config: EncoderPredictionConfig | None = None,
    ) -> TensorDict:
        predictions: dict[str, Prediction] = {}

        if config and (cfg := config.attention_rollout) is not None:
            if episode is None:
                msg = "episode is required when requesting ATTENTION_ROLLOUT"
                raise ValueError(msg)

            rollout = self.compute_attention_rollout(
                src=src,
                mask=mask,
                head_fusion=cfg.head_fusion,
                discard_ratio=cfg.discard_ratio,
            )
            _, t = episode.input.batch_size
            predictions["attention_rollout"] = Prediction(
                value=self._attention_rollout_visualization(
                    episode=episode, attention_rollout=rollout
                ),
                timestep_indices=slice(t - 1, None),
            )

        return TensorDict(predictions).auto_batch_size_(1)  # ty:ignore[invalid-argument-type]

    @staticmethod
    def _attention_rollout_visualization(
        *, episode: Any, attention_rollout: Tensor
    ) -> dict[str, Tensor]:
        observation_keys = episode.timestep.get(TokenType.OBSERVATION).keys(
            include_nested=True, leaves_only=True
        )

        attention = (
            episode.index
            .parse(attention_rollout, dim=1)
            .select((Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY))[:, -1]
            .apply(
                lambda x: (
                    episode.index
                    .parse(x, dim=2)
                    .select(*observation_keys)
                    .squeeze(dim=1)
                )
            )
            .named_apply(
                partial(
                    TransformerEncoder._expand_attn_for_visualization,
                    input=episode.input,
                ),
                nested_keys=True,
            )
            .update({"input": episode.input.select(Modality.IMAGE)})
        )

        return {
            f"{k}": v
            for (k, v) in enumerate(attention.auto_batch_size_(2).split(1, dim=1))
        }

    @staticmethod
    def _expand_attn_for_visualization(
        path: tuple[str, ...], attn: Tensor, *, input: TensorDict
    ) -> Tensor:
        match path:
            case (*_, Modality.IMAGE, token_to):
                (_b, _t, hw_attn) = attn.shape
                (_b, _t, _c, h_img, w_img) = input.get_item_shape((
                    Modality.IMAGE,
                    token_to,
                ))
                attn = rearrange(
                    attn,
                    "... (h_attn w_attn) -> ... h_attn w_attn",
                    h_attn=int(sqrt(hw_attn * h_img / w_img)),
                )

                return F.interpolate(attn, size=(h_img, w_img))

            case _:
                return rearrange(attn, "b t d -> b t 1 d")

    @staticmethod
    def _discard_attention(
        attn: Tensor, mask: AttentionMask, discard_ratio: float | None
    ) -> Tensor:
        """Set `discard_ratio` of non-masked-out values (per-row) in `attn` to zero."""
        if not discard_ratio:
            return attn

        attn_mask = mask.mask == mask.legend.DO_ATTEND
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
    @override
    def forward(self, input: Input) -> Tensor:
        query = input.query
        context = input.context

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
