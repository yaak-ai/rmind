from functools import partial
from math import sqrt
from typing import TYPE_CHECKING, Any, override

import torch.nn.functional as F
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.modules.module import Module

from rmind.components.base import Modality, SummaryToken, TokenType
from rmind.components.episode import Episode
from rmind.components.mask import AttentionMask, FactorizedAttentionMask
from rmind.components.nn import default_weight_init_fn
from rmind.components.transformer.attention import MaskedSelfAttention
from rmind.components.transformer.config import EncoderPredictionConfig
from rmind.components.transformer.feed_forward import MLPGLU
from rmind.components.transformer.utils import run_layer_stack

if TYPE_CHECKING:
    from rmind.components.objectives.base import Prediction


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
        emb_norm: InstanceOf[nn.Module] | None = None,
        rope: InstanceOf[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedTransformerEncoderBlock(
                embedding_dim=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                resid_dropout=resid_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
                rope=rope,
            )
            for _ in range(num_layers)
        ])
        self.emb_norm: nn.Module | None = emb_norm

    @override
    def forward(
        self, *, src: Tensor, mask: FactorizedAttentionMask, flatten: bool = False
    ) -> Tensor:
        x = self.emb_norm(src) if self.emb_norm is not None else src
        out = run_layer_stack(
            self.layers,
            x,
            mask.spatial.as_torch_attn_mask(),
            mask.temporal.as_torch_attn_mask(),
            training=self.training,
        )
        return rearrange(out, "b t s d -> b (t s) d") if flatten else out

    @override
    def predict(  # ty:ignore[invalid-explicit-override]
        self,
        *,
        src: InstanceOf[Tensor],
        mask: InstanceOf[FactorizedAttentionMask],
        episode: InstanceOf[Episode] | None = None,
        config: EncoderPredictionConfig | None = None,
    ) -> TensorDict:
        # Harsi: Implement actual attention_rollout for factorized attention
        predictions: dict[str, Prediction] = {}

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

        attn_mask = mask.mask_tensor == mask.legend.DO_ATTEND
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


class FactorizedTransformerEncoderBlock(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        embedding_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 1,
        rope: Module | None = None,
    ) -> None:
        super().__init__()

        self.temporal_norm = nn.LayerNorm(embedding_dim)
        self.spatial_norm = nn.LayerNorm(embedding_dim)
        self.mlp_norm = nn.LayerNorm(embedding_dim)

        self.temporal_mha: MaskedSelfAttention = MaskedSelfAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            rope=rope,
            attn_dropout=attn_dropout,
        )

        self.spatial_mha: MaskedSelfAttention = MaskedSelfAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            rope=None,
            attn_dropout=attn_dropout,
        )

        self.resid_drop = nn.Dropout(resid_dropout, inplace=False)

        self.mlp = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            default_weight_init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_attention(  # noqa: PLR0913
        self,
        *,
        attention: MaskedSelfAttention,
        norm: nn.LayerNorm,
        x: Tensor,
        mask: Tensor,
        need_weights: bool,
        average_attn_weights: bool,
    ) -> tuple[Tensor, Tensor | None]:
        residual = x
        x_norm = norm(x)

        attn_output = attention(
            x_norm,
            mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        attn_value, attn_weights = (
            attn_output if isinstance(attn_output, tuple) else (attn_output, None)
        )

        return residual + self.resid_drop(attn_value), attn_weights

    @override
    def forward(
        self,
        x: Tensor,
        spatial_mask: Tensor,
        temporal_mask: Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> tuple[Tensor, Tensor | None] | Tensor:
        _, t, s, _ = x.shape

        # Temporal attention: each spatial slot attends over timesteps independently.
        x, _ = self._apply_attention(
            attention=self.temporal_mha,
            norm=self.temporal_norm,
            x=rearrange(x, "b t s d -> (b s) t d"),
            mask=temporal_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = rearrange(x, "(b s) t d -> b t s d", s=s)

        # Spatial attention: each timestep attends over within-step tokens independently.
        x, spatial_attention_weights = self._apply_attention(
            attention=self.spatial_mha,
            norm=self.spatial_norm,
            x=rearrange(x, "b t s d -> (b t) s d"),
            mask=spatial_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = rearrange(x, "(b t) s d -> b t s d", t=t)

        residual = x
        out = residual + self.mlp(self.mlp_norm(x))
        return (out, spatial_attention_weights) if need_weights else out
