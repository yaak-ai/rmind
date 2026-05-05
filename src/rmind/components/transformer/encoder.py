from functools import partial
from math import sqrt
from typing import TYPE_CHECKING, Any, override

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor, nn
from torch.nn.modules.module import Module

from rmind.components.base import Modality, SummaryToken, TokenType
from rmind.components.episode import Episode
from rmind.components.mask import AttentionMask
from rmind.components.objectives.base import Prediction
from rmind.components.transformer.attention import MaskedSelfAttention
from rmind.components.transformer.config import (
    AttentionRolloutPredictionConfig,
    EncoderPredictionConfig,
)
from rmind.components.transformer.feed_forward import MLPGLU
from rmind.components.transformer.utils import run_layer_stack

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
        rope: Module | None = None,
    ) -> None:
        super().__init__()

        # self.pre_norm and self.mha mimic f in
        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/reversible.py#L72
        self.pre_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)  # pre-norm
        self.attn = MaskedSelfAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            rope=rope,
        )

        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/multi_head_dispatch.py#L258
        self.resid_drop: nn.Dropout = nn.Dropout(resid_dropout, inplace=False)

        # self.post_norm and self.mlp mimic g in
        # https://github.com/facebookresearch/xformers/blob/v0.0.28.post2/xformers/components/reversible.py#L72

        self.post_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)  # ffn

        self.mlp: MLPGLU = MLPGLU(
            dim_model=embedding_dim,
            dropout=mlp_dropout,
            hidden_layer_multiplier=hidden_layer_multiplier,
        )

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
        if need_weights:
            attn_out, attn_weights = self.attn(
                x_norm,
                mask,
                need_weights=True,
                average_attn_weights=average_attn_weights,
            )
        else:
            attn_out = self.attn(x_norm, mask, need_weights=False)
            attn_weights = None
        x = residual + self.resid_drop(attn_out)

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
        emb_norm: InstanceOf[nn.Module] | None = None,
        rope: InstanceOf[nn.Module] | None = None,
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
                rope=rope,
            )
            for _ in range(num_layers)
        ])
        self.emb_norm: nn.Module | None = emb_norm
        # https://github.com/karpathy/nanoGPT/blob/master/model.py#L182
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(dim_model)

    @override
    def forward(self, *, src: Tensor, mask: Tensor) -> Tensor:
        x = self.emb_norm(src) if self.emb_norm is not None else src
        x = run_layer_stack(self.layers, x, mask, training=self.training)
        return self.layer_norm(x)

    @validate_call
    def compute_attention_rollout(
        self,
        *,
        src: InstanceOf[Tensor],
        mask: InstanceOf[AttentionMask],
        config: AttentionRolloutPredictionConfig,
    ) -> Tensor:
        fuse_heads: Callable[[Tensor], Tensor]
        match config.head_fusion:
            case "mean":
                fuse_heads = lambda x: x.mean(dim=1)  # noqa: E731
            case "max":
                fuse_heads = lambda x: x.max(dim=1).values  # noqa: E731
            case "min":
                fuse_heads = lambda x: x.min(dim=1).values  # noqa: E731

        _, s, _ = src.shape
        identity = torch.eye(s, s, device=src.device)
        attn_rollout = identity.clone()

        x = src
        for layer in self.layers:
            x, attn = layer(
                x, mask.mask_tensor, need_weights=True, average_attn_weights=False
            )
            attn_fused = fuse_heads(attn)
            attn_discarded = self._discard_attention(
                attn_fused, mask, config.discard_ratio
            )
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

        if config and config.attention_rollout:
            if episode is None:
                msg = "episode is required when requesting ATTENTION_ROLLOUT"
                raise ValueError(msg)

            rollout = self.compute_attention_rollout(
                src=src, mask=mask, config=config.attention_rollout
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
