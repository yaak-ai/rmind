from typing import override

import torch
from torch import Tensor, nn

from rmind.components.drivor.ego_state import EgoStateEncoder
from rmind.components.transformer import (
    CrossAttentionDecoder,
    CrossAttentionDecoderHead,
)


class TrajectoryDecoderHead(nn.Module):
    """DETR-style multi-hypothesis trajectory decoder (DrivoR, arXiv:2601.05083).

    `num_queries` learnable trajectory queries, offset by the encoded ego
    state, cross-attend to the compressed per-camera register tokens
    (`context`, see `rmind.components.drivor.backbone.RegisterViTBackbone`)
    via `decoder`, then a linear head decodes `num_poses` `(x, y, theta)`
    poses per candidate trajectory.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: CrossAttentionDecoder,
        ego_state_encoder: EgoStateEncoder,
        num_queries: int = 64,
        dim_model: int = 256,
        num_poses: int = 10,
        pose_dims: int = 3,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.empty(1, num_queries, dim_model))
        # paper: N(0, 1e-6); ambiguous variance-vs-std, treated as tunable (see plan risks)
        nn.init.normal_(self.query, std=1e-3)

        self.ego_state_encoder = ego_state_encoder
        self.head = CrossAttentionDecoderHead(
            decoder=decoder,
            output_projection=nn.Linear(dim_model, num_poses * pose_dims),
        )
        self.num_poses = num_poses
        self.pose_dims = pose_dims

    def forward_features(
        self,
        *,
        context: Tensor,
        ego_continuous: Tensor,
        ego_turn_signal: Tensor,
        ego_route_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """`(poses (b, Q, num_poses, pose_dims), decoded (b, Q, dim_model))` -- the
        per-query embeddings alongside the poses they were projected from (free:
        they are computed either way), so a scorer can rank candidates (see
        `rmind.models.drivor.DrivoR._predict_scores`).

        `decoded` is the decoder's RAW residual stream: `CrossAttentionDecoder`
        applies no final LayerNorm (`CrossAttentionDecoderBlock.forward` ends on a
        residual add), so a consumer must normalize it itself.
        """
        b = context.shape[0]
        ego_embed = self.ego_state_encoder(
            continuous=ego_continuous,
            turn_signal=ego_turn_signal,
            route_embedding=ego_route_embedding,
        )
        queries = self.query.expand(b, -1, -1) + ego_embed.unsqueeze(1)
        decoded = self.head.decode(
            CrossAttentionDecoderHead.Input(query=queries, context=context)
        )
        poses = self.head.output_projection(decoded).reshape(
            b, -1, self.num_poses, self.pose_dims
        )
        return poses, decoded

    @override
    def forward(
        self,
        *,
        context: Tensor,
        ego_continuous: Tensor,
        ego_turn_signal: Tensor,
        ego_route_embedding: Tensor,
    ) -> Tensor:
        poses, _ = self.forward_features(
            context=context,
            ego_continuous=ego_continuous,
            ego_turn_signal=ego_turn_signal,
            ego_route_embedding=ego_route_embedding,
        )
        return poses
