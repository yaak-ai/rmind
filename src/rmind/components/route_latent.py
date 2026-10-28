from typing import final, override

import torch
from einops import rearrange
from pydantic import InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.mask import FactorizedAttentionMask, TorchAttentionMaskLegend
from rmind.components.objectives.base import PATCHES

WAYPOINTS = ("input", "context", "waypoints")


@final
class RouteWorldLatent(Module):
    """Route-guided per-timestep latent followed by a causal temporal world model.

        q_t = proj(z_q(waypoints_t))                         the route asks the image; z_q is
                                                             the frozen waypoint tokenizer latent
        E_t = latent(Q = latent queries + q_t, K = V = DINO patches_t)
        W_t = world(E_{<=t})                                  the factorized TransformerEncoder:
                                                             causal across time, full within a step

    Only the route (waypoints) shapes the latent: no speed and no actions enter it, so the
    heads reading W cannot recover the action history from it. Speed and actions are left
    to the objectives (targets / forward-dynamics input).
    """

    @validate_call
    def __init__(
        self,
        *,
        context_tokenizer: InstanceOf[Module],
        question_projection: InstanceOf[Module],
        latent: InstanceOf[Module],
        world: InstanceOf[Module],
        key_norm: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()
        # frozen: its residual-VQ codebook must not update during training
        self.context_tokenizer = context_tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.question_projection = question_projection
        self.latent = latent
        self.world = world
        self.key_norm: Module | None = key_norm

    @override
    def train(self, mode: bool = True) -> "RouteWorldLatent":
        super().train(mode)
        self.context_tokenizer.eval()
        return self

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        patches = episode.get(PATCHES)  # (b, t, p, d) raw DINO patches, per frame
        key = self.key_norm(patches) if self.key_norm is not None else patches
        waypoints = episode.get(WAYPOINTS)  # (b, t, n_wp, 2)
        with torch.no_grad():
            z_q = self.context_tokenizer(waypoints.float())  # (b, t, 1, d_tok)
        question = self.question_projection(z_q.to(key.dtype))  # (b, t, 1, d)
        query = episode.embeddings.get((Modality.UTILITY, "latent")) + question
        evidence = self.latent({"query": query, "key": key, "value": key})  # (b, t, n, d)
        _, t, n, _ = evidence.shape
        # factorized: full attention within a timestep, causal across timesteps
        mask = FactorizedAttentionMask.from_tensors(
            spatial_mask_tensor=torch.full(
                (n, n), TorchAttentionMaskLegend.DO_ATTEND, device=evidence.device
            ),
            temporal_mask_tensor=torch.ones(
                t, t, dtype=torch.bool, device=evidence.device
            ).triu(1),  # True = DO_NOT_ATTEND (future)
            legend=TorchAttentionMaskLegend,
        )
        world = self.world(src=evidence, mask=mask)  # (b, t * n, d)
        return rearrange(world, "b (t n) d -> b t n d", t=t)
