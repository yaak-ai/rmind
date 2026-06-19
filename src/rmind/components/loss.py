from typing import Annotated, Any, Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange
from pydantic import Field, validate_call
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


@runtime_checkable
class HasLogitBias(Protocol):
    logit_bias: Tensor | None


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf."""

    def __init__(self, *, gamma: float = 2.0) -> None:
        super().__init__()

        self.gamma: float = gamma

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)

        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


class LogitBiasFocalLoss(FocalLoss, HasLogitBias):
    def __init__(self, *, logit_bias: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__(gamma=gamma)

        self.logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class LogitBiasCrossEntropyLoss(CrossEntropyLoss, HasLogitBias):
    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class GramAnchoringLoss(Module):
    """
    Gram-based anchoring loss for feature matching.
    Based on DINOv3 implementation:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
    Uses target-driven within-frame patch uniqueness weights.
    """

    def __init__(
        self,
        *args: Any,
        patches: int,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_sim: float = weight_sim
        self.weight_gram: float = weight_gram
        self.patches: int = patches

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.detach()
        eps = 1e-6

        input_view = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        target_view = rearrange(target, "(bt p) d -> bt p d", p=self.patches)
        input_n = F.normalize(input_view, dim=-1)
        target_n = F.normalize(target_view, dim=-1)

        # Target-driven within-frame patch uniqueness weights.
        # Patches similar to many others in the same frame are downweighted.
        frame_sim = torch.einsum("bpd,bqd->bpq", target_n, target_n).clamp_min(0.0)
        eye = torch.eye(self.patches, dtype=torch.bool, device=target.device)
        weights = 1.0 / (
            frame_sim.masked_fill(eye, 0.0).sum(dim=-1) / (self.patches - 1) + eps
        )
        weights /= weights.sum(dim=1, keepdim=True) + eps

        patch_loss = F.mse_loss(input_view, target_view, reduction="none").mean(dim=-1)
        sim_loss = (weights * patch_loss).sum(dim=1).mean()

        if self.weight_gram <= 0:
            return self.weight_sim * sim_loss

        # Gram on L2-normed features weighted by patch uniqueness.
        gram_pred = torch.einsum("bpd,bqd->bpq", input_n, input_n)
        gram_gt = torch.einsum("bpd,bqd->bpq", target_n, target_n)
        pair_weights = torch.einsum("bp,bq->bpq", weights, weights)  # (bt, p, p)
        gram_loss = (pair_weights * (gram_pred - gram_gt).pow(2)).sum(dim=(1, 2)).mean()

        return self.weight_sim * sim_loss + self.weight_gram * gram_loss


class FlowMatchingLoss(Module):
    """Flow-matching velocity loss for the action policy.

    Velocity MSE between the predicted and target flow, plus an optional
    auxiliary within-chunk delta term (MSE on adjacent-slot deltas of the implied
    clean actions vs the target, weighted by chunk_delta_weight) that shapes the
    within-chunk trajectory rather than its DC level. The delta term needs the
    interpolant state (noised actions + flow-time) and >= 2 horizon slots (a diff
    over a length-1 axis is empty), so it is skipped below that.

    An optional per-chunk importance weight (e.g. LDS), broadcast over slots,
    multiplies both terms so they stay on a consistent scale. Returns the scalar
    training loss; with weight=None and chunk_delta_weight=0 that is plain
    velocity MSE.
    """

    @validate_call
    def __init__(
        self, *, chunk_delta_weight: Annotated[float, Field(ge=0)] = 0.0
    ) -> None:
        super().__init__()
        self.chunk_delta_weight = chunk_delta_weight

    @override
    def forward(
        self,
        predicted_flow: Tensor,
        target_flow: Tensor,
        *,
        noised_actions: Tensor,
        flow_time: Tensor,
        target_actions: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        if weight is None:
            flow_term = F.mse_loss(predicted_flow, target_flow)
        else:
            flow_term = (weight * (predicted_flow - target_flow).pow(2)).mean()

        # The within-chunk delta needs >= 2 horizon slots (diff over a length-1
        # axis is empty -> NaN); skip the term otherwise.
        if self.chunk_delta_weight <= 0 or noised_actions.shape[1] < 2:  # noqa: PLR2004
            return flow_term

        implied_actions = noised_actions + (1.0 - flow_time) * predicted_flow
        delta_pred = implied_actions.diff(dim=1)
        delta_target = target_actions.diff(dim=1)
        if weight is None:
            chunk_delta = F.mse_loss(delta_pred, delta_target)
        else:
            chunk_delta = (weight * (delta_pred - delta_target).pow(2)).mean()
        return flow_term + self.chunk_delta_weight * chunk_delta
