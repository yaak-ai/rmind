from collections.abc import Callable
from typing import Any, Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange
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
    def forward(
        self, input: Tensor, target: Tensor, _weight: Tensor | None = None
    ) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    def __init__(
        self,
        *args: Any,
        # NOTE: use torch.ones_like to get vanilla MSE
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        log_var_clamp: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function
        self.log_var_clamp: tuple[float, float] | None = log_var_clamp

    @override
    def forward(
        self, input: Tensor, target: Tensor, weight: Tensor | None = None
    ) -> Tensor:  # ty:ignore[invalid-method-override]
        mean, log_var = input[..., 0], input[..., 1]
        if self.log_var_clamp is not None:
            log_var = log_var.clamp(*self.log_var_clamp)
        var = self.var_pos_function(log_var)

        if weight is not None:
            per_sample = F.gaussian_nll_loss(
                mean, target, var, full=self.full, eps=self.eps, reduction="none"
            )
            return (weight * per_sample).mean()

        return super().forward(input=mean, target=target, var=var)


class ActivityWeightedGaussianNLLLoss(GaussianNLLLoss):
    """GaussianNLLLoss with per-sample weights proportional to target magnitude.

    Samples where the control is actively engaged (high brake, large steer, etc.)
    receive higher gradient weight, counteracting zero-inflation in the dataset.
    """

    def __init__(self, *args: Any, activity_weight: float = 5.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.activity_weight = activity_weight

    @override
    def forward(
        self, input: Tensor, target: Tensor, weight: Tensor | None = None
    ) -> Tensor:
        if weight is None:
            weight = target.abs()
        weights = 1.0 + self.activity_weight * weight
        weights /= weights.mean()
        return super().forward(input, target, weight=weights)


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
