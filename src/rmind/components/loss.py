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
        self,
        *args: Any,
        logit_bias: Tensor | None = None,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # NOTE: `weight` is captured here (keyword-only) and stored as
        # `self.loss_weight` -- it is a scalar per-head loss multiplier, NOT the
        # per-class weight Tensor of `CrossEntropyLoss`. It is deliberately NOT
        # forwarded to `super().__init__`. The name `self.weight` is also avoided
        # because `_WeightedLoss` registers a `weight` buffer (assigning a float
        # there raises).
        super().__init__(*args, **kwargs)

        self.logit_bias: Tensor | None = logit_bias
        self.loss_weight: float = weight

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    def __init__(
        self,
        *args: Any,
        # NOTE: use torch.ones_like to get vanilla MSE
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        # scalar per-head loss multiplier (default 1.0 => no-op). Stored as
        # `self.loss_weight` to mirror `LogitBiasCrossEntropyLoss`.
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function
        self.loss_weight: float = weight

    @override
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:  # ty:ignore[invalid-method-override]
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return self.loss_weight * super().forward(input=mean, target=target, var=var)


class BetaNLLLoss(GaussianNLLLoss):
    """β-NLL loss (Seitzer et al. 2022, https://arxiv.org/abs/2203.09168).

    Identical to :class:`GaussianNLLLoss` but each sample's NLL is multiplied by
    ``detach(var) ** beta``. This down-weights the variance's effect on the mean
    gradient, interpolating between vanilla Gaussian NLL (``beta=0``) and an
    MSE-like objective (``beta=1``), which avoids the variance-collapse /
    overconfidence failure mode of heteroscedastic NLL.
    """

    def __init__(self, *args: Any, beta: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.beta: float = beta

    @override
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        loss = F.gaussian_nll_loss(
            mean, target, var, full=self.full, eps=self.eps, reduction="none"
        )
        loss = var.detach() ** self.beta * loss

        match self.reduction:
            case "mean":
                return self.loss_weight * loss.mean()
            case "sum":
                return self.loss_weight * loss.sum()
            case _:
                return self.loss_weight * loss


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
