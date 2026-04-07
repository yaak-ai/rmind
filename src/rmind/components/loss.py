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
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    def __init__(
        self,
        *args: Any,
        # NOTE: use torch.ones_like to get vanilla MSE
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function

    @override
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:  # ty:ignore[invalid-method-override]
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return super().forward(input=mean, target=target, var=var)


# https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
class GramAnchoringObjective(Module):
    """
    Gram-based anchoring loss for feature matching.
    Based on DINOv3 implementation:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
    """

    def __init__(
        self,
        *args: Any,
        patches: int | None = None,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        **kwargs: Any,
    ) -> None:
        if weight_gram > 0 and patches is None:
            msg = "patches must be provided if weight_gram > 0"
            raise ValueError(msg)
        super().__init__(*args, **kwargs)
        self.weight_sim: float = weight_sim
        self.weight_gram: float = weight_gram
        self.patches: int | None = patches

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:  # noqa: PLR0914
        target = target.detach()
        eps = 1e-6

        input_n = F.normalize(input, dim=-1)
        target_n = F.normalize(target, dim=-1)

        input_view = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        input_n = rearrange(input_n, "(bt p) d -> bt p d", p=self.patches)
        target_view = rearrange(target, "(bt p) d -> bt p d", p=self.patches)
        target_n = rearrange(target_n, "(bt p) d -> bt p d", p=self.patches)

        # TF-IDF patch uniqueness weights from GT
        # TF: within-frame — patches similar to many others in the same frame are downweighted
        within_sim = torch.einsum("bpd,bqd->bpq", target_n, target_n)
        tf = 1.0 / (within_sim.mean(dim=-1) + eps)

        # IDF: cross-frame — patches common across frames in the batch are downweighted
        bt, p, d = target_n.shape
        all_patches = target_n.reshape(bt * p, d)
        cross_sim = torch.einsum("pd,qd->pq", all_patches, all_patches)
        idf = 1.0 / (cross_sim.mean(dim=-1).reshape(bt, p) + eps)

        weights = tf * idf
        weights /= weights.sum(dim=1, keepdim=True) + eps

        patch_loss = F.mse_loss(input_view, target_view, reduction="none").mean(dim=-1)
        sim_loss = (weights * patch_loss).sum(dim=1).mean()

        if self.weight_gram <= 0:
            return self.weight_sim * sim_loss

        # Gram on L2-normed features weighted by TF-IDF pair uniqueness
        gram_pred = torch.einsum("bpd,bqd->bpq", input_n, input_n)
        gram_gt = torch.einsum("bpd,bqd->bpq", target_n, target_n)
        pair_weights = torch.einsum("bp,bq->bpq", weights, weights)  # (bt, p, p)
        gram_loss = (pair_weights * (gram_pred - gram_gt).pow(2)).sum(dim=(1, 2)).mean()

        return self.weight_sim * sim_loss + self.weight_gram * gram_loss
