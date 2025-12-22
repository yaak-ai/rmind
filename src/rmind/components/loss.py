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
        return super().forward(input + self.logit_bias, target)  # pyright: ignore[reportOperatorIssue]


class LogitBiasCrossEntropyLoss(CrossEntropyLoss, HasLogitBias):
    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # pyright: ignore[reportOperatorIssue]


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
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return super().forward(input=mean, target=target, var=var)


# https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
class GramAnchoringObjective(Module):
    def __init__(
        self,
        *args: Any,
        patches: int | None = None,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_sim: float = weight_sim
        self.weight_gram: float = weight_gram
        if weight_gram > 0 and patches is None:
            msg = "patches must be provided if weight_gram > 0"
            raise ValueError(msg)
        self.patches: int = patches  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.detach()

        # (b t p) d
        input = F.normalize(input, dim=-1)
        target = F.normalize(target, dim=-1)

        sim_loss = (1.0 - (input * target).sum(dim=-1)).mean()

        if self.weight_gram <= 0:
            return self.weight_sim * sim_loss

        input_view = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        target_view = rearrange(target, "(bt p) d -> bt p d", p=self.patches)

        # (b t) p p
        gram_pred = torch.matmul(input_view, input_view.transpose(1, 2))
        gram_gt = torch.matmul(target_view, target_view.transpose(1, 2))

        gram_loss = F.mse_loss(gram_pred, gram_gt)

        return self.weight_sim * sim_loss + self.weight_gram * gram_loss
