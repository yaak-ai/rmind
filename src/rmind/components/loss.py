from collections.abc import Callable
from typing import Any, final, override

import torch
import torch.nn.functional as F
import torchmetrics as tm
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


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


class LogitBiasMixin:
    _logit_bias: Tensor | None  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def logit_bias(self) -> Tensor | None:
        return self._logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Tensor | None) -> None:
        match value:
            case Tensor():
                if hasattr(self, "_logit_bias"):
                    del self._logit_bias

                self.register_buffer("_logit_bias", value, persistent=False)  # pyright: ignore[reportAttributeAccessIssue]

            case None:
                self._logit_bias = None


class LogitBiasFocalLoss(LogitBiasMixin, FocalLoss):
    def __init__(self, *, logit_bias: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__(gamma=gamma)

        self._logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # pyright: ignore[reportOperatorIssue]


class LogitBiasCrossEntropyLoss(LogitBiasMixin, CrossEntropyLoss):
    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self._logit_bias: Tensor | None = logit_bias

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
@final
class GramAnchoringObjective(Module):
    def __init__(self, *args: Any, weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.weight: float = weight
        self.mse_loss = torch.nn.MSELoss()

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Don't update target
        target = target.detch()  # pyright: ignore[reportAttributeAccessIssue]

        sim_input = tm.pairwise_cosine_similarity(input, input)  # pyright: ignore[reportAttributeAccessIssue]
        sim_target = tm.pairwise_cosine_similarity(target, target)  # pyright: ignore[reportAttributeAccessIssue]

        return self.weight * self.mse_loss(sim_input, sim_target)
