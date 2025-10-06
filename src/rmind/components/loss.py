from collections.abc import Callable
from typing import Any, final, override

import torch
import torch.nn.functional as F
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


@final
class SoftCLIPLoss(Module):
    def __init__(
        self,
        T: int = 6,  # noqa: N803
        N: int = 180,  # noqa: N803
        temp_pred: float = 0.1,
        temp_soft: float = 0.01,
        reduction: str | None = None,
    ) -> None:
        super().__init__()
        self.temp_pred = temp_pred
        self.temp_soft = temp_soft
        self.reduction = reduction
        self.T = T
        self.N = N

    @override
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: PLR0914
        """
        input, target: [B * T * S, D]
        """
        _, D = input.shape  # noqa: N806
        input = input.view(-1, self.T, self.N, D)
        target = target.view(-1, self.T, self.N, D)
        B, T, S, _ = input.shape  # noqa: N806

        input = F.normalize(input.view(B, T, S, D), dim=-1)  # [B, T, S, D]
        target = F.normalize(target.view(B, T, S, D), dim=-1)  # [B, T, S, D]

        # [B, T, S, S] similarities
        logits = torch.einsum("btsd,btud->btsu", input, target) / self.temp_pred
        log_probs = F.log_softmax(logits, dim=-1)  # along last dim

        sim_tt = torch.einsum("btsd,btud->btsu", target, target) / self.temp_soft
        soft_targets = F.softmax(sim_tt, dim=-1).detach()

        loss_i = -(soft_targets * log_probs).sum(dim=-1).mean()

        # Symmetric direction
        logits_t = torch.einsum("btsd,btud->btsu", target, input) / self.temp_pred
        log_probs_t = F.log_softmax(logits_t, dim=-1)

        sim_pp = torch.einsum("btsd,btud->btsu", input, input) / self.temp_soft
        soft_targets_t = F.softmax(sim_pp, dim=-1).detach()

        loss_t = -(soft_targets_t * log_probs_t).sum(dim=-1).mean()

        return (loss_i + loss_t) / 2
