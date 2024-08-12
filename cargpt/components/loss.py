from collections.abc import Callable
from enum import StrEnum, auto

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from typing_extensions import override


class LossType(StrEnum):
    REGRESSION = auto()
    CLASSIFICATION = auto()


class GenericRegressionLoss:
    loss_type = LossType.REGRESSION


class GenericClassificationLoss:
    loss_type = LossType.CLASSIFICATION


class FocalLoss(Module, GenericClassificationLoss):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, *, gamma: float = 2.0):
        super().__init__()

        self.gamma = gamma

    @override
    def forward(
        self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]
    ) -> Float[Tensor, ""]:
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)

        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


class LogitBiasMixin:
    @property
    def logit_bias(self) -> Float[Tensor, "d"] | None:
        return self._logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Float[Tensor, "d"] | None):
        match value:
            case Tensor():
                if hasattr(self, "_logit_bias"):
                    del self._logit_bias

                self.register_buffer("_logit_bias", value, persistent=True)  # pyright: ignore[reportAttributeAccessIssue]

            case None:
                self._logit_bias = None


class LogitBiasFocalLoss(LogitBiasMixin, FocalLoss, GenericClassificationLoss):
    def __init__(
        self, *, logit_bias: Float[Tensor, "d"] | None = None, gamma: float = 2.0
    ):
        super().__init__(gamma=gamma)

        self.logit_bias = logit_bias

    @override
    def forward(self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]):
        return super().forward(input + self.logit_bias, target)


class LogitBiasCrossEntropyLoss(
    LogitBiasMixin, CrossEntropyLoss, GenericClassificationLoss
):
    def __init__(self, *args, logit_bias: Float[Tensor, "d"] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.logit_bias = logit_bias

    @override
    def forward(self, input: Float[Tensor, "b d"], target: Int[Tensor, "b"]):
        return super().forward(input + self.logit_bias, target)


class GaussianNLLLoss(torch.nn.GaussianNLLLoss, GenericRegressionLoss):
    """
    Class that makes vanilla torch.nn.GaussianNLLLoss compatible with carGPT pipeline
    """

    def __init__(
        self,
        *args,
        var_pos_function: Callable[
            [Tensor], Tensor
        ] = torch.exp,  # NOTE: use torch.ones_like to get vanilla MSE
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logit_bias = None
        self.var_pos_function = var_pos_function

    @override
    def forward(self, input: Tensor, target_values: Float[Tensor, "b"]):
        mean, log_var = input[..., 0], input[..., 1]
        return super().forward(
            input=mean, target=target_values, var=self.var_pos_function(log_var)
        )
