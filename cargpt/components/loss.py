import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, *, gamma: float = 2.0):
        super().__init__()

        self.gamma = gamma

    def forward(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
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

                self.register_buffer("_logit_bias", value)  # pyright: ignore

            case None:
                self._logit_bias = None


class LogitBiasFocalLoss(FocalLoss, LogitBiasMixin):
    def __init__(
        self,
        *,
        logit_bias: Float[Tensor, "d"] | None = None,
        gamma: float = 2.0,
    ):
        super().__init__(gamma=gamma)

        self.logit_bias = logit_bias

    def forward(self, inputs: Float[Tensor, "b d"], targets: Int[Tensor, "b"]):
        return super().forward(inputs + self.logit_bias, targets)


class LogitBiasCrossEntropyLoss(CrossEntropyLoss, LogitBiasMixin):
    def __init__(
        self,
        *args,
        logit_bias: Float[Tensor, "d"] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.logit_bias = logit_bias

    def forward(self, inputs: Float[Tensor, "b d"], targets: Int[Tensor, "b"]):
        return super().forward(inputs + self.logit_bias, targets)
