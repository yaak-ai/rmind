import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()

        self.gamma = gamma

    def compute(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, "b"]:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        return (1 - pt).pow(self.gamma) * ce_loss

    def forward(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        loss = self.compute(inputs, targets)

        return loss.mean()


class AlphaFocalLoss(FocalLoss):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(
        self,
        *,
        alpha: Float[Tensor, "n"] | float | None = None,
        gamma: float = 2.0,
    ):
        super().__init__(gamma)

        self.alpha = alpha

    @property
    def alpha(self) -> Float[Tensor, "n"] | float | None:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Float[Tensor, "n"] | float | None):
        match value:
            case Tensor():
                if hasattr(self, "_alpha"):
                    del self._alpha

                self.register_buffer("_alpha", value)

            case float() | None:
                self._alpha = value

            case _:
                raise NotImplementedError

    def compute(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, "b"]:
        match self.alpha:
            case Tensor():
                alpha = self.alpha[targets]

            case float():
                alpha = self.alpha

            case _:
                raise NotImplementedError

        loss = super().compute(inputs, targets)

        return alpha * loss

    def forward(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        loss = self.compute(inputs, targets)

        return loss.mean()


class DeltaFocalLoss(FocalLoss):
    def __init__(self, alpha: float, beta: float, gamma: float = 2.0):
        super().__init__(gamma)

        self.alpha = alpha
        self.beta = beta

    def compute(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
        deltas: Float[Tensor, "b"],
    ) -> Float[Tensor, "b"]:
        weights = self.alpha * deltas.abs() + self.beta
        loss = super().compute(inputs, targets)

        return weights * loss

    def forward(
        self,
        inputs: Float[Tensor, "b d"],
        targets: Int[Tensor, "b"],
        deltas: Float[Tensor, "b"],
    ) -> Float[Tensor, ""]:
        loss = self.compute(inputs, targets, deltas)

        return loss.mean()
