import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class FocalLoss(nn.Module):
    """https://arxiv.org/pdf/1708.02002.pdf"""

    def __init__(self, *, gamma=2):
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
