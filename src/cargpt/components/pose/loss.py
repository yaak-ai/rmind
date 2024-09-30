from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any
from torch.nn import Module

import torch
from jaxtyping import Float
from torch import Tensor

from .types import Pose

type Loss = Float[Tensor, ""]
type Metrics = dict[str, Float[Tensor, ""]]


class PoseLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Pose, label: Pose) -> Loss: ...


class MaskedPoseLoss(PoseLoss):
    def __init__(self, fn: Callable, mask: Sequence[int] = (1, 1, 1, 1, 1, 1)):
        super().__init__()
        self._fn = fn
        self._mask = torch.tensor(mask).to(torch.bool)

    def forward(self, pred: Pose, label: Pose) -> Loss:
        pred_masked = pred[..., self._mask]
        label_masked = label[..., self._mask].nan_to_num(0)
        return self._fn(pred_masked, label_masked)


class MaskedNormPoseLoss(MaskedPoseLoss):
    def __init__(self, fn: Callable, mask: Sequence[int]):
        super().__init__(fn=fn, mask=mask)

    def forward(self, pred: Pose, label: Pose) -> tuple[Loss, Metrics | None]:
        pred_norm = pred[..., self._mask].norm(p=2, dim=-1)
        label_norm = label[..., self._mask].nan_to_num(0).norm(p=2, dim=-1)
        return self._fn(pred_norm, label_norm)


class TranslationNormPoseLoss(MaskedNormPoseLoss):
    def __init__(self, fn: Callable = torch.nn.functional.l1_loss):
        super().__init__(fn=fn, mask=(1, 1, 1, 0, 0, 0))
