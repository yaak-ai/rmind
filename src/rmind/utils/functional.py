from typing import NamedTuple

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F


def gauss_prob(
    x: Tensor, mean: Tensor, std: Tensor, x_eps: float | Tensor = 0.1
) -> Tensor:
    dist = Normal(loc=mean, scale=std)
    return dist.cdf(x + x_eps / 2) - dist.cdf(x - x_eps / 2)


def diff_last(input: Tensor, n: int = 1, *, append: float | None = None) -> Tensor:
    append_ = (
        torch.tensor([append], device=input.device).expand(*input.shape[:-1], 1)
        if append is not None
        else None
    )
    return torch.diff(input, n=n, dim=-1, append=append_)


class SignalWithThresholdResult(NamedTuple):
    class_idx: Tensor
    prob: Tensor


def non_zero_signal_with_threshold(
    logits: Tensor, threshold: float = 0.8
) -> SignalWithThresholdResult:
    # assuming zero signal is at index 0
    probs = F.softmax(logits, dim=-1)
    max_prob_interest, max_idx_relative = torch.max(probs[..., 1:], dim=-1)
    final_class_idx = torch.where(
        max_prob_interest > threshold, max_idx_relative + 1, 0
    )
    final_prob = torch.gather(probs, -1, final_class_idx.unsqueeze(-1)).squeeze(-1)
    return SignalWithThresholdResult(class_idx=final_class_idx, prob=final_prob)
