from functools import partial
from typing import Any

import more_itertools as mit
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F


def pad_dim(input: Tensor, *, pad: tuple[int, int], dim: int, **kwargs: Any) -> Tensor:
    pad_ = [(0, 0) for _ in input.shape]
    pad_[dim] = pad
    pad_ = list(mit.flatten(reversed(pad_)))

    if not torch.is_floating_point(input):
        input = input.float()

    return F.pad(input, pad_, **kwargs)


def nan_padder(*, pad: tuple[int, int], dim: int) -> partial[Tensor]:
    return partial(pad_dim, pad=pad, dim=dim, mode="constant", value=torch.nan)


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
