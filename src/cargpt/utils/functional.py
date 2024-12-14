from functools import partial

import more_itertools as mit
import torch
from jaxtyping import Float, Shaped
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F


def pad_dim(
    input: Shaped[Tensor, "..."], *, pad: tuple[int, int], dim: int, **kwargs
) -> Float[Tensor, "..."]:
    pad_ = [(0, 0) for _ in input.shape]
    pad_[dim] = pad
    pad_ = list(mit.flatten(reversed(pad_)))

    if not torch.is_floating_point(input):
        input = input.float()

    return F.pad(input, pad_, **kwargs)


def nan_padder(*, pad: tuple[int, int], dim: int):
    return partial(pad_dim, pad=pad, dim=dim, mode="constant", value=torch.nan)


def gauss_prob(x: Tensor, mean: Tensor, std: Tensor, x_eps: float | Tensor = 0.1):
    dist = Normal(loc=mean, scale=std)
    return dist.cdf(x + x_eps / 2) - dist.cdf(x - x_eps / 2)
