from functools import partial

import more_itertools as mit
import torch
from einops import rearrange
from jaxtyping import Float, Shaped
from torch import Tensor
from torch.distributions import Normal
from torch.nn import functional as F


def pad_dim(
    input: Shaped[Tensor, "..."], *, pad: tuple[int, int], dim: int, **kwargs
) -> Float[Tensor, "..."]:
    _pad = [(0, 0) for _ in input.shape]
    _pad[dim] = pad
    _pad = list(mit.flatten(reversed(_pad)))

    if not torch.is_floating_point(input):
        input = input.float()

    return F.pad(input, _pad, **kwargs)


def nan_padder(*, pad: tuple[int, int], dim: int):
    return partial(pad_dim, pad=pad, dim=dim, mode="constant", value=torch.nan)


def gauss_prob(x: Tensor, mean: Tensor, std: Tensor, x_eps: float | Tensor = 0.1):
    dist = Normal(loc=mean, scale=std)
    return dist.cdf(x + x_eps / 2) - dist.cdf(x - x_eps / 2)

def flatten_batch_time(input: torch.Tensor) -> torch.Tensor:
    """Flattens the batch and time dimensions of the input tensor."""
    return rearrange(input, "b t ... -> (b t) ...")
