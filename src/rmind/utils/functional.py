from typing import NamedTuple

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module
from torch.nn import functional as F


def compiled(module: Module, *, disable: bool = False, **kwargs: object) -> Module:
    """Compile a module in-place and return it. For use as a Hydra wrapper.

    Unlike torch.compile(), Module.compile() mutates the module in-place so
    state dict keys are unchanged (no _orig_mod prefix). Set ``disable=True``
    to skip compilation entirely (e.g. for debug configs).
    """
    if not disable:
        module.compile(**kwargs)
    return module


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


def build_local_trajectory(
    xy: Tensor,
    heading_deg: Tensor,
    *,
    history_steps: int,
) -> Tensor:
    """Build a local trajectory from absolute UTM positions and headings.

    Takes the positions at steps [history_steps:] and expresses them in the
    ego-local frame at step [history_steps - 1], matching the convention used
    in dataset preprocessing (translate to ego, rotate counterclockwise by heading).

    Args:
        xy: (batch, T, 2) absolute UTM positions.
        heading_deg: (batch, T, 1) heading in degrees.
        history_steps: number of history steps; the reference frame is at
            step [history_steps - 1].

    Returns:
        (batch, T - history_steps, 2) positions in the ego-local frame.
    """
    ref_xy = xy[:, history_steps - 1]                              # (batch, 2)
    ref_heading = heading_deg[:, history_steps - 1].squeeze(-1)  # (batch,) degrees
    future_xy = xy[:, history_steps:]                    # (batch, n_future, 2)

    delta = future_xy - ref_xy.unsqueeze(1)              # (batch, n_future, 2)
    theta = torch.deg2rad(ref_heading)
    cos_t = theta.cos().unsqueeze(1)                     # (batch, 1)
    sin_t = theta.sin().unsqueeze(1)

    dx, dy = delta[..., 0], delta[..., 1]
    x_local = dx * cos_t - dy * sin_t
    y_local = dx * sin_t + dy * cos_t

    return torch.stack([x_local, y_local], dim=-1)       # (batch, n_future, 2)


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
