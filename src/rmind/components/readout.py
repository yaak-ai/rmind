"""Readout: collapse K candidate action-chunk draws into one anchor.

Objective-agnostic: a sample-then-select objective produces (B, K, H, A) draws
(K = `Readout.num_samples`) and a Readout reduces them to (B, H, A). The objective
owns sampling; the readout owns aggregation, so strategies are reusable/swappable.

- `SingleReadout` — one honest draw (num_samples=1).
- `AxisModeReadout` — winner-take-all consensus along one action axis (`mode_axis`).

Why single-axis consensus: mean-of-K splits bimodal conditionals (turn left XOR
right), and the catastrophic case is directional — averaging left and right drives
through the median, while longitudinal modes average gracefully. So we cluster on
the steering axis and commit to the dominant mode (val: spike steering L1 0.53 ->
0.34 vs mean-of-K). `mode_axis` is config, since which axis must not be averaged is
task-specific.

Why 1-D, not clustering the full R^(H*A) point: at K~16 high-D clustering is noisy;
a raw Euclidean metric lets per-step longitudinal jitter swamp the steering split;
and the chunk-mean is phase-invariant ("which way overall", not "when"). The cost
is a blind spot for modes orthogonal to the axis; a richer signature (PCA axis,
maneuver features) would slot in behind this same interface.
"""

from abc import ABC, abstractmethod
from typing import Annotated, override

import torch
from pydantic import ConfigDict, Field, validate_call
from torch import Tensor, nn


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def axis_mode_anchor(
    samples: Tensor,
    mode_axis: int,
    *,
    gap_the: Annotated[float, Field(ge=0)] = 0.15,
    min_frac: Annotated[float, Field(ge=0, le=1)] = 0.125,
) -> Tensor:
    """Winner-take-all consensus over draws along one axis -> anchor (B, H, A).

    samples: (B, K, H, A) action chunks (K draws/frame). mode_axis: channel to
    cluster on (negative indexes from the end, e.g. -1 = last).

    Per frame: take each draw's chunk-mean on `mode_axis`, split the K by the
    largest gap; if it exceeds `gap_the` and the minority holds >= `min_frac`,
    commit to the dominant cluster and average its full chunks, else mean-of-K.
    Non-finite draws keep their (non-finite) mean.

    Raises:
        ValueError: if samples is not 4-D or mode_axis is out of range.
    """
    if samples.ndim != 4:  # noqa: PLR2004
        msg = f"samples must be 4-D (B, K, H, A), got {samples.ndim}-D"
        raise ValueError(msg)
    channels = samples.shape[-1]
    axis = mode_axis + channels if mode_axis < 0 else mode_axis
    if not 0 <= axis < channels:
        msg = f"mode_axis {mode_axis} out of range for {channels} channels"
        raise ValueError(msg)
    k = samples.shape[1]
    sig = samples[..., axis].mean(dim=2)  # (B, K) chunk-mean on the mode axis
    sig_sorted, order = torch.sort(sig, dim=1)
    mode_sep, split = sig_sorted.diff(dim=1).max(dim=1)  # largest gap + its index
    left_n = split + 1
    minority = torch.minimum(left_n, k - left_n).float() / k
    bimodal = (mode_sep > gap_the) & (minority >= min_frac) & torch.isfinite(mode_sep)

    out = samples.mean(dim=1)  # (B, H, A) default: mean-of-K
    for i in torch.nonzero(bimodal).flatten().tolist():
        n_left = int(left_n[i])
        members = order[i, :n_left] if n_left >= k - n_left else order[i, n_left:]
        out[i] = samples[i, members].mean(dim=0)
    return out


class Readout(nn.Module, ABC):
    """Collapse K candidate action-chunk draws to one anchor: (B,K,H,A)->(B,H,A).

    Objective-agnostic aggregation: `num_samples` declares how many draws the
    objective should generate, and `forward` reduces them. The objective owns
    sampling; subclasses own only the reduction.
    """

    num_samples: int

    @abstractmethod
    def forward(self, draws: Tensor) -> Tensor:
        """(B, K, H, A) candidate draws -> (B, H, A) anchor chunk."""
        ...


class SingleReadout(Readout):
    """One honest draw (num_samples=1); deterministic given the sampling RNG."""

    num_samples = 1

    @override
    def forward(self, draws: Tensor) -> Tensor:
        return draws[:, 0]


class AxisModeReadout(Readout):
    """Winner-take-all consensus along a single action axis (`axis_mode_anchor`).

    `mode_axis` (default -1, the last channel — steering for the driving policy)
    selects the axis whose modes must not be averaged; `num_samples` (>= 2) is the
    number of draws to cluster. `gap_the`/`min_frac` tune the bimodality test.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        num_samples: Annotated[int, Field(gt=1)],
        mode_axis: int = -1,
        gap_the: Annotated[float, Field(ge=0)] = 0.15,
        min_frac: Annotated[float, Field(ge=0, le=1)] = 0.125,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.mode_axis = mode_axis
        self.gap_the = gap_the
        self.min_frac = min_frac

    @override
    def forward(self, draws: Tensor) -> Tensor:
        return axis_mode_anchor(
            draws, self.mode_axis, gap_the=self.gap_the, min_frac=self.min_frac
        )
