from typing import Literal, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _per_candidate_pose_errors(
    input: Tensor, target_xy: Tensor, target_heading: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-candidate, per-pose xy L1 error and wrapped heading error (radians)
    -- the shared core of `winner_takes_all_pose_l1` and
    `winner_takes_all_pose_l1_components`.

    Returns:
        `(xy_err, heading_err)`, each `(*batch, Q, P)`.
    """
    pred_xy, pred_heading = input[..., :2], input[..., 2]
    xy_err = F.l1_loss(
        pred_xy, target_xy.unsqueeze(-3).expand_as(pred_xy), reduction="none"
    ).sum(dim=-1)
    heading_diff = pred_heading - target_heading.unsqueeze(-2)
    heading_err = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff)).abs()
    return xy_err, heading_err


def winner_takes_all_pose_l1(
    input: Tensor,
    target_xy: Tensor,
    target_heading: Tensor,
    *,
    heading_weight: float = 0.1,
    reduction: Literal["mean", "sum"] = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Winner-takes-all pose loss (DrivoR, arXiv:2601.05083): for each sample,
    minimize the L1 (position) + wrapped-angular (heading) error of only the
    best-matching candidate trajectory. `torch.min` is natively
    differentiable, so gradient flows only to the winning candidate per
    sample -- no straight-through estimator needed.

    Args:
        input: `(*batch, Q, P, 3)` candidate poses, last dim `(x, y, theta)`.
        target_xy: `(*batch, P, 2)` ground-truth position.
        target_heading: `(*batch, P)` ground-truth heading, radians.
        heading_weight: weight balancing wrapped-heading error (radians)
            against `/100`-normalized position error -- not specified by the
            paper, tunable.
        reduction: `"mean"` or `"sum"` over the batch.

    Returns:
        `(loss, best_index, per_candidate_loss)`: `loss` is a scalar;
        `best_index` is `(*batch,)`; `per_candidate_loss` is `(*batch, Q)`.
    """
    xy_err, heading_err = _per_candidate_pose_errors(input, target_xy, target_heading)
    per_candidate = (xy_err + heading_weight * heading_err).mean(dim=-1)

    min_loss, best_index = per_candidate.min(dim=-1)
    loss = min_loss.mean() if reduction == "mean" else min_loss.sum()

    return loss, best_index, per_candidate


def winner_takes_all_pose_l1_components(
    input: Tensor,
    target_xy: Tensor,
    target_heading: Tensor,
    *,
    heading_weight: float = 0.1,
    reduction: Literal["mean", "sum"] = "mean",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Like `winner_takes_all_pose_l1`, but also breaks the winning
    candidate's error back out into its separate xy-L1 and heading terms --
    `winner_takes_all_pose_l1` only ever exposes the two summed together
    (scaled by `heading_weight`), which makes it impossible to tell from
    `loss` alone whether position or heading is driving it. Intended for
    logging (e.g. `train/loss_xy`, `train/loss_heading`), not for computing
    the actual optimized loss (`loss` here is numerically identical to
    `winner_takes_all_pose_l1`'s).

    Returns:
        `(loss, best_index, per_candidate_loss, winner_xy_loss,
        winner_heading_loss)`: the first three match
        `winner_takes_all_pose_l1`; `winner_xy_loss`/`winner_heading_loss` are
        `(*batch,)`, the winning candidate's mean-over-poses xy/heading error,
        unweighted (i.e. before `heading_weight` is applied).
    """
    xy_err, heading_err = _per_candidate_pose_errors(input, target_xy, target_heading)
    per_candidate = (xy_err + heading_weight * heading_err).mean(dim=-1)

    min_loss, best_index = per_candidate.min(dim=-1)
    loss = min_loss.mean() if reduction == "mean" else min_loss.sum()

    index = best_index.unsqueeze(-1)
    winner_xy_loss = xy_err.mean(dim=-1).gather(-1, index).squeeze(-1)
    winner_heading_loss = heading_err.mean(dim=-1).gather(-1, index).squeeze(-1)

    return loss, best_index, per_candidate, winner_xy_loss, winner_heading_loss


class WinnerTakesAllPoseLoss(nn.Module):
    def __init__(
        self, *, heading_weight: float = 0.1, reduction: Literal["mean", "sum"] = "mean"
    ) -> None:
        super().__init__()

        self.heading_weight = heading_weight
        self.reduction = reduction

    @override
    def forward(
        self, input: Tensor, target_xy: Tensor, target_heading: Tensor
    ) -> Tensor:
        loss, _, _ = winner_takes_all_pose_l1(
            input,
            target_xy,
            target_heading,
            heading_weight=self.heading_weight,
            reduction=self.reduction,
        )
        return loss
