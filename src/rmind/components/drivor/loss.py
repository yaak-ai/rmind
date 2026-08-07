from typing import Literal, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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
    pred_xy, pred_heading = input[..., :2], input[..., 2]
    xy_err = F.l1_loss(
        pred_xy, target_xy.unsqueeze(-3).expand_as(pred_xy), reduction="none"
    ).sum(dim=-1)
    heading_diff = pred_heading - target_heading.unsqueeze(-2)
    heading_err = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff)).abs()
    per_candidate = (xy_err + heading_weight * heading_err).mean(dim=-1)

    min_loss, best_index = per_candidate.min(dim=-1)
    loss = min_loss.mean() if reduction == "mean" else min_loss.sum()

    return loss, best_index, per_candidate


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
