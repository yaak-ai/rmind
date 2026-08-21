from collections.abc import Callable
from typing import Any, Literal, Protocol, override, runtime_checkable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module


@runtime_checkable
class HasLogitBias(Protocol):
    logit_bias: Tensor | None


class FocalLoss(Module):
    """https://arxiv.org/pdf/1708.02002.pdf.

    `label_smoothing` uses the smoothed-CE decomposition
    `(1-eps) * ce_true + eps * ce_uniform`, but focal-modulates ONLY the
    true-class term. Modulating both would let `(1-pt)^gamma -> 0` cancel
    the smoothing penalty exactly on confident predictions -- the case
    smoothing exists for. The unmodulated `eps * ce_uniform` term grows
    with the logit margin, capping overconfidence (and the entropy
    collapse behind the calibration blowup). At `label_smoothing=0.0`
    this is exactly the unsmoothed focal loss.
    """

    def __init__(self, *, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()

        self.gamma: float = gamma
        self.label_smoothing: float = label_smoothing

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_raw = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_raw)
        focal = (1 - pt).pow(self.gamma) * ce_raw
        eps = self.label_smoothing
        if not eps:
            return focal.mean()

        ce_uniform = -F.log_softmax(input, dim=-1).mean(dim=-1)
        return ((1 - eps) * focal + eps * ce_uniform).mean()


class LogitBiasFocalLoss(FocalLoss, HasLogitBias):
    def __init__(self, *, logit_bias: Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__(gamma=gamma)

        self.logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class LogitBiasCrossEntropyLoss(CrossEntropyLoss, HasLogitBias):
    def __init__(
        self, *args: Any, logit_bias: Tensor | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.logit_bias: Tensor | None = logit_bias

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input + self.logit_bias, target)  # ty:ignore[unsupported-operator]


class GaussianNLLLoss(torch.nn.GaussianNLLLoss):
    def __init__(
        self,
        *args: Any,
        # NOTE: use torch.ones_like to get vanilla MSE
        var_pos_function: Callable[[Tensor], Tensor] = torch.exp,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.var_pos_function: Callable[[Tensor], Tensor] = var_pos_function

    @override
    def forward(
        self, input: Tensor, target: Tensor, var: Tensor | None = None
    ) -> Tensor:  # ty:ignore[invalid-method-override]
        if var is not None:
            raise ValueError

        mean, log_var = input[..., 0], input[..., 1]
        var = self.var_pos_function(log_var)

        return super().forward(input=mean, target=target, var=var)


class GramAnchoringLoss(Module):
    """Gram-based anchoring loss for feature matching.
    Based on DINOv3 implementation:
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
    Uses target-driven within-frame patch uniqueness weights.
    """

    def __init__(
        self,
        *args: Any,
        patches: int,
        weight_sim: float = 1.0,
        weight_gram: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.weight_sim: float = weight_sim
        self.weight_gram: float = weight_gram
        self.patches: int = patches

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.detach()
        eps = 1e-6

        input_view = rearrange(input, "(bt p) d -> bt p d", p=self.patches)
        target_view = rearrange(target, "(bt p) d -> bt p d", p=self.patches)
        input_n = F.normalize(input_view, dim=-1)
        target_n = F.normalize(target_view, dim=-1)

        # Target-driven within-frame patch uniqueness weights.
        # Patches similar to many others in the same frame are downweighted.
        frame_sim = torch.einsum("bpd,bqd->bpq", target_n, target_n).clamp_min(0.0)
        eye = torch.eye(self.patches, dtype=torch.bool, device=target.device)
        weights = 1.0 / (
            frame_sim.masked_fill(eye, 0.0).sum(dim=-1) / (self.patches - 1) + eps
        )
        weights /= weights.sum(dim=1, keepdim=True) + eps

        patch_loss = F.mse_loss(input_view, target_view, reduction="none").mean(dim=-1)
        sim_loss = (weights * patch_loss).sum(dim=1).mean()

        if self.weight_gram <= 0:
            return self.weight_sim * sim_loss

        # Gram on L2-normed features weighted by patch uniqueness.
        gram_pred = torch.einsum("bpd,bqd->bpq", input_n, input_n)
        gram_gt = torch.einsum("bpd,bqd->bpq", target_n, target_n)
        pair_weights = torch.einsum("bp,bq->bpq", weights, weights)  # (bt, p, p)
        gram_loss = (pair_weights * (gram_pred - gram_gt).pow(2)).sum(dim=(1, 2)).mean()

        return self.weight_sim * sim_loss + self.weight_gram * gram_loss


def _per_candidate_pose_errors(input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Per-candidate, per-pose xy L1 error and wrapped heading error (radians)
    -- the shared core of `winner_takes_all_pose_l1` and
    `winner_takes_all_pose_l1_components`.

    Args:
        input: `(*batch, Q, P, 3)` candidate poses, last dim `(x, y, theta)`.
        target: `(*batch, P, 3)` ground-truth poses, same layout.

    Returns:
        `(xy_err, heading_err)`, each `(*batch, Q, P)`.
    """
    pred_xy, pred_heading = input[..., :2], input[..., 2]
    target_xy, target_heading = target[..., :2], target[..., 2]
    xy_err = F.l1_loss(
        pred_xy, target_xy.unsqueeze(-3).expand_as(pred_xy), reduction="none"
    ).sum(dim=-1)
    heading_diff = pred_heading - target_heading.unsqueeze(-2)
    heading_err = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff)).abs()
    return xy_err, heading_err


def winner_takes_all_pose_l1(
    input: Tensor,
    target: Tensor,
    *,
    heading_weight: float = 0.1,
    reduction: Literal["mean", "sum"] = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Winner-takes-all trajectory pose loss (DrivoR, arXiv:2601.05083): for
    each sample, minimize the L1 (position) + wrapped-angular (heading) error
    of only the best-matching candidate trajectory. `torch.min` is natively
    differentiable, so gradient flows only to the winning candidate per
    sample -- no straight-through estimator needed.

    Args:
        input: `(*batch, Q, P, 3)` candidate poses, last dim `(x, y, theta)`.
        target: `(*batch, P, 3)` ground-truth poses, same layout (e.g.
            `rmind.components.trajectory.rolling_dead_reckoned_trajectory`'s
            output).
        heading_weight: weight balancing wrapped-heading error (radians)
            against `/100`-normalized position error -- not specified by the
            paper, tunable.
        reduction: `"mean"` or `"sum"` over the batch.

    Returns:
        `(loss, best_index, per_candidate_loss)`: `loss` is a scalar;
        `best_index` is `(*batch,)`; `per_candidate_loss` is `(*batch, Q)`.
    """
    xy_err, heading_err = _per_candidate_pose_errors(input, target)
    per_candidate = (xy_err + heading_weight * heading_err).mean(dim=-1)

    min_loss, best_index = per_candidate.min(dim=-1)
    loss = min_loss.mean() if reduction == "mean" else min_loss.sum()

    return loss, best_index, per_candidate


def winner_takes_all_pose_l1_components(
    input: Tensor,
    target: Tensor,
    *,
    heading_weight: float = 0.1,
    reduction: Literal["mean", "sum"] = "mean",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Like `winner_takes_all_pose_l1`, but also breaks the winning
    candidate's error back out into its separate xy-L1 and heading terms --
    `winner_takes_all_pose_l1` only ever exposes the two summed together
    (scaled by `heading_weight`), which makes it impossible to tell from
    `loss` alone whether position or heading is driving it. Intended for
    logging (e.g. `trajectory_loss_xy`/`trajectory_loss_heading`), not for
    computing the actual optimized loss (`loss` here is numerically identical
    to `winner_takes_all_pose_l1`'s).

    Returns:
        `(loss, best_index, per_candidate_loss, winner_xy_loss,
        winner_heading_loss)`: the first three match
        `winner_takes_all_pose_l1`; `winner_xy_loss`/`winner_heading_loss` are
        `(*batch,)`, the winning candidate's mean-over-poses xy/heading error,
        unweighted (i.e. before `heading_weight` is applied).
    """
    xy_err, heading_err = _per_candidate_pose_errors(input, target)
    per_candidate = (xy_err + heading_weight * heading_err).mean(dim=-1)

    min_loss, best_index = per_candidate.min(dim=-1)
    loss = min_loss.mean() if reduction == "mean" else min_loss.sum()

    index = best_index.unsqueeze(-1)
    winner_xy_loss = xy_err.mean(dim=-1).gather(-1, index).squeeze(-1)
    winner_heading_loss = heading_err.mean(dim=-1).gather(-1, index).squeeze(-1)

    return loss, best_index, per_candidate, winner_xy_loss, winner_heading_loss


class WinnerTakesAllPoseLoss(Module):
    """`losses: ModuleDict` wrapper around `winner_takes_all_pose_l1` -- see
    that function for the loss definition. `input`/`target` combine `(x, y,
    theta)` in one trailing axis (rather than separate xy/heading args) so
    this matches the `forward(input, target) -> Tensor` convention every
    other `losses[...]` entry in this file uses.
    """

    def __init__(
        self, *, heading_weight: float = 0.1, reduction: Literal["mean", "sum"] = "mean"
    ) -> None:
        super().__init__()

        self.heading_weight = heading_weight
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss, _, _ = winner_takes_all_pose_l1(
            input, target, heading_weight=self.heading_weight, reduction=self.reduction
        )
        return loss
