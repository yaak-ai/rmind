"""Unit tests for `rmind.components.loss`'s winner-takes-all pose loss
(DrivoR, arXiv:2601.05083), the auxiliary trajectory head's training
objective (docs/phase3_trajectory_head_plan.md).
"""

import torch

from rmind.components.loss import (
    WinnerTakesAllPoseLoss,
    winner_takes_all_pose_l1,
    winner_takes_all_pose_l1_components,
)


def test_winner_takes_all_pose_l1_selects_min_error_hypothesis() -> None:
    target = torch.zeros(1, 3, 3)  # (batch=1, num_poses=3, [x, y, theta])
    pred = torch.zeros(1, 2, 3, 3, requires_grad=True)  # (batch, Q=2, P=3, 3)
    with torch.no_grad():
        pred[:, 0] = 1.0  # far from target
        pred[:, 1] = 0.1  # close to target -- should win

    loss, best_index, per_candidate = winner_takes_all_pose_l1(pred, target)
    assert best_index.tolist() == [1]
    assert per_candidate.shape == (1, 2)

    loss.backward()
    assert pred.grad is not None
    assert torch.all(pred.grad[:, 1] != 0)
    assert torch.all(pred.grad[:, 0] == 0)


def test_winner_takes_all_pose_loss_module_matches_function() -> None:
    pred = torch.randn(2, 4, 5, 3)
    target = torch.randn(2, 5, 3)

    module_loss = WinnerTakesAllPoseLoss(heading_weight=0.2)(pred, target)
    fn_loss, _, _ = winner_takes_all_pose_l1(pred, target, heading_weight=0.2)

    torch.testing.assert_close(module_loss, fn_loss)


def test_winner_takes_all_pose_l1_components_matches_plain_function() -> None:
    """`winner_takes_all_pose_l1_components`'s `loss` output must be numerically
    identical to `winner_takes_all_pose_l1`'s -- the components function only
    adds a further breakdown, it must not change the optimized value.
    """
    pred = torch.randn(2, 4, 5, 3)
    target = torch.randn(2, 5, 3)

    loss, best_index, per_candidate = winner_takes_all_pose_l1(
        pred, target, xy_weight=0.02, heading_weight=0.3
    )
    (
        components_loss,
        components_best_index,
        components_per_candidate,
        winner_xy_loss,
        winner_heading_loss,
    ) = winner_takes_all_pose_l1_components(
        pred, target, xy_weight=0.02, heading_weight=0.3
    )

    torch.testing.assert_close(components_loss, loss)
    torch.testing.assert_close(components_best_index, best_index)
    torch.testing.assert_close(components_per_candidate, per_candidate)
    assert winner_xy_loss.shape == (2,)
    assert winner_heading_loss.shape == (2,)

    # the winner's unweighted xy_weight*xy + heading_weight*heading must
    # reconstruct the winning per-candidate loss entry
    index = best_index.unsqueeze(-1)
    winner_per_candidate = per_candidate.gather(-1, index).squeeze(-1)
    torch.testing.assert_close(
        0.02 * winner_xy_loss + 0.3 * winner_heading_loss, winner_per_candidate
    )
