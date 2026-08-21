"""Unit tests for the auxiliary trajectory head's ground truth and loss
(DrivoR, arXiv:2601.05083): `rolling_dead_reckoned_trajectory`
(`rmind.components.trajectory`) and `winner_takes_all_pose_l1`
(`rmind.components.loss`).
"""

import math

import pytest
import torch

from rmind.components.loss import WinnerTakesAllPoseLoss, winner_takes_all_pose_l1
from rmind.components.trajectory import rolling_dead_reckoned_trajectory


def _dead_reckon_single_anchor(
    *,
    speed_kmh: torch.Tensor,
    heading_deg: torch.Tensor,
    time_stamp_s: torch.Tensor,
    reference_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct port of the single-anchor `dead_reckon_future_trajectory`
    (DrivoR, `feat/drivor` branch) -- used ONLY as an independent reference to
    cross-check `rolling_dead_reckoned_trajectory`'s closed-form rewrite
    against, at every anchor, not just the one this originally computes.
    """
    t0 = reference_index

    speed_m_s = speed_kmh[..., t0:-1] / 3.6
    heading_rad = torch.deg2rad(heading_deg)
    ref_heading = heading_rad[..., t0 : t0 + 1]
    step_heading_rel = heading_rad[..., t0:-1] - ref_heading
    dt = time_stamp_s[..., t0 + 1 :] - time_stamp_s[..., t0:-1]

    dx = speed_m_s * torch.sin(step_heading_rel) * dt
    dy = speed_m_s * torch.cos(step_heading_rel) * dt
    position = torch.stack([dx, dy], dim=-1).cumsum(dim=-2) / 100.0

    heading_rel = heading_rad[..., t0 + 1 :] - ref_heading
    heading_rel = torch.atan2(torch.sin(heading_rel), torch.cos(heading_rel))

    return position, heading_rel


def _make_series(
    *, seed: int = 0, t: int = 12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    speed_kmh = torch.rand(t, generator=g) * 80.0
    heading_deg = torch.rand(t, generator=g) * 720.0 - 360.0  # exercise wraparound
    dt_s = torch.rand(t - 1, generator=g) * 0.3 + 0.05
    time_stamp_s = torch.cat([torch.zeros(1), dt_s.cumsum(0)])
    time_stamp_us = (time_stamp_s * 1e6).long()
    return speed_kmh, heading_deg, time_stamp_us, time_stamp_s


def test_rolling_trajectory_shape() -> None:
    speed_kmh, heading_deg, time_stamp_us, _ = _make_series(t=12)
    out = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=5,
        num_poses=6,
    )
    assert out.shape == (5, 6, 3)


def test_rolling_trajectory_raises_when_not_enough_future_context() -> None:
    speed_kmh, heading_deg, time_stamp_us, _ = _make_series(t=10)
    with pytest.raises(ValueError, match="clip_horizon"):
        rolling_dead_reckoned_trajectory(
            speed_kmh=speed_kmh,
            heading_deg=heading_deg,
            time_stamp_us=time_stamp_us,
            episode_length=5,
            num_poses=6,
        )


def test_rolling_trajectory_heading_is_wrapped() -> None:
    t = 8
    speed_kmh = torch.zeros(t)  # position irrelevant to this check
    heading_deg = torch.zeros(t)
    heading_deg[0] = -179.0
    heading_deg[-1] = 179.0  # ~2deg apart the short way around, not ~358deg
    time_stamp_us = torch.arange(t) * 100_000

    out = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=1,
        num_poses=t - 1,
    )
    heading_rel = out[0, -1, 2]
    assert heading_rel.abs() < math.radians(5)


def test_rolling_trajectory_matches_single_anchor_reference_at_every_anchor() -> None:
    """The closed-form rewrite must equal a fresh single-anchor computation at
    EVERY anchor -- this is the actual claim being ported.
    """
    speed_kmh, heading_deg, time_stamp_us, time_stamp_s = _make_series(t=14)
    episode_length, num_poses = 6, 5

    rolling = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=episode_length,
        num_poses=num_poses,
    )

    for t0 in range(episode_length):
        ref_position, ref_heading = _dead_reckon_single_anchor(
            speed_kmh=speed_kmh,
            heading_deg=heading_deg,
            time_stamp_s=time_stamp_s,
            reference_index=t0,
        )
        torch.testing.assert_close(rolling[t0, :, :2], ref_position[:num_poses])
        torch.testing.assert_close(rolling[t0, :, 2], ref_heading[:num_poses])


def test_rolling_trajectory_batched() -> None:
    b = 3
    series = [_make_series(seed=i, t=12) for i in range(b)]
    speed_kmh = torch.stack([s[0] for s in series])
    heading_deg = torch.stack([s[1] for s in series])
    time_stamp_us = torch.stack([s[2] for s in series])

    out = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=4,
        num_poses=6,
    )
    assert out.shape == (b, 4, 6, 3)


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
