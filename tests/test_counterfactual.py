import math

import pytest
import torch
from torch import Tensor

from rmind.components.counterfactual import (
    expected_plan,
    goal_sensitivity,
    speed_echo,
    synthetic_turn_waypoints,
)

B, T, N = 4, 3, 10
POSE_TIMES_S = torch.arange(1, 6) / 3.0  # 5 poses x 1/3 s


def _inputs() -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    waypoints = torch.randn((B, T, N, 2), generator=generator)
    speed = torch.rand((B, T, 1), generator=generator) * 50.0
    return waypoints, speed


def _speed_copier(_waypoints: Tensor, speed: Tensor) -> Tensor:
    """Straight plan, forward distance = newest speed x elapsed time."""
    forward = speed[:, -1] / 3.6 * POSE_TIMES_S  # (b, p)
    zeros = torch.zeros_like(forward)
    return torch.stack([zeros, forward, zeros], dim=-1)


def _goal_follower(waypoints: Tensor, _speed: Tensor) -> Tensor:
    """Heads for the newest frame's last waypoint (x right, y forward)."""
    end = waypoints[:, -1, -1] * 100.0  # (b, 2) meters
    poses = end[:, None, :] * (POSE_TIMES_S[:, None] / POSE_TIMES_S[-1])
    heading = torch.atan2(end[:, 0], end[:, 1])[:, None].expand(-1, len(POSE_TIMES_S))
    return torch.cat([poses, heading[..., None]], dim=-1)


def test_synthetic_turn_geometry() -> None:
    right = synthetic_turn_waypoints(left=False) * 100.0
    left = synthetic_turn_waypoints(left=True) * 100.0

    assert right.shape == (10, 2)
    torch.testing.assert_close(left, right * torch.tensor([-1.0, 1.0]))
    # 1 m straight ahead, then into a 90 deg right turn of radius 10 m entered 2 m out
    torch.testing.assert_close(right[0], torch.tensor([0.0, 1.0]))
    phi = 9.0 / 10.0
    torch.testing.assert_close(
        right[1], torch.tensor([10.0 * (1 - math.cos(phi)), 2.0 + 10.0 * math.sin(phi)])
    )
    # past the arc: heading +x at y = 2 + R
    exit_x = 10.0 + (89.0 - math.pi / 2 * 10.0)
    torch.testing.assert_close(right[-1], torch.tensor([exit_x, 12.0]))
    # sampled by arc length: chords never exceed the 10 m spacing
    assert (right.diff(dim=0).norm(dim=-1) <= 10.0 + 1e-4).all()


def test_goal_sensitivity_zero_for_goal_ignoring_policy() -> None:
    waypoints, speed = _inputs()
    yaw, lateral = goal_sensitivity(_speed_copier, waypoints, speed)

    assert yaw.shape == lateral.shape == (B,)
    assert (yaw == 0).all()
    assert (lateral == 0).all()


def test_goal_sensitivity_sign_for_goal_following_policy() -> None:
    waypoints, speed = _inputs()
    yaw, lateral = goal_sensitivity(_goal_follower, waypoints, speed)

    # left is negative heading / negative x in this frame
    assert (yaw < 0).all()
    assert (lateral < 0).all()


def test_speed_echo_of_speed_copier() -> None:
    waypoints, _ = _inputs()
    stopped = torch.zeros((B, T, 1))
    echo = speed_echo(_speed_copier, waypoints, stopped)

    torch.testing.assert_close(echo, torch.full((B,), 10.0 / 3.6 * 5.0 / 3.0))
    assert echo[0].item() == pytest.approx(4.63, abs=0.01)


def test_speed_echo_of_vision_policy_stays_at_real_plan() -> None:
    waypoints, _ = _inputs()
    stopped = torch.zeros((B, T, 1))
    echo = speed_echo(_goal_follower, waypoints, stopped)

    torch.testing.assert_close(echo, _goal_follower(waypoints, stopped)[:, -1, 1])


def test_expected_plan() -> None:
    trajectory = torch.randn(B, 3, 5, 3)
    torch.testing.assert_close(expected_plan(trajectory), trajectory.mean(dim=1))

    one_hot = torch.full((B, 3), -1e4)
    one_hot[:, 1] = 0.0
    torch.testing.assert_close(expected_plan(trajectory, one_hot), trajectory[:, 1])
