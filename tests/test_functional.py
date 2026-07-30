import math

import torch
from torch.testing import assert_close

from rmind.utils.functional import (
    build_local_trajectory,
    build_relative_trajectory,
    compose_relative_trajectory,
    diff_last,
    point_to_polyline_distance,
    route_convergence_signal,
)


def test_compose_relative_trajectory_inverts_build_relative_trajectory() -> None:
    """compose_relative_trajectory(build_relative_trajectory(...)) should
    recover exactly the same reference-frame positions as build_local_trajectory
    -- the two are different parameterizations (chained per-step deltas vs. one
    fixed-frame rotation) of the identical physical path."""
    generator = torch.Generator().manual_seed(0)

    for batch_size, total_steps, history_steps in [
        (1, 6, 5),
        (4, 11, 5),
        (3, 8, 2),
    ]:
        heading_deg = torch.rand(
            batch_size, total_steps, 1, generator=generator
        ) * 360 - 180
        step_xy = torch.randn(batch_size, total_steps, 2, generator=generator)
        xy = torch.cumsum(step_xy, dim=1)

        expected = build_local_trajectory(
            xy=xy, heading_deg=heading_deg, history_steps=history_steps
        )
        rel = build_relative_trajectory(
            xy=xy, heading_deg=heading_deg, history_steps=history_steps
        )
        actual = compose_relative_trajectory(rel)

        assert_close(actual, expected)


def test_diff_last_append() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x, append=math.nan)
    assert_close(out[:, :2], torch.tensor([[1.0, 2.0]]))
    assert torch.isnan(out[:, 2]).all()


def test_diff_last_prepend() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x, prepend=math.nan)
    assert torch.isnan(out[:, 0]).all()
    assert_close(out[:, 1:], torch.tensor([[1.0, 2.0]]))


def test_diff_last_neither() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x)
    assert_close(out, torch.tensor([[1.0, 2.0]]))


def test_point_to_polyline_distance_on_segment_is_zero() -> None:
    vertices = torch.tensor([[[0.0, 0.0], [0.0, 10.0], [5.0, 20.0]]])  # (1, 3, 2)
    points = torch.tensor([[[0.0, 5.0], [2.5, 15.0]]])  # (1, 2, 2): on each segment
    out = point_to_polyline_distance(points, vertices)
    assert_close(out, torch.zeros(1, 2))


def test_point_to_polyline_distance_perpendicular_offset() -> None:
    vertices = torch.tensor([[[0.0, 0.0], [0.0, 10.0]]])  # single segment along +y
    points = torch.tensor([[[3.0, 5.0]]])  # 3m to the side, midway along the segment
    out = point_to_polyline_distance(points, vertices)
    assert_close(out, torch.tensor([[3.0]]))


def test_point_to_polyline_distance_clamps_past_endpoints() -> None:
    """A point beyond either endpoint measures to that endpoint, not to the
    segment's infinite extension."""
    vertices = torch.tensor([[[0.0, 0.0], [0.0, 10.0]]])
    points = torch.tensor([[[0.0, -3.0], [0.0, 13.0]]])  # short of start / past end
    out = point_to_polyline_distance(points, vertices)
    assert_close(out, torch.tensor([[3.0, 3.0]]))


def test_point_to_polyline_distance_picks_nearest_segment() -> None:
    """An L-shaped polyline: a point near the corner must measure to whichever
    of the two segments is actually closer, not just the first one."""
    vertices = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]])
    point_near_first_leg = torch.tensor([[[5.0, 1.0]]])
    point_near_second_leg = torch.tensor([[[9.0, 5.0]]])
    out = point_to_polyline_distance(
        torch.cat([point_near_first_leg, point_near_second_leg], dim=1), vertices
    )
    assert_close(out, torch.tensor([[1.0, 1.0]]))


def test_point_to_polyline_distance_batched() -> None:
    vertices = torch.tensor([[[0.0, 0.0], [0.0, 10.0]], [[0.0, 0.0], [10.0, 0.0]]])
    points = torch.tensor([[[2.0, 5.0]], [[0.0, 4.0]]])
    out = point_to_polyline_distance(points, vertices)
    assert_close(out, torch.tensor([[2.0], [4.0]]))


def test_route_convergence_signal_rewards_closing_the_gap() -> None:
    """Route straight ahead along +y; the car starts 3m to the side and ends
    up exactly on the route -> positive convergence."""
    route_waypoints = torch.tensor([[[0.0, 5.0], [0.0, 10.0]]])
    xy = torch.tensor([[[3.0, 0.0], [3.0, 5.0], [0.0, 10.0]]])
    signal = route_convergence_signal(xy, route_waypoints, scale=3.0)
    assert_close(signal, torch.tanh(torch.tensor([1.0])))  # convergence = 3, scale = 3


def test_route_convergence_signal_zero_when_staying_parallel() -> None:
    """Same constant offset from the route throughout -> no convergence."""
    route_waypoints = torch.tensor([[[0.0, 5.0], [0.0, 10.0]]])
    xy = torch.tensor([[[3.0, 0.0], [3.0, 5.0], [3.0, 10.0]]])
    signal = route_convergence_signal(xy, route_waypoints, scale=3.0)
    assert_close(signal, torch.tensor([0.0]))


def test_route_convergence_signal_clamps_divergence_to_zero() -> None:
    """Moving away from the route must sit at the same baseline (0) as
    staying parallel -- never a negative signal."""
    route_waypoints = torch.tensor([[[0.0, 5.0], [0.0, 10.0]]])
    xy = torch.tensor([[[1.0, 0.0], [3.0, 5.0], [5.0, 10.0]]])
    signal = route_convergence_signal(xy, route_waypoints, scale=3.0)
    assert_close(signal, torch.tensor([0.0]))


def test_route_convergence_signal_batched() -> None:
    route_waypoints = torch.tensor([
        [[0.0, 5.0], [0.0, 10.0]],
        [[0.0, 5.0], [0.0, 10.0]],
    ])
    xy = torch.tensor([
        [[3.0, 0.0], [3.0, 5.0], [0.0, 10.0]],  # converges
        [[3.0, 0.0], [3.0, 5.0], [3.0, 10.0]],  # parallel
    ])
    signal = route_convergence_signal(xy, route_waypoints, scale=3.0)
    assert_close(signal, torch.tensor([math.tanh(1.0), 0.0]))


def test_route_convergence_signal_saturates_for_large_convergence() -> None:
    near_one = 0.999
    route_waypoints = torch.tensor([[[0.0, 5.0], [0.0, 10.0]]])
    xy = torch.tensor([[[20.0, 0.0], [0.0, 10.0]]])  # 20m -> 0m: far past `scale`
    signal = route_convergence_signal(xy, route_waypoints, scale=1.0)
    assert signal.item() > near_one
