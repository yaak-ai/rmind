"""Counterfactual conditioning probes for a goal/speed-conditioned trajectory
policy: how much the plan moves when ONLY the goal (route) or ONLY the speed
input is swapped, with the camera frames held fixed.

Two failure modes motivated these (rsim round 23, `19wag98x:v51`): goal use
falls as goal dropout rises (wide turns, off-route collisions), and at
standstill the plan copies the speed input (`plan ~= speed * horizon`)
instead of reading the scene (stalls). Both are measurable offline by exact
replay, so they are logged as validation metrics rather than left to CARLA.

Frame convention (shared by `waypoints/xy_normalized` and the dead-reckoned
trajectory target, see `rmind.components.dead_reckoning`): ego-centric,
`x` = right, `y` = forward; heading is a compass delta, positive = clockwise,
so a LEFT turn has negative plan-end heading.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

# route-matched waypoint sampling (rmind #261, and what rsim feeds): 10 points
# at 1, 11, ..., 91 m along the path
WAYPOINT_ARC_LENGTHS_M: tuple[float, ...] = tuple(float(s) for s in range(1, 92, 10))

# (waypoints `(b, t, n, 2)`, speed `(b, t, 1)` km/h) -> the probability-weighted
# plan at each clip's NEWEST frame, `(b, num_poses, 3)`
type PlanFn = Callable[[Tensor, Tensor], Tensor]


def synthetic_turn_waypoints(
    *,
    left: bool,
    radius_m: float = 10.0,
    straight_m: float = 2.0,
    arc_lengths_m: tuple[float, ...] = WAYPOINT_ARC_LENGTHS_M,
    scale: float = 100.0,
) -> Tensor:
    """A 90-degree turn route `(n, 2)` in the `waypoints/xy_normalized` frame.

    Straight ahead for `straight_m`, a quarter circle of `radius_m`, then
    straight along the exit direction; sampled at `arc_lengths_m` along the
    path and divided by `scale` (the dataset's `/100`).
    """
    s = torch.tensor(arc_lengths_m, dtype=torch.float64)
    arc = math.pi / 2 * radius_m
    u = s - straight_m  # distance travelled past the turn entry
    phi = u.clamp(0.0, arc) / radius_m
    x = radius_m * (1.0 - torch.cos(phi)) + (u - arc).clamp(min=0.0)
    y = s.clamp(max=straight_m) + radius_m * torch.sin(phi)
    if left:
        x = -x
    return (torch.stack([x, y], dim=-1) / scale).float()


def expected_plan(trajectory: Tensor, mode_logits: Tensor | None = None) -> Tensor:
    """Probability-weighted trajectory `(..., p, 3)` from hypotheses `(..., q, p, 3)`.

    Softmax over `mode_logits` `(..., q)` (what rsim serves); a uniform mean
    over hypotheses when there is no mode head.
    """
    if mode_logits is None:
        return trajectory.mean(dim=-3)
    weights = mode_logits.float().softmax(dim=-1).to(trajectory.dtype)
    return torch.einsum("...q,...qpc->...pc", weights, trajectory)


def goal_sensitivity(
    plan_fn: PlanFn, waypoints: Tensor, speed: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-clip `(yaw_deg, lateral_m)` plan-end difference, left-turn minus
    right-turn route, with every frame's waypoints replaced by the synthetic
    route. Both are 0 for a policy that ignores its goal; a goal-following
    policy gives negative values (left is negative in this frame).
    """
    plans = [
        plan_fn(
            synthetic_turn_waypoints(left=left)
            .to(waypoints)
            .expand_as(waypoints)
            .contiguous(),
            speed,
        )[:, -1]
        for left in (True, False)
    ]
    delta = plans[0] - plans[1]  # (b, 3)
    return torch.rad2deg(delta[:, 2]), delta[:, 0]


def speed_echo(
    plan_fn: PlanFn, waypoints: Tensor, speed: Tensor, *, probe_kmh: float = 10.0
) -> Tensor:
    """Per-clip plan-end forward distance (m) with the speed input on every
    frame replaced by `probe_kmh`. A pure speed copier gives `probe_kmh / 3.6
    * horizon` (~4.6 m at 10 km/h over 5 x 1/3 s); a policy that reads the
    scene stays near its real-input plan.
    """
    return plan_fn(waypoints, torch.full_like(speed, probe_kmh))[:, -1, 1]
