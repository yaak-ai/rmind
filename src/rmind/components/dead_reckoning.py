# Ported from feat/drivor:src/rmind/components/drivor/trajectory_target.py
# (commit 920030a, "fix: correct heading-convention mismatch in
# dead_reckon_future_trajectory"). Copied verbatim rather than re-derived: the
# heading-convention bug this fixes (GNSS-anchor drift 12.6m -> 2.05m) was
# real and easy to reintroduce -- see the convention note in
# `dead_reckon_future_trajectory`'s body below.
import torch
from torch import Tensor


def dead_reckon_future_trajectory(
    *,
    speed_kmh: Tensor,
    heading_deg: Tensor,
    time_stamp_s: Tensor,
    reference_index: int = 0,
) -> tuple[Tensor, Tensor]:
    """Dead-reckon a future ego-centric `(x, y, theta)` trajectory from
    CAN-bus speed and EKF/RTS-denoised heading, anchored at `reference_index`.

    `waypoints/xy_normalized` is NOT used here -- it is a route reference (see
    `rmind.models.drivor` module docstring), not the ego's realized future
    path. This integrates `speed`/`heading_deg` forward in time instead,
    using real per-step timestamps for `dt` (no assumption about uniform step
    spacing). Position and heading are expressed relative to the ego's own
    pose at `reference_index`, matching the ego-centric rotate/translate
    convention already used for `waypoints/xy_normalized` in
    `config/dataset/yaak/*.yaml` (`ST_Rotate(ST_Translate(geom, -ego_x,
    -ego_y), radians(heading))`) -- EXCEPT for that convention's `/100`
    position scale, which this function deliberately does NOT apply.
    `waypoints/xy_normalized` is a route reference (see module docstring
    above), a different quantity from this realized-path trajectory; the two
    sharing a scale was only ever a copy-paste artifact of this code's
    origin, not a real requirement. Position is plain meters here. (If a
    consumer needs xy and heading error balanced against each other -- e.g.
    a combined pose loss -- do that scaling explicitly at the loss/weighting
    layer, not by baking a scale into the target; see
    `rmind.components.loss.winner_takes_all_pose_l1`.)

    Args:
        speed_kmh: `(*batch, T)` CAN-bus speed, km/h.
        heading_deg: `(*batch, T)` EKF+RTS-denoised heading, degrees.
        time_stamp_s: `(*batch, T)` timestamps, float seconds.
        reference_index: index `t0` to anchor the ego frame at; poses are
            returned for `t0+1 .. T-1` (`P = T - 1 - t0` future poses).

    Returns:
        `(position, heading)`: `position` is `(*batch, P, 2)` ego-centric
        `(x, y)`, meters; `heading` is `(*batch, P)` ego-centric heading,
        radians, wrapped to `[-pi, pi]`.
    """
    t0 = reference_index

    speed_m_s = speed_kmh[..., t0:-1] / 3.6
    heading_rad = torch.deg2rad(heading_deg)
    ref_heading = heading_rad[..., t0 : t0 + 1]
    step_heading_rel = heading_rad[..., t0:-1] - ref_heading
    dt = time_stamp_s[..., t0 + 1 :] - time_stamp_s[..., t0:-1]

    # `heading_deg` is a compass bearing (0deg = north = world +y/northing,
    # 90deg = east = world +x/easting, clockwise) -- confirmed empirically
    # against real GNSS traces (see plan history). `gnss_anchor_drift_m` below
    # mirrors the production SQL's `ST_Rotate(_, radians(heading))`, which
    # maps that bearing convention's "forward" onto local +y (verified: R(+h)
    # applied to the compass-bearing forward vector (sin h, cos h) reduces to
    # exactly (0, 1)). So "forward" here must also be local +y, not +x --
    # using (cos, sin) instead (a plain math-angle convention) silently
    # rotates the dead-reckoned trajectory ~90deg out of alignment with the
    # QA check and with any other ego-centric convention in this codebase.
    dx = speed_m_s * torch.sin(step_heading_rel) * dt
    dy = speed_m_s * torch.cos(step_heading_rel) * dt
    position = torch.stack([dx, dy], dim=-1).cumsum(dim=-2)

    heading_rel = heading_rad[..., t0 + 1 :] - ref_heading
    heading_rel = torch.atan2(torch.sin(heading_rel), torch.cos(heading_rel))

    return position, heading_rel


def rolling_dead_reckoned_trajectory(
    *,
    speed_kmh: Tensor,
    heading_deg: Tensor,
    time_stamp_us: Tensor,
    episode_length: int,
    num_poses: int,
) -> Tensor:
    """Per-frame future ego-centric `(x, y, theta)` trajectory, dead-reckoned
    independently at every one of the first `episode_length` anchor frames --
    the auxiliary trajectory head's ground truth
    (`docs/phase3_trajectory_head_plan.md`).

    Implemented as a loop over anchors calling the tested single-anchor
    `dead_reckon_future_trajectory` above, not a reimplementation of the
    trig -- the axis-convention bug that function's docstring documents (and
    which has reportedly been rediscovered twice already) is easy to
    reintroduce in a second, independent implementation.

    Args:
        speed_kmh: `(*batch, T)` CAN-bus speed, km/h.
        heading_deg: `(*batch, T)` EKF/RTS-denoised heading, degrees.
        time_stamp_us: `(*batch, T)` raw timestamps, microseconds (e.g. a
            polars `Datetime[us]` column cast to its physical int64
            representation). Converted to float64 seconds here -- casting
            real Unix-epoch microsecond timestamps to float32 silently zeros
            every `dt` (float32's ~7 significant digits can't hold
            sub-second resolution at that magnitude); see
            `tests/test_dead_reckoning.py::test_dead_reckon_survives_float32_epoch_timestamps`.
        episode_length: number of independent anchor frames, `t0 = 0 ..
            episode_length - 1`.
        num_poses: number of future poses per anchor, `t0 + 1 .. t0 +
            num_poses`.

    Returns:
        `(*batch, episode_length, num_poses, 3)`: ego-centric `(x, y, theta)`
        per anchor frame, `(x, y)` meters, `theta` wrapped to `[-pi, pi]`.

    Raises:
        ValueError: if `T < episode_length + num_poses`, i.e. there aren't
            enough future steps to dead-reckon every anchor's full horizon.
    """
    *_batch, t = speed_kmh.shape
    needed = episode_length + num_poses
    if t < needed:
        msg = (
            f"need {needed} steps (episode_length + num_poses), got {t} -- "
            "increase clip_horizon"
        )
        raise ValueError(msg)

    time_stamp_s = time_stamp_us.double() / 1e6
    poses = []
    for t0 in range(episode_length):
        position, heading = dead_reckon_future_trajectory(
            speed_kmh=speed_kmh,
            heading_deg=heading_deg,
            time_stamp_s=time_stamp_s,
            reference_index=t0,
        )
        poses.append(
            torch.cat(
                [position[..., :num_poses, :], heading[..., :num_poses, None]], dim=-1
            ).float()
        )
    return torch.stack(poses, dim=-3)  # (*batch, episode_length, num_poses, 3)


def gnss_anchor_drift_m(
    *,
    dead_reckoned_position_m: Tensor,
    gnss_xy: Tensor,
    heading_deg: Tensor,
    reference_index: int = 0,
) -> Tensor:
    """QA-only: meters between the dead-reckoned final pose and the raw GNSS
    fix at the last window step, both expressed in the `reference_index` ego
    frame. Large values flag wheel slip, GPS multipath, or a heading-filter
    failure for that window -- see `rmind.models.drivor` verification notes.

    Args:
        dead_reckoned_position_m: `(*batch, P, 2)` meters, the `position`
            output of `dead_reckon_future_trajectory` for the same batch.
        gnss_xy: `(*batch, T, 2)` raw absolute GNSS position, meters (UTM).
        heading_deg: `(*batch, T)` heading, degrees, same tensor passed to
            `dead_reckon_future_trajectory`.
        reference_index: must match the `reference_index` used to produce
            `dead_reckoned_position_m`.

    Returns:
        `(*batch,)` drift, meters.
    """
    t0 = reference_index

    ref_xy = gnss_xy[..., t0 : t0 + 1, :]
    ref_heading = torch.deg2rad(heading_deg[..., t0 : t0 + 1])
    delta = gnss_xy[..., -1:, :] - ref_xy
    cos_h, sin_h = torch.cos(ref_heading), torch.sin(ref_heading)
    gnss_anchor_xy = torch.stack(
        [
            delta[..., 0] * cos_h - delta[..., 1] * sin_h,
            delta[..., 0] * sin_h + delta[..., 1] * cos_h,
        ],
        dim=-1,
    ).squeeze(-2)

    dead_reckoned_xy_m = dead_reckoned_position_m[..., -1, :]
    return (dead_reckoned_xy_m - gnss_anchor_xy).norm(dim=-1)
