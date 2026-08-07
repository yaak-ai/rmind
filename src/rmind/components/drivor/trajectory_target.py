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
    -ego_y), radians(heading))`), including its `/100` position scale.

    Args:
        speed_kmh: `(*batch, T)` CAN-bus speed, km/h.
        heading_deg: `(*batch, T)` EKF+RTS-denoised heading, degrees.
        time_stamp_s: `(*batch, T)` timestamps, float seconds.
        reference_index: index `t0` to anchor the ego frame at; poses are
            returned for `t0+1 .. T-1` (`P = T - 1 - t0` future poses).

    Returns:
        `(position, heading)`: `position` is `(*batch, P, 2)` ego-centric
        `(x, y)`, `/100`-normalized; `heading` is `(*batch, P)` ego-centric
        heading, radians, wrapped to `[-pi, pi]`.
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
    position = torch.stack([dx, dy], dim=-1).cumsum(dim=-2) / 100.0

    heading_rel = heading_rad[..., t0 + 1 :] - ref_heading
    heading_rel = torch.atan2(torch.sin(heading_rel), torch.cos(heading_rel))

    return position, heading_rel


def gnss_anchor_drift_m(
    *,
    dead_reckoned_position_normalized: Tensor,
    gnss_xy: Tensor,
    heading_deg: Tensor,
    reference_index: int = 0,
) -> Tensor:
    """QA-only: meters between the dead-reckoned final pose and the raw GNSS
    fix at the last window step, both expressed in the `reference_index` ego
    frame. Large values flag wheel slip, GPS multipath, or a heading-filter
    failure for that window -- see `rmind.models.drivor` verification notes.

    Args:
        dead_reckoned_position_normalized: `(*batch, P, 2)`, the `position`
            output of `dead_reckon_future_trajectory` for the same batch.
        gnss_xy: `(*batch, T, 2)` raw absolute GNSS position, meters (UTM).
        heading_deg: `(*batch, T)` heading, degrees, same tensor passed to
            `dead_reckon_future_trajectory`.
        reference_index: must match the `reference_index` used to produce
            `dead_reckoned_position_normalized`.

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

    dead_reckoned_xy_m = dead_reckoned_position_normalized[..., -1, :] * 100.0
    return (dead_reckoned_xy_m - gnss_anchor_xy).norm(dim=-1)
