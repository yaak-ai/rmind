import torch
from torch import Tensor


def rolling_dead_reckoned_trajectory(  # noqa: PLR0914
    *,
    speed_kmh: Tensor,
    heading_deg: Tensor,
    time_stamp_us: Tensor,
    episode_length: int,
    num_poses: int,
) -> Tensor:
    """Dead-reckon a future ego-centric `(x, y, theta)` trajectory from CAN-bus
    speed and EKF/RTS-denoised heading, anchored INDEPENDENTLY at every one of
    the first `episode_length` frames -- the continuous analogue of
    `rmind.components.nn.ChunkFields`' per-frame action-chunk unfold, but for a
    dead-reckoned pose instead of a raw field.

    Ports the single-anchor `dead_reckon_future_trajectory` (DrivoR,
    arXiv:2601.05083, `feat/drivor` branch) to all `episode_length` anchors at
    once via a closed-form rewrite instead of looping per anchor:
    `dead_reckon_future_trajectory` integrates the RELATIVE heading
    `sin(h_i - h0)`/`cos(h_i - h0)` per anchor `h0`; the angle-difference
    identities expand that into exactly `R(h0) @ (world-frame displacement at
    step i)`, where `R(h0)` is the same 2D rotation the dataset SQL applies
    (`ST_Rotate(_, radians(heading))`). Since the world-frame displacement
    itself doesn't depend on the anchor, it only needs to be integrated ONCE
    into a single running position; every anchor's relative trajectory is then
    just `R(h[t0]) @ (pos_world[t] - pos_world[t0])`, read off with plain fancy
    indexing instead of a Python loop over anchors.

    `heading_deg` is a compass bearing (0deg = north = world +y, 90deg = east
    = world +x, clockwise), so "forward" is the local +y direction, i.e.
    `(sin h, cos h)` -- see `dead_reckon_future_trajectory`'s docstring for how
    this was confirmed against real GNSS traces.

    Args:
        speed_kmh: `(*batch, T)` CAN-bus speed, km/h.
        heading_deg: `(*batch, T)` EKF+RTS-denoised heading, degrees.
        time_stamp_us: `(*batch, T)` raw int64 microseconds-since-epoch
            timestamps (e.g. a polars `Datetime[us]` column cast to its
            physical int64 representation). Only consecutive differences are
            ever used, so no particular reference point matters -- the clip's
            own first step is subtracted before casting to float32 purely to
            stay precise at epoch scale (~1.67e15 us loses sub-second
            precision if cast directly).
        episode_length: number of independent anchor frames, `t0 = 0 ..
            episode_length - 1`.
        num_poses: number of future poses per anchor, `t0 + 1 .. t0 +
            num_poses`.

    Returns:
        `(*batch, episode_length, num_poses, 3)`: ego-centric `(x, y, theta)`
        per anchor frame, `(x, y)` `/100`-normalized (matching
        `waypoints/xy_normalized`'s scale convention), `theta` wrapped to
        `[-pi, pi]`.

    Raises:
        ValueError: if `T < episode_length + num_poses`, i.e. there aren't
            enough future steps to dead-reckon every anchor's full horizon.
    """
    *_batch, t = speed_kmh.shape
    needed = episode_length + num_poses
    if t < needed:
        msg = (
            f"need {needed} steps of speed/heading/timestamp to dead-reckon "
            f"{num_poses} future poses for {episode_length} anchor frames, got "
            f"{t} -- increase clip_horizon"
        )
        raise ValueError(msg)

    # int64 epoch microseconds lose sub-second precision if cast to float32
    # directly (~7 significant digits vs ~1.67e15 us) -- subtract the clip's
    # own first timestamp in exact int64 arithmetic first. Safe because only
    # consecutive differences (`dt` below) are ever used.
    time_stamp_s = (time_stamp_us - time_stamp_us[..., :1]).float() / 1e6
    heading_rad = torch.deg2rad(heading_deg)
    speed_m_s = speed_kmh / 3.6

    dt = time_stamp_s[..., 1:] - time_stamp_s[..., :-1]
    dx_world = speed_m_s[..., :-1] * torch.sin(heading_rad[..., :-1]) * dt
    dy_world = speed_m_s[..., :-1] * torch.cos(heading_rad[..., :-1]) * dt

    zero = dx_world.new_zeros(*dx_world.shape[:-1], 1)
    pos_world_x = torch.cat([zero, dx_world.cumsum(dim=-1)], dim=-1)  # (*batch, T)
    pos_world_y = torch.cat([zero, dy_world.cumsum(dim=-1)], dim=-1)

    anchor = torch.arange(episode_length, device=speed_kmh.device)
    target = anchor[:, None] + 1 + torch.arange(num_poses, device=speed_kmh.device)

    anchor_x = pos_world_x[..., anchor].unsqueeze(-1)  # (*batch, episode_length, 1)
    anchor_y = pos_world_y[..., anchor].unsqueeze(-1)
    anchor_h = heading_rad[..., anchor].unsqueeze(-1)

    target_x = pos_world_x[..., target]  # (*batch, episode_length, num_poses)
    target_y = pos_world_y[..., target]
    target_h = heading_rad[..., target]

    dx = target_x - anchor_x
    dy = target_y - anchor_y
    cos_h0, sin_h0 = torch.cos(anchor_h), torch.sin(anchor_h)
    x_rel = (dx * cos_h0 - dy * sin_h0) / 100.0
    y_rel = (dy * cos_h0 + dx * sin_h0) / 100.0

    heading_rel = target_h - anchor_h
    heading_rel = torch.atan2(torch.sin(heading_rel), torch.cos(heading_rel))

    return torch.stack([x_rel, y_rel, heading_rel], dim=-1)
