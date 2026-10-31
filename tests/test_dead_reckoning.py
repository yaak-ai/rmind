# Ported from feat/drivor:tests/test_drivor.py (dead_reckon_future_trajectory /
# gnss_anchor_drift_m cases only) -- these are the existing proof that the
# compass-bearing convention is right; see `dead_reckoning.py` for the math.
import math

import pytest
import torch

from rmind.components.dead_reckoning import (
    bicycle_cleaned_heading,
    dead_reckon_future_trajectory,
    gnss_anchor_drift_m,
    rolling_dead_reckoned_trajectory,
)


def test_dead_reckon_constant_velocity() -> None:
    t = 5
    speed_kmh = torch.full((1, t), 36.0)  # 10 m/s
    heading_deg = torch.zeros(1, t)
    time_stamp_s = torch.arange(t, dtype=torch.float32).unsqueeze(0)  # 1s steps

    position, heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # heading is a compass bearing (0deg=north); forward at heading=0 is
    # local +y, not +x -- see dead_reckoning.py's convention note.
    expected_y = torch.tensor([10.0, 20.0, 30.0, 40.0])
    assert torch.allclose(position[0, :, 1], expected_y, atol=1e-4)
    assert torch.allclose(position[0, :, 0], torch.zeros(4), atol=1e-4)
    assert torch.allclose(heading[0], torch.zeros(4), atol=1e-6)


def test_dead_reckon_heading_change() -> None:
    speed_kmh = torch.full((1, 3), 36.0)  # 10 m/s
    heading_deg = torch.tensor([[0.0, 90.0, 90.0]])
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # heading is a compass bearing (0deg=north=local+y, 90deg=east=local+x).
    # interval 0->1 uses heading@t0=0deg (rel. to ref=0deg) -> +y; interval
    # 1->2 uses heading@t1=90deg (rel. to ref=0deg) -> +x.
    expected_position = torch.tensor([[0.0, 10.0], [10.0, 10.0]])
    assert torch.allclose(position[0], expected_position, atol=1e-4)

    expected_heading = torch.deg2rad(torch.tensor([90.0, 90.0]))
    assert torch.allclose(heading[0], expected_heading, atol=1e-4)


def test_gnss_anchor_drift_zero_when_consistent() -> None:
    speed_kmh = torch.full((1, 3), 36.0)  # 10 m/s
    heading_deg = torch.zeros(1, 3)
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # ego frame == world frame here (heading=0 throughout, identity rotation);
    # dead-reckoned final position is (0, 20) m (forward=local+y at heading=0),
    # so a consistent GNSS trace has the same endpoint.
    gnss_xy = torch.tensor([[[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]]])
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_m=position,
        gnss_xy=gnss_xy,
        heading_deg=heading_deg,
        reference_index=0,
    )
    assert torch.allclose(drift, torch.zeros(1), atol=1e-4)


def test_gnss_anchor_drift_nonzero_when_inconsistent() -> None:
    speed_kmh = torch.full((1, 3), 36.0)
    heading_deg = torch.zeros(1, 3)
    time_stamp_s = torch.tensor([[0.0, 1.0, 2.0]])

    position, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    gnss_xy = torch.tensor([
        [[0.0, 0.0], [0.0, 10.0], [0.0, 200.0]]
    ])  # last fix way off
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_m=position,
        gnss_xy=gnss_xy,
        heading_deg=heading_deg,
        reference_index=0,
    )
    assert drift.item() > 100.0  # noqa: PLR2004


def test_dead_reckon_survives_float32_epoch_timestamps() -> None:
    """Regression test: a caller that casts real Unix-epoch microsecond
    timestamps (~1.7e15) to float32 before calling this function silently
    zeroes every `dt` (float32's ~7 significant digits can't hold sub-second
    resolution at that magnitude), and therefore the whole dead-reckoned
    position -- found while wiring up
    `rmind.scripts.trajectory_action_controller`. `dead_reckon_future_trajectory`
    itself takes whatever dtype it's given, so this only passes if the caller
    keeps `time_stamp_s` in float64.
    """
    t = 4
    speed_kmh = torch.full((1, t), 36.0)  # 10 m/s
    heading_deg = torch.zeros(1, t)
    epoch_us = 1_673_444_902_908_844 + torch.arange(t) * 333_333  # ~3Hz ticks
    time_stamp_s = (epoch_us.double() / 1e6).unsqueeze(0)

    position, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_s=time_stamp_s,
        reference_index=0,
    )

    # ~10 m/s * 0.333s ~= 3.33m per tick, forward = local +y at heading=0.
    assert position[0, :, 1].min() > 0.02  # noqa: PLR2004 (would be ~0 under the bug)
    assert torch.allclose(
        position[0, :, 0], torch.zeros(3, dtype=position.dtype), atol=1e-4
    )


def test_dead_reckon_prefix_matches_shorter_horizon() -> None:
    """The first N poses of a long roll-out must be bit-for-bit the N-pose
    roll-out of the same anchor.

    The horizon sweep in `rmind.scripts.trajectory_action_controller` depends
    on this: it dead-reckons once at the longest horizon and slices, so that
    every horizon trains on one identical anchor set. Rebuilding the anchors
    per horizon would instead give the longer horizons a strictly more
    contiguous subset of each drive, confounding the comparison with
    survivorship. The property holds because position is a `cumsum` from the
    anchor and heading is differenced against it -- neither looks ahead.
    """
    generator = torch.Generator().manual_seed(0)
    long_horizon, short_horizon = 30, 6
    t = long_horizon + 1
    speed_kmh = torch.rand(4, t, generator=generator) * 130.0
    heading_deg = torch.rand(4, t, generator=generator) * 360.0
    time_stamp_s = torch.arange(t, dtype=torch.float64).unsqueeze(0).expand(4, t) / 3.0

    long_position, long_heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh, heading_deg=heading_deg, time_stamp_s=time_stamp_s
    )
    short_position, short_heading = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh[:, : short_horizon + 1],
        heading_deg=heading_deg[:, : short_horizon + 1],
        time_stamp_s=time_stamp_s[:, : short_horizon + 1],
    )

    assert long_position.shape[1] == long_horizon
    assert short_position.shape[1] == short_horizon
    torch.testing.assert_close(
        long_position[:, :short_horizon], short_position, rtol=0, atol=0
    )
    torch.testing.assert_close(
        long_heading[:, :short_horizon], short_heading, rtol=0, atol=0
    )


# --------------------------------------------------------------------------- #
# rolling_dead_reckoned_trajectory (docs/phase3_trajectory_head_plan.md step 1)
# --------------------------------------------------------------------------- #


def _make_series(
    *, seed: int = 0, t: int = 12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    speed_kmh = torch.rand(t, generator=g) * 80.0
    heading_deg = torch.rand(t, generator=g) * 720.0 - 360.0  # exercise wraparound
    dt_s = torch.rand(t - 1, generator=g) * 0.3 + 0.05
    time_stamp_s = torch.cat([torch.zeros(1, dtype=torch.float64), dt_s.cumsum(0)])
    time_stamp_us = (time_stamp_s * 1e6).long()
    return speed_kmh, heading_deg, time_stamp_us


def test_rolling_trajectory_shape() -> None:
    speed_kmh, heading_deg, time_stamp_us = _make_series(t=12)
    out = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=5,
        num_poses=6,
    )
    assert out.shape == (5, 6, 3)


def test_rolling_trajectory_raises_when_not_enough_future_context() -> None:
    speed_kmh, heading_deg, time_stamp_us = _make_series(t=10)
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


def test_rolling_trajectory_matches_dead_reckon_future_trajectory_at_every_anchor() -> (
    None
):
    """A plumbing check against the already-proven single-anchor primitive, not
    a fresh algebra proof -- `rolling_dead_reckoned_trajectory` is a loop over
    `dead_reckon_future_trajectory` (docs/phase3_trajectory_head_plan.md step
    1), so this pins that the wiring is correct at every anchor.
    """
    speed_kmh, heading_deg, time_stamp_us = _make_series(t=14)
    episode_length, num_poses = 6, 5

    rolling = rolling_dead_reckoned_trajectory(
        speed_kmh=speed_kmh,
        heading_deg=heading_deg,
        time_stamp_us=time_stamp_us,
        episode_length=episode_length,
        num_poses=num_poses,
    )

    time_stamp_s = time_stamp_us.double() / 1e6
    for t0 in range(episode_length):
        ref_position, ref_heading = dead_reckon_future_trajectory(
            speed_kmh=speed_kmh,
            heading_deg=heading_deg,
            time_stamp_s=time_stamp_s,
            reference_index=t0,
        )
        torch.testing.assert_close(rolling[t0, :, :2], ref_position[:num_poses].float())
        torch.testing.assert_close(rolling[t0, :, 2], ref_heading[:num_poses].float())


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


def _cleanup_inputs(
    speed_kmh: float, heading_deg: list[float], steering: float = 0.0
) -> dict[str, torch.Tensor]:
    t = len(heading_deg)
    return {
        "speed_kmh": torch.full((t,), speed_kmh, dtype=torch.float64),
        "heading_deg": torch.tensor(heading_deg, dtype=torch.float64),
        "steering": torch.full((t,), steering, dtype=torch.float64),
        "time_stamp_s": torch.arange(t, dtype=torch.float64) / 3,
    }


def test_bicycle_cleaned_heading_keeps_consistent_heading_at_speed() -> None:
    # 50 km/h, straight, heading drifts 0.5 deg/step: consistent with steering 0 -> kept
    inputs = _cleanup_inputs(50.0, [10.0, 10.5, 11.0, 11.5])
    assert torch.allclose(bicycle_cleaned_heading(**inputs), inputs["heading_deg"])


def test_bicycle_cleaned_heading_replaces_glitch_at_speed() -> None:
    # a 110 deg jump within one step at 80 km/h with steering 0 -> replaced (heading stays put)
    inputs = _cleanup_inputs(80.0, [10.0, 10.0, 120.0, 120.0])
    cleaned = bicycle_cleaned_heading(**inputs)
    assert torch.allclose(cleaned, torch.full((4,), 10.0, dtype=torch.float64))


def test_bicycle_cleaned_heading_uses_bicycle_at_low_speed() -> None:
    # 10 km/h, full right steering: heading follows v / L * tan(delta) * dt, filter ignored
    inputs = _cleanup_inputs(10.0, [0.0, 0.0, 0.0], steering=1.0)
    cleaned = bicycle_cleaned_heading(**inputs, wheelbase_m=2.7, road_wheel_deg_per_unit=30.0)
    step = math.degrees(10 / 3.6 / 2.7 * math.tan(math.radians(30.0)) / 3)
    assert torch.allclose(cleaned, torch.tensor([0.0, step, 2 * step], dtype=torch.float64))


def test_rolling_trajectory_with_steering_drops_heading_glitch() -> None:
    inputs = _cleanup_inputs(80.0, [0.0, 0.0, 120.0, 120.0, 0.0, 0.0])
    poses = rolling_dead_reckoned_trajectory(
        speed_kmh=inputs["speed_kmh"],
        heading_deg=inputs["heading_deg"],
        time_stamp_us=(inputs["time_stamp_s"] * 1e6).long(),
        steering=inputs["steering"],
        episode_length=1,
        num_poses=5,
    )
    assert torch.allclose(poses[0, :, 0], torch.zeros(5), atol=1e-5)  # straight ahead
    assert torch.allclose(poses[0, :, 2], torch.zeros(5), atol=1e-6)


def test_trapezoid_integration_matches_constant_deceleration() -> None:
    # 54 -> 18 km/h over 3 s in 1/3 s steps, straight: exact distance = mean speed * T = 10 m/s * 3 s
    t = torch.arange(10, dtype=torch.float64) / 3
    speed_kmh = 54.0 - 12.0 * t * 1.0  # -12 km/h per s
    heading = torch.zeros(10, dtype=torch.float64)
    left, _ = dead_reckon_future_trajectory(speed_kmh=speed_kmh, heading_deg=heading, time_stamp_s=t)
    trap, _ = dead_reckon_future_trajectory(
        speed_kmh=speed_kmh, heading_deg=heading, time_stamp_s=t, integration="trapezoid"
    )
    exact = float((speed_kmh[0] + speed_kmh[-1]) / 2 / 3.6 * t[-1])
    assert float(trap[-1, 1]) == pytest.approx(exact, abs=1e-9)
    # start-of-step overestimates distance when braking: by a * T * dt / 2 = 0.5 m here
    assert float(left[-1, 1]) - exact == pytest.approx(10 / 3 * 3 * (1 / 3) / 2, abs=1e-9)
