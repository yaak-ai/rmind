import pytest
import torch

from rmind.components.controller import (
    NUM_TURN_SIGNAL_CLASSES,
    LaggedLongitudinalDynamicsParams,
    LateralDynamicsParams,
    LongitudinalDynamicsParams,
    TrajectoryToActionMLP,
    build_features,
    fit_lagged_longitudinal_dynamics,
    fit_lateral_dynamics,
    fit_longitudinal_dynamics,
    invert_lagged_longitudinal_dynamics,
    invert_lateral_dynamics,
    invert_longitudinal_dynamics,
)

_TOL = 1e-3


def test_fit_longitudinal_dynamics_recovers_known_params() -> None:
    """Synthetic data generated from a known (gas_gain, brake_gain,
    drag_coeff, offset) should be recovered exactly (up to float precision)
    by OLS, since the generating process is noise-free and linear in the
    fitted parameters."""
    generator = torch.Generator().manual_seed(0)
    n = 500
    true = LongitudinalDynamicsParams(
        gas_gain=3.0, brake_gain=5.0, drag_coeff=0.02, offset=0.1
    )
    gas = torch.rand(n, generator=generator)
    brake = torch.rand(n, generator=generator) * (
        torch.rand(n, generator=generator) < 0.3  # noqa: PLR2004
    )
    gas *= brake == 0  # never both pedals at once, matching real driving
    speed = torch.rand(n, generator=generator) * 130.0
    dv = (
        true.gas_gain * gas
        - true.brake_gain * brake
        - true.drag_coeff * speed
        - true.offset
    )

    fitted = fit_longitudinal_dynamics(gas=gas, brake=brake, speed=speed, dv=dv)

    assert abs(fitted.gas_gain - true.gas_gain) < _TOL
    assert abs(fitted.brake_gain - true.brake_gain) < _TOL
    assert abs(fitted.drag_coeff - true.drag_coeff) < _TOL
    assert abs(fitted.offset - true.offset) < _TOL


def test_invert_longitudinal_dynamics_round_trip() -> None:
    """Inverting the same model that generated `target_dv` should reproduce
    the pedal that was actually used to hit it (single-pedal case: gas OR
    brake, never both, which is what the inverse always returns)."""
    params = LongitudinalDynamicsParams(
        gas_gain=3.0, brake_gain=5.0, drag_coeff=0.02, offset=0.1
    )
    speed = torch.tensor([0.0, 20.0, 60.0, 100.0])
    gas_used = torch.tensor([0.2, 0.5, 0.1, 0.0])
    brake_used = torch.tensor([0.0, 0.0, 0.0, 0.3])
    dv = (
        params.gas_gain * gas_used
        - params.brake_gain * brake_used
        - params.drag_coeff * speed
        - params.offset
    )

    gas_pred, brake_pred = invert_longitudinal_dynamics(
        target_dv=dv, speed=speed, params=params
    )

    assert torch.allclose(gas_pred, gas_used, atol=1e-4)
    assert torch.allclose(brake_pred, brake_used, atol=1e-4)


def test_invert_longitudinal_dynamics_clamps_to_valid_pedal_range() -> None:
    params = LongitudinalDynamicsParams(
        gas_gain=1.0, brake_gain=1.0, drag_coeff=0.0, offset=0.0
    )
    speed = torch.zeros(2)

    gas, brake = invert_longitudinal_dynamics(
        target_dv=torch.tensor([50.0, -50.0]), speed=speed, params=params
    )

    assert torch.equal(gas, torch.tensor([1.0, 0.0]))
    assert torch.equal(brake, torch.tensor([0.0, 1.0]))


def test_fit_lateral_dynamics_recovers_known_params() -> None:
    generator = torch.Generator().manual_seed(1)
    n = 500
    true = LateralDynamicsParams(steer_gain=0.05, camber_bias=0.002)
    steering_angle = torch.rand(n, generator=generator) * 2.0 - 1.0
    speed = torch.rand(n, generator=generator) * 130.0
    dheading = true.steer_gain * (steering_angle * speed) + true.camber_bias

    fitted = fit_lateral_dynamics(
        steering_angle=steering_angle, speed=speed, dheading=dheading
    )

    assert abs(fitted.steer_gain - true.steer_gain) < _TOL
    assert abs(fitted.camber_bias - true.camber_bias) < _TOL


def test_invert_lateral_dynamics_round_trip() -> None:
    params = LateralDynamicsParams(steer_gain=0.05, camber_bias=0.002)
    speed = torch.tensor([20.0, 60.0, 100.0])
    steering_used = torch.tensor([0.3, -0.1, 0.05])
    dheading = params.steer_gain * (steering_used * speed) + params.camber_bias

    steering_pred = invert_lateral_dynamics(
        target_dheading=dheading, speed=speed, params=params
    )

    assert torch.allclose(steering_pred, steering_used, atol=1e-4)


def test_invert_lateral_dynamics_returns_zero_below_min_speed() -> None:
    """The model itself says steering has no effect at a standstill -- the
    inverse must not divide by a near-zero denominator and blow up."""
    params = LateralDynamicsParams(steer_gain=0.05, camber_bias=0.0)
    speed = torch.tensor([0.0, 0.5, 60.0])

    steering_pred = invert_lateral_dynamics(
        target_dheading=torch.full((3,), 0.1), speed=speed, params=params, min_speed=1.0
    )

    assert torch.equal(steering_pred[:2], torch.zeros(2))
    assert steering_pred[2] != 0


def test_invert_lateral_dynamics_clamps_to_valid_steering_range() -> None:
    params = LateralDynamicsParams(steer_gain=1.0, camber_bias=0.0)
    speed = torch.tensor([10.0, 10.0])

    steering_pred = invert_lateral_dynamics(
        target_dheading=torch.tensor([50.0, -50.0]), speed=speed, params=params
    )

    assert torch.equal(steering_pred, torch.tensor([1.0, -1.0]))


def test_fit_lagged_longitudinal_dynamics_recovers_known_params() -> None:
    generator = torch.Generator().manual_seed(2)
    n = 800
    true_gas_gains = (2.0, 0.8, 0.3)  # current tick, 1-tick-lag, 2-tick-lag
    true_brake_gains = (4.0, 1.5, 0.5)  # must match gas_lags' lag count (see docstring)
    true_drag, true_offset = 0.02, 0.05

    gas_lags = torch.rand(n, len(true_gas_gains), generator=generator)
    brake_lags = torch.rand(n, len(true_brake_gains), generator=generator) * 0.2
    speed = torch.rand(n, generator=generator) * 130.0
    dv = (
        (gas_lags * torch.tensor(true_gas_gains)).sum(-1)
        - (brake_lags * torch.tensor(true_brake_gains)).sum(-1)
        - true_drag * speed
        - true_offset
    )

    fitted = fit_lagged_longitudinal_dynamics(
        gas_lags=gas_lags, brake_lags=brake_lags, speed=speed, dv=dv
    )

    for got, want in zip(fitted.gas_gains, true_gas_gains, strict=True):
        assert abs(got - want) < _TOL
    for got, want in zip(fitted.brake_gains, true_brake_gains, strict=True):
        assert abs(got - want) < _TOL
    assert abs(fitted.drag_coeff - true_drag) < _TOL
    assert abs(fitted.offset - true_offset) < _TOL


def test_lagged_longitudinal_dynamics_num_lags_1_matches_plain_model() -> None:
    """A single-lag-tap fit/invert should reduce exactly to the plain
    (unlagged) model -- same numbers, just through the generalized API."""
    generator = torch.Generator().manual_seed(3)
    n = 300
    gas = torch.rand(n, generator=generator)
    brake = torch.rand(n, generator=generator) * (
        brake_mask := (torch.rand(n, generator=generator) < 0.3)  # noqa: PLR2004
    )
    gas *= ~brake_mask
    speed = torch.rand(n, generator=generator) * 130.0
    true = LongitudinalDynamicsParams(
        gas_gain=3.0, brake_gain=5.0, drag_coeff=0.02, offset=0.1
    )
    dv = (
        true.gas_gain * gas
        - true.brake_gain * brake
        - true.drag_coeff * speed
        - true.offset
    )

    plain = fit_longitudinal_dynamics(gas=gas, brake=brake, speed=speed, dv=dv)
    lagged = fit_lagged_longitudinal_dynamics(
        gas_lags=gas.unsqueeze(-1), brake_lags=brake.unsqueeze(-1), speed=speed, dv=dv
    )
    assert abs(lagged.gas_gains[0] - plain.gas_gain) < _TOL
    assert abs(lagged.brake_gains[0] - plain.brake_gain) < _TOL

    empty = torch.empty(n, 0)
    gas_pred, brake_pred = invert_lagged_longitudinal_dynamics(
        target_dv=dv, speed=speed, past_gas=empty, past_brake=empty, params=lagged
    )
    plain_gas_pred, plain_brake_pred = invert_longitudinal_dynamics(
        target_dv=dv, speed=speed, params=plain
    )
    assert torch.allclose(gas_pred, plain_gas_pred, atol=1e-3)
    assert torch.allclose(brake_pred, plain_brake_pred, atol=1e-3)


def test_invert_lagged_longitudinal_dynamics_nets_out_past_pedal_contribution() -> None:
    params = LaggedLongitudinalDynamicsParams(
        gas_gains=(2.0, 1.0), brake_gains=(4.0, 0.0), drag_coeff=0.0, offset=0.0
    )
    speed = torch.tensor([0.0])
    past_gas = torch.tensor([
        [0.5]
    ])  # gas was 0.5 one tick ago -> contributes 1.0*0.5=0.5 to dv
    past_brake = torch.tensor([[0.0]])
    # want total dv=1.5; past tick already contributes 0.5, so current gas must supply 1.0 -> gas=0.5
    target_dv = torch.tensor([1.5])

    gas_pred, brake_pred = invert_lagged_longitudinal_dynamics(
        target_dv=target_dv,
        speed=speed,
        past_gas=past_gas,
        past_brake=past_brake,
        params=params,
    )

    assert torch.allclose(gas_pred, torch.tensor([0.5]), atol=1e-4)
    assert torch.allclose(brake_pred, torch.tensor([0.0]), atol=1e-4)


def test_action_steps_1_matches_the_single_action_shapes() -> None:
    """`action_steps=1` must stay a drop-in for the pre-sequence model:
    same head weight shapes (so old checkpoints load) and same output shapes
    (so `eval` and the reporting helpers are unchanged)."""
    model = TrajectoryToActionMLP(in_features=19, hidden_size=8)

    out = model(torch.zeros(4, 19))

    assert model.continuous_head.out_features == len(model.continuous_fields) * 2
    assert model.turn_signal_head.out_features == NUM_TURN_SIGNAL_CLASSES
    assert out["turn_signal"].shape == (4, NUM_TURN_SIGNAL_CLASSES)
    for field in model.continuous_fields:
        assert out["continuous"][field].shape == (4, 2)


def test_action_steps_emits_a_control_sequence() -> None:
    """Beyond one step the model emits `(mean, logvar)` per field per step
    and class logits per step -- the open-loop control sequence a
    receding-horizon controller would take index 0 of."""
    model = TrajectoryToActionMLP(in_features=91, hidden_size=8, action_steps=6)

    out = model(torch.zeros(4, 91))

    assert out["turn_signal"].shape == (4, 6, NUM_TURN_SIGNAL_CLASSES)
    for field in model.continuous_fields:
        assert out["continuous"][field].shape == (4, 6, 2)


def test_action_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="action_steps must be >= 1"):
        TrajectoryToActionMLP(in_features=19, action_steps=0)


def test_build_features_width_tracks_the_trajectory_horizon() -> None:
    """`in_features` for a horizon-N controller is `N * 3 + 1` (N poses x
    (x, y, heading), plus current speed) -- the arithmetic the horizon sweep
    relies on when it slices a longer trajectory down."""
    for horizon in (6, 15, 30):
        features = build_features(
            position=torch.zeros(4, horizon, 2),
            heading=torch.zeros(4, horizon),
            speed=torch.zeros(4),
        )
        assert features.shape == (4, horizon * 3 + 1)
