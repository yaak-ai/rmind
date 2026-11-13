from types import SimpleNamespace

import pytest
import torch

from rmind.components.sample_weights import speed_change_flags, speed_change_weights

DT = 1.0 / 3.0


def _clips() -> torch.Tensor:
    cruise = torch.full((21,), 50.0)  # km/h, constant
    brake = torch.linspace(50.0, 20.0, 21)  # steady ~1.2 m/s^2
    hard = torch.cat([torch.full((10,), 50.0), torch.full((11,), 20.0)])  # step
    launch = torch.linspace(0.0, 30.0, 21)
    return torch.stack([cruise, cruise, cruise, brake, hard, launch])


def test_flags() -> None:
    f = speed_change_flags(_clips(), dt=DT, accel_threshold=2.0)
    assert f["accel"].tolist() == [False, False, False, False, True, False]
    assert f["stop_launch"].tolist() == [False, False, False, False, False, True]
    assert f["any"].tolist() == [False, False, False, False, True, True]


def test_weights_hit_target_fraction() -> None:
    ds = SimpleNamespace(data={"meta/VehicleMotion/speed": _clips()})
    w = speed_change_weights(ds, target_fraction=0.5)  # ty: ignore[invalid-argument-type]
    flagged = torch.tensor([False, False, False, False, True, True])
    assert float(w[flagged].sum() / w.sum()) == pytest.approx(0.5)


def test_weights_unchanged_when_share_already_reached() -> None:
    ds = SimpleNamespace(data={"meta/VehicleMotion/speed": _clips()})
    w = speed_change_weights(ds, target_fraction=0.2)  # ty: ignore[invalid-argument-type]
    assert torch.equal(w, torch.ones(6, dtype=torch.double))


def test_target_fraction_validated() -> None:
    ds = SimpleNamespace(data={"meta/VehicleMotion/speed": _clips()})
    with pytest.raises(ValueError, match="target_fraction"):
        speed_change_weights(ds, target_fraction=1.0)  # ty: ignore[invalid-argument-type]
