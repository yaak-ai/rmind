import torch
from torch.testing import assert_close, make_tensor

from rmind.components.nn import AxisShrinkage

STEPS, AXES, FORK = 50, 3, 2


def _mod(init: float = 0.05, **kw: object) -> AxisShrinkage:
    return AxisShrinkage(
        num_steps=STEPS, num_axes=AXES, axes=(FORK,), init_threshold=init, **kw
    )


def test_small_values_on_the_named_axis_become_exactly_zero(
    device: torch.device,
) -> None:
    m = _mod().to(device)
    x = torch.zeros(1, STEPS * AXES, device=device)
    x.reshape(1, STEPS, AXES)[0, :, FORK] = 0.02  # inside the +/-0.05 dead zone

    out = m(x).reshape(1, STEPS, AXES)

    # exact equality is the POINT: the whole feature is that the dead zone decodes to
    # exactly 0.0, which is what a continuous head cannot do
    assert (out[0, :, FORK] == 0.0).all()  # noqa: RUF069


def test_other_axes_are_untouched(device: torch.device) -> None:
    # traction and steering have no zero atom; shrinking them would only bias them
    m = _mod().to(device)
    x = make_tensor(
        4, STEPS * AXES, dtype=torch.float, device=device, low=-0.04, high=0.04
    )

    out = m(x).reshape(4, STEPS, AXES)
    ref = x.reshape(4, STEPS, AXES)

    assert_close(out[:, :, 0], ref[:, :, 0])
    assert_close(out[:, :, 1], ref[:, :, 1])
    assert (out[:, :, FORK] == 0.0).all()  # noqa: RUF069


def test_events_survive_shifted_by_tau(device: torch.device) -> None:
    m = _mod().to(device)
    x = torch.zeros(1, STEPS * AXES, device=device)
    x.reshape(1, STEPS, AXES)[0, 10, FORK] = 0.9
    x.reshape(1, STEPS, AXES)[0, 11, FORK] = -0.9

    out = m(x).reshape(1, STEPS, AXES)

    assert_close(
        out[0, 10, FORK], torch.tensor(0.85, device=device), atol=1e-5, rtol=1e-4
    )
    assert_close(
        out[0, 11, FORK], torch.tensor(-0.85, device=device), atol=1e-5, rtol=1e-4
    )


def test_threshold_is_positive_and_learnable(device: torch.device) -> None:
    m = _mod(init=0.05).to(device)

    assert_close(
        m.thresholds, torch.tensor([0.05], device=device), atol=1e-5, rtol=1e-3
    )

    x = make_tensor(4, STEPS * AXES, dtype=torch.float, device=device, low=-1, high=1)
    m(x).pow(2).mean().backward()

    assert m.raw_threshold.grad is not None
    assert m.raw_threshold.grad.abs().sum() > 0
    # softplus keeps tau > 0 however far the parameter is driven
    with torch.no_grad():
        m.raw_threshold.fill_(-50.0)
    assert (m.thresholds > 0).all()


def test_rejects_an_out_of_range_axis() -> None:
    try:
        AxisShrinkage(num_steps=STEPS, num_axes=AXES, axes=(7,))
    except ValueError:
        return
    msg = "expected ValueError for an out-of-range axis"
    raise AssertionError(msg)
