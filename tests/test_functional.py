import math

import torch
from torch.testing import assert_close

from rmind.utils.functional import (
    build_local_trajectory,
    build_relative_trajectory,
    compose_relative_trajectory,
    diff_last,
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
