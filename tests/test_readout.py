import pytest
import torch
from pydantic import ValidationError
from torch.testing import assert_close

from rmind.components.readout import AxisModeReadout, SingleReadout, axis_mode_anchor

K = 8  # draws per frame
H, A = 4, 2  # horizon, action channels (longitudinal, steering)
STEER = 1


def _bimodal_draws() -> torch.Tensor:
    # (B=1, K, H, A): steering (channel 1) splits into a dominant cluster of 5
    # draws (~ -0.8) and a minority of 3 (~ +0.8); longitudinal is shared.
    steering = torch.cat([torch.full((5, H), -0.8), torch.full((3, H), 0.8)])
    longitudinal = torch.full((K, H), 0.3)
    return torch.stack([longitudinal, steering], dim=-1).unsqueeze(0)  # (1, K, H, A)


def test_single_readout_takes_first_draw() -> None:
    readout = SingleReadout()
    assert readout.num_samples == 1
    draws = torch.randn(3, 1, H, A)
    assert_close(readout(draws), draws[:, 0])


def test_axis_mode_anchor_commits_to_dominant_cluster() -> None:
    # mean-of-K steering = (5*-0.8 + 3*0.8)/8 = -0.2; the dominant cluster is -0.8.
    draws = _bimodal_draws()
    anchor = axis_mode_anchor(draws, STEER)
    assert anchor.shape == (1, H, A)
    assert_close(anchor[0, :, STEER], torch.full((H,), -0.8))  # not the -0.2 mean
    assert_close(anchor[0, :, 0], torch.full((H,), 0.3))  # longitudinal preserved


def test_axis_mode_anchor_unimodal_falls_back_to_mean() -> None:
    # Steering spread (0.02) stays under gap_the, so no split -> plain mean-of-K.
    draws = torch.randn(2, 8, H, A) * 0.02
    assert_close(axis_mode_anchor(draws, STEER), draws.mean(dim=1))


def test_axis_mode_anchor_negative_axis_indexes_from_end() -> None:
    draws = _bimodal_draws()
    assert_close(axis_mode_anchor(draws, -1), axis_mode_anchor(draws, STEER))


def test_axis_mode_anchor_rejects_non_4d() -> None:
    with pytest.raises(ValueError, match="4-D"):
        axis_mode_anchor(torch.randn(8, H, A), STEER)


def test_axis_mode_anchor_rejects_out_of_range_axis() -> None:
    with pytest.raises(ValueError, match="out of range"):
        axis_mode_anchor(_bimodal_draws(), A)  # A == one past the last channel


def test_axis_mode_readout_matches_kernel() -> None:
    draws = _bimodal_draws()
    readout = AxisModeReadout(num_samples=K, mode_axis=-1)
    assert readout.num_samples == K
    assert_close(readout(draws), axis_mode_anchor(draws, -1))


def test_axis_mode_readout_requires_multiple_samples() -> None:
    with pytest.raises(ValidationError):
        AxisModeReadout(num_samples=1)
