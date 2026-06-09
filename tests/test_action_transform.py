import json
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from rmind.components.objectives import FlowPolicyObjective
from rmind.components.objectives.action_transform import (
    GaussianizeActionTransform,
    dct_basis,
)
from rmind.components.transformer import FlowActionDecoder

NUM_KNOTS = 65
HORIZON = 6
DELTA_WEIGHT = 10.0


def _stats(*, dct: bool) -> dict:
    # Synthetic but realistic: longitudinal/steering ~ uniform on [-1, 1], so
    # the Gaussianize knots are an affine grid; DCT sigmas decay like the
    # measured spectrum (sigma_0 ~ sqrt(H), higher coefficients small).
    grid = torch.linspace(0.0, 1.0, NUM_KNOTS)
    knots = (grid * 2.0 - 1.0).expand(2, NUM_KNOTS)
    stats = {
        "action_keys": ["gas_pedal", "brake_pedal", "steering_angle"],
        "merge": True,
        "grid": grid.tolist(),
        "knots": knots.tolist(),
    }
    if dct:
        stats["dct"] = {
            "mu": torch.zeros(HORIZON, 2).tolist(),
            "sigma": [
                [2.47, 2.30],
                [0.48, 0.72],
                [0.20, 0.36],
                [0.12, 0.23],
                [0.09, 0.17],
                [0.08, 0.15],
            ],
            "sigma_floor_frac": 0.05,
        }
    return stats


def _stats_file(tmp_path: Path, *, dct: bool) -> str:
    path = tmp_path / "action_norm.json"
    path.write_text(json.dumps(_stats(dct=dct)))
    return str(path)


def _raw_chunks(batch_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    longitudinal = (
        torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    )
    steering = torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    gas = longitudinal.clamp_min(0.0)
    brake = (-longitudinal).clamp_min(0.0)
    return torch.stack([gas, brake, steering], dim=-1)  # (B, H, 3)


def test_dct_basis_is_orthonormal() -> None:
    basis = dct_basis(HORIZON)
    assert_close(basis @ basis.T, torch.eye(HORIZON), atol=1e-6, rtol=0)


def test_dct_basis_isolates_constant_chunk_in_k0() -> None:
    coeff = dct_basis(HORIZON) @ torch.ones(HORIZON)
    assert coeff[0] > 0
    assert_close(coeff[1:], torch.zeros(HORIZON - 1), atol=1e-6, rtol=0)


@pytest.mark.parametrize("dct", [False, True])
def test_roundtrip_is_identity(tmp_path: Path, *, dct: bool) -> None:
    transform = GaussianizeActionTransform.from_stats_file(
        _stats_file(tmp_path, dct=dct)
    )
    assert transform.mixes_horizon is dct
    raw = _raw_chunks()
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_dct_roundtrip_with_stacked_sample_dim(tmp_path: Path) -> None:
    # _to_raw_space also runs on (N, B, H, A) sample stacks; the DCT must hit
    # the horizon axis (-2) regardless of leading dims.
    transform = GaussianizeActionTransform.from_stats_file(
        _stats_file(tmp_path, dct=True)
    )
    raw = torch.stack([_raw_chunks(), _raw_chunks()])  # (N, B, H, 3)
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_dct_sigma_floor_clamps_small_coefficients(tmp_path: Path) -> None:
    transform = GaussianizeActionTransform.from_stats_file(
        _stats_file(tmp_path, dct=True)
    )
    sigma = torch.tensor(_stats(dct=True)["dct"]["sigma"])
    floor = 0.05 * sigma[0]
    assert_close(transform.dct_sigma_eff, sigma.clamp_min(floor), atol=0, rtol=0)


def _decoder(*, action_horizon: int = HORIZON) -> FlowActionDecoder:
    return FlowActionDecoder(
        condition_dim=32,
        dim_model=32,
        action_dim=2,
        action_horizon=action_horizon,
        num_layers=1,
        num_heads=2,
        flow_sampling_steps=2,
    )


def test_flow_policy_disables_chunk_delta_with_dct_transform(
    tmp_path: Path,
) -> None:
    objective = FlowPolicyObjective(
        decoder=_decoder(),
        loss=torch.nn.MSELoss(),
        chunk_delta_weight=DELTA_WEIGHT,
        action_transform_stats=_stats_file(tmp_path, dct=True),
    )
    assert objective.chunk_delta_weight == pytest.approx(0.0)


def test_flow_policy_keeps_chunk_delta_without_horizon_mixing(
    tmp_path: Path,
) -> None:
    objective = FlowPolicyObjective(
        decoder=_decoder(),
        loss=torch.nn.MSELoss(),
        chunk_delta_weight=DELTA_WEIGHT,
        action_transform_stats=_stats_file(tmp_path, dct=False),
    )
    assert objective.chunk_delta_weight == pytest.approx(DELTA_WEIGHT)


def test_flow_policy_rejects_dct_horizon_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dct horizon"):
        FlowPolicyObjective(
            decoder=_decoder(action_horizon=12),
            loss=torch.nn.MSELoss(),
            action_transform_stats=_stats_file(tmp_path, dct=True),
        )
