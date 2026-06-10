import json
from pathlib import Path

import torch
from torch.testing import assert_close

from rmind.components.objectives.action_transform import GaussianizeActionTransform

NUM_KNOTS = 65
HORIZON = 6


def _stats() -> dict:
    # Synthetic but realistic: longitudinal/steering ~ uniform on [-1, 1], so
    # the Gaussianize knots are an affine grid over the model channels.
    grid = torch.linspace(0.0, 1.0, NUM_KNOTS)
    knots = (grid * 2.0 - 1.0).expand(2, NUM_KNOTS)
    return {
        "action_keys": ["gas_pedal", "brake_pedal", "steering_angle"],
        "merge": True,
        "grid": grid.tolist(),
        "knots": knots.tolist(),
    }


def _stats_file(tmp_path: Path) -> str:
    path = tmp_path / "action_norm.json"
    path.write_text(json.dumps(_stats()))
    return str(path)


def _raw_chunks(batch_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    longitudinal = torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    steering = torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    gas = longitudinal.clamp_min(0.0)
    brake = (-longitudinal).clamp_min(0.0)
    return torch.stack([gas, brake, steering], dim=-1)  # (B, H, 3)


def test_roundtrip_is_identity(tmp_path: Path) -> None:
    transform = GaussianizeActionTransform.from_stats_file(_stats_file(tmp_path))
    raw = _raw_chunks()
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_roundtrip_with_stacked_sample_dim(tmp_path: Path) -> None:
    # _to_raw_space also runs on (N, B, H, A) sample stacks; the transform must
    # handle arbitrary leading dims (channel axis is -1).
    transform = GaussianizeActionTransform.from_stats_file(_stats_file(tmp_path))
    raw = torch.stack([_raw_chunks(), _raw_chunks()])  # (N, B, H, 3)
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_merge_dims_and_keys(tmp_path: Path) -> None:
    transform = GaussianizeActionTransform.from_stats_file(_stats_file(tmp_path))
    raw_dim, model_dim = 3, 2  # gas/brake/steering -> longitudinal/steering
    assert transform.raw_dim == raw_dim
    assert transform.model_dim == model_dim
    assert transform(_raw_chunks()).shape[-1] == model_dim
