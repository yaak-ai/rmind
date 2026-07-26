import json
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from rmind.components.action_transform import (
    MERGE_MODEL_KEYS,
    ActionTransform,
    GasBrakeMerge,
    Gaussianize,
    build_action_transform,
)

NUM_KNOTS = 65
HORIZON = 6


def _stats(*, merge: bool = True) -> dict:
    # Legacy on-disk format (action_keys + merge flag, no model_keys): the loader
    # must still derive the channel space from it. Synthetic but realistic:
    # channels ~ uniform on [-1, 1], so the Gaussianize knots are an affine grid.
    grid = torch.linspace(0.0, 1.0, NUM_KNOTS)
    channels = len(MERGE_MODEL_KEYS) if merge else 3
    knots = (grid * 2.0 - 1.0).expand(channels, NUM_KNOTS)
    return {
        "action_keys": ["gas_pedal", "brake_pedal", "steering_angle"],
        "merge": merge,
        "grid": grid.tolist(),
        "knots": knots.tolist(),
    }


def _stats_file(tmp_path: Path, *, merge: bool = True) -> str:
    path = tmp_path / "action_norm.json"
    path.write_text(json.dumps(_stats(merge=merge)))
    return str(path)


def _raw_chunks(batch_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    longitudinal = torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    steering = torch.rand(batch_size, HORIZON, generator=generator) * 1.8 - 0.9
    gas = longitudinal.clamp_min(0.0)
    brake = (-longitudinal).clamp_min(0.0)
    return torch.stack([gas, brake, steering], dim=-1)  # (B, H, 3); gas*brake == 0


def test_rejects_non_uniform_grid() -> None:
    # the inverse maps u -> index via u*(K-1), exact only for a uniform grid;
    # a non-uniform grid must be rejected at construction rather than silently
    # producing a wrong inverse.
    grid = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0])  # non-uniform
    knots = (grid * 2.0 - 1.0).expand(2, grid.shape[0])
    with pytest.raises(ValueError, match="uniform"):
        Gaussianize(grid=grid, knots=knots, model_keys=MERGE_MODEL_KEYS)


def test_legacy_file_derives_model_keys(tmp_path: Path) -> None:
    # Files fit before the split stored action_keys + merge, not model_keys.
    gaussianize = Gaussianize.from_stats_file(_stats_file(tmp_path, merge=True))
    assert gaussianize.model_keys == MERGE_MODEL_KEYS
    assert gaussianize.model_dim == len(MERGE_MODEL_KEYS)


def test_roundtrip_is_identity(tmp_path: Path) -> None:
    transform = build_action_transform(merge=True, stats_path=_stats_file(tmp_path))
    assert transform is not None
    raw = _raw_chunks()
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_roundtrip_with_stacked_sample_dim(tmp_path: Path) -> None:
    # _to_raw_space also runs on (N, B, H, A) sample stacks; the transform must
    # handle arbitrary leading dims (channel axis is -1).
    transform = build_action_transform(merge=True, stats_path=_stats_file(tmp_path))
    assert transform is not None
    raw = torch.stack([_raw_chunks(), _raw_chunks()])  # (N, B, H, 3)
    recovered = transform.inverse(transform(raw))
    assert_close(recovered, raw, atol=1e-4, rtol=0)


def test_merge_dims_and_keys(tmp_path: Path) -> None:
    transform = build_action_transform(merge=True, stats_path=_stats_file(tmp_path))
    assert transform is not None
    raw_dim, model_dim = 3, 2  # gas/brake/steering -> longitudinal/steering
    assert transform.raw_dim == raw_dim
    assert transform.model_dim == model_dim
    assert transform.action_keys == ("gas_pedal", "brake_pedal", "steering_angle")
    assert transform.model_action_keys == MERGE_MODEL_KEYS
    assert transform(_raw_chunks()).shape[-1] == model_dim


def test_merge_only_needs_no_file() -> None:
    # The structural merge is file-less: usable on its own (e.g. a "merge but no
    # warp" ablation) straight from config.
    transform = build_action_transform(merge=True, stats_path=None)
    assert transform is not None
    assert isinstance(transform.merge, GasBrakeMerge)
    assert transform.gaussianize is None
    assert (transform.raw_dim, transform.model_dim) == (3, 2)
    raw = _raw_chunks()
    # physical_model == forward (no warp), and the split inverts exactly.
    assert_close(transform(raw), transform.physical_model(raw), atol=0, rtol=0)
    assert_close(transform.inverse(transform(raw)), raw, atol=1e-6, rtol=0)


def test_gaussianize_only_passes_raw_through(tmp_path: Path) -> None:
    transform = build_action_transform(
        merge=False, stats_path=_stats_file(tmp_path, merge=False)
    )
    assert transform is not None
    assert transform.merge is None
    assert (transform.raw_dim, transform.model_dim) == (3, 3)
    # no merge => physical model space is the raw space.
    raw = torch.rand(4, HORIZON, 3) * 1.6 - 0.8
    assert_close(transform.physical_model(raw), raw, atol=0, rtol=0)
    assert_close(transform.inverse(transform(raw)), raw, atol=1e-4, rtol=0)


def test_both_off_is_identity_none() -> None:
    assert build_action_transform(merge=False, stats_path=None) is None


def test_keys_mismatch_rejected() -> None:
    # gaussianize knots fit in a space whose keys disagree with the merge output
    # => the warp would run in the wrong coordinate system.
    grid = torch.linspace(0.0, 1.0, NUM_KNOTS)
    knots = (grid * 2.0 - 1.0).expand(3, NUM_KNOTS)
    gaussianize = Gaussianize(
        grid=grid,
        knots=knots,
        model_keys=("gas_pedal", "brake_pedal", "steering_angle"),
    )
    with pytest.raises(ValueError, match="post-merge space"):
        ActionTransform(merge=GasBrakeMerge(), gaussianize=gaussianize)


def test_action_transform_needs_a_stage() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        ActionTransform(merge=None, gaussianize=None)
