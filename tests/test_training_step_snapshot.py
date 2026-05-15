"""Golden-snapshot regression test for the loss values produced by a deterministic
forward + compute_metrics pass on a fixed batch.

Parameters are filled with an RNG-free, index-based pattern (sin of the parameter
index in `named_parameters()` order) before the forward pass. This sidesteps
torch's RNG entirely, so weights are bit-identical across:
  - platforms (no SIMD/CPU/arch differences in init)
  - test ordering (no dependence on what consumed RNG before this test ran)
  - pytest sessions (no dependence on fixture cache state)

The snapshot lives at `tests/snapshots/training_step_losses.json` and contains
the per-objective scalar losses. Refresh with:

    just update-snapshots

CPU-only. The recorded losses don't correspond to any "real" training
configuration — they're a stable function of the architecture and the input
shape. The test catches code regressions that change forward-pass arithmetic.
"""

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from rbyte.types import Batch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.testing import make_tensor

from rmind.components.base import TensorTree
from rmind.components.episode import EpisodeBuilder
from rmind.components.objectives import (
    ForwardDynamicsPredictionObjective,
    InverseDynamicsPredictionObjective,
    MemoryExtractionObjective,
    PolicyObjective,
)

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "training_step_losses.json"
UPDATE_ENV_VAR = "RMIND_UPDATE_SNAPSHOTS"
RTOL = 1e-4
ATOL = 1e-6


BATCH_SEED = 42
BATCH_B, BATCH_T = 2, 6


def _fresh_batch(device: torch.device) -> TensorTree:
    """Build a fresh batch with a fixed seed, independent of any shared fixture.

    Module-scoped batch fixtures get mutated in-place by transforms inside
    episode_builder.forward (e.g. in-place Normalize). Rebuilding from scratch here
    ensures every run of this test starts from identical input bytes.
    """
    pl.seed_everything(BATCH_SEED, workers=True, verbose=False)
    b, t = BATCH_B, BATCH_T
    batch = Batch(
        data=TensorDict(
            {
                "cam_front_left": make_tensor(
                    (b, t, 324, 576, 3),
                    dtype=torch.uint8,
                    device=device,
                    low=0,
                    high=256,
                ),
                "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=1.0
                ),
                "meta/VehicleMotion/steering_angle_normalized": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=-1.0, high=1.0
                ),
                "meta/VehicleMotion/speed": make_tensor(
                    (b, t), dtype=torch.float32, device=device, low=0.0, high=130.0
                ),
                "meta/VehicleState/turn_signal": make_tensor(
                    (b, t), dtype=torch.int64, device=device, low=0, high=3
                ),
                "waypoints/xy_normalized": make_tensor(
                    (b, t, 10, 2),
                    dtype=torch.float32,
                    device=device,
                    low=0.0,
                    high=20.0,
                ),
            },
            batch_size=[b],
            device=device,
        ),
        batch_size=[b],
        device=device,
    )
    return batch.to_dict(retain_none=False)


def _fill_deterministic(modules: Iterable[Module]) -> None:
    """Overwrite every parameter with a deterministic, RNG-free pattern, and clear
    any module-level caches that mutate during forward.

    Each parameter gets values `0.02 * sin(arange(numel) + offset)` where `offset`
    is its position in `named_parameters()` traversal order. Bit-identical across
    platforms and runs because no torch random source is consulted. EpisodeBuilder
    caches attention masks in non-persistent buffers on first forward — clear those
    so the cache state matches a fresh build regardless of prior calls.
    """
    offset = 0
    for m in modules:
        for _name, p in m.named_parameters():
            n = p.numel()
            values = torch.sin(torch.arange(n, dtype=p.dtype) + offset * 1.0e3) * 2.0e-2
            with torch.no_grad():
                p.copy_(values.reshape(p.shape))
            offset += 1
        if hasattr(m, "_attention_mask_spatial"):
            m.register_buffer("_attention_mask_spatial", None, persistent=False)
        if hasattr(m, "_attention_mask_temporal"):
            m.register_buffer("_attention_mask_temporal", None, persistent=False)


def _flatten_scalars(td: TensorDict) -> dict[str, float]:
    return {
        "/".join(map(str, k)): v.item()
        for k, v in td.items(include_nested=True, leaves_only=True)
        if isinstance(v, Tensor) and v.ndim == 0
    }


def _diagnose(actual: dict[str, float], expected: dict[str, float]) -> str:
    drifted = {
        k: (actual[k], expected[k])
        for k in expected
        if k in actual
        and not math.isclose(actual[k], expected[k], rel_tol=RTOL, abs_tol=ATOL)
    }
    missing = set(expected) - set(actual)
    added = set(actual) - set(expected)

    parts = [
        (
            "loss diff vs snapshot — weights are filled deterministically (no RNG), so "
            "this is a CODE REGRESSION unless the architecture or input shape changed. "
            f"rel_tol={RTOL}, abs_tol={ATOL}."
        )
    ]
    if drifted:
        parts.append("drifted losses:")
        for k, (a, e) in sorted(drifted.items()):
            parts.append(f"  {k}: actual={a!r} expected={e!r} delta={a - e:+.4g}")
    if missing:
        parts.append(f"missing from actual: {sorted(missing)}")
    if added:
        parts.append(f"new keys not in snapshot: {sorted(added)}")
    parts.append("if intentional, refresh with `just update-snapshots`.")
    return "\n".join(parts)


def test_training_step_losses_snapshot(  # noqa: PLR0913, PLR0917
    device: torch.device,
    episode_builder: EpisodeBuilder,
    encoder: Module,
    inverse_dynamics_prediction_objective: InverseDynamicsPredictionObjective,
    forward_dynamics_prediction_objective: ForwardDynamicsPredictionObjective,
    memory_extraction_objective: MemoryExtractionObjective,
    policy_objective: PolicyObjective,
) -> None:
    if device.type != "cpu":
        pytest.skip("snapshot is CPU-only for cross-machine reproducibility")

    objectives = {
        "inverse_dynamics": inverse_dynamics_prediction_objective,
        "forward_dynamics": forward_dynamics_prediction_objective,
        "memory_extraction": memory_extraction_objective,
        "policy_objective": policy_objective,
    }
    modules = [episode_builder, encoder, *objectives.values()]

    _fill_deterministic(modules)
    batch = _fresh_batch(device)

    episode = episode_builder(batch)
    embedding = encoder(src=episode.embeddings_unpacked, mask=episode.attention_mask)

    metrics = TensorDict({  # ty:ignore[invalid-argument-type]
        name: obj.compute_metrics(episode=episode, embedding=embedding)
        for name, obj in objectives.items()
    })
    losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
    metrics["loss", "total"] = losses.sum(reduce=True)
    actual = _flatten_scalars(metrics.detach())

    if os.environ.get(UPDATE_ENV_VAR):
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SNAPSHOT_PATH.open("w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")
        return

    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"snapshot {SNAPSHOT_PATH} missing; run `just update-snapshots` to create it"
        )

    with SNAPSHOT_PATH.open() as f:
        expected = json.load(f)

    keys_match = actual.keys() == expected.keys()
    values_match = keys_match and all(
        math.isclose(actual[k], expected[k], rel_tol=RTOL, abs_tol=ATOL)
        for k in expected
    )
    if values_match:
        return

    pytest.fail(_diagnose(actual, expected))
