"""Golden-snapshot regression test for the loss values produced by a deterministic
forward + compute_metrics pass on a fixed batch.

The snapshot lives under `tests/snapshots/training_step_losses.json`. Refresh it with:

    just update-snapshots

This test is CPU-only on purpose: floating-point results across CUDA/MPS would
not match a single recorded baseline.
"""

import json
import math
import os
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.nn import Module

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
RTOL = 1e-5


@pytest.fixture(scope="module", autouse=True)
def _reseed_module() -> None:
    """Re-seed at module start so module-scoped fixtures (episode_builder, encoder,
    objectives) construct from a known RNG state regardless of preceding test modules.
    """
    pl.seed_everything(42, workers=True, verbose=False)


def _flatten_scalars(td: TensorDict) -> dict[str, float]:
    return {
        "/".join(map(str, k)): v.item()
        for k, v in td.items(include_nested=True, leaves_only=True)
        if isinstance(v, torch.Tensor) and v.ndim == 0
    }


def test_training_step_losses_snapshot(  # noqa: PLR0913, PLR0917
    device: torch.device,
    episode_builder: EpisodeBuilder,
    encoder: Module,
    inverse_dynamics_prediction_objective: InverseDynamicsPredictionObjective,
    forward_dynamics_prediction_objective: ForwardDynamicsPredictionObjective,
    memory_extraction_objective: MemoryExtractionObjective,
    policy_objective: PolicyObjective,
    batch_dict: TensorTree,
) -> None:
    if device.type != "cpu":
        pytest.skip("snapshot is CPU-only for cross-machine reproducibility")

    objectives = {
        "inverse_dynamics": inverse_dynamics_prediction_objective,
        "forward_dynamics": forward_dynamics_prediction_objective,
        "memory_extraction": memory_extraction_objective,
        "policy_objective": policy_objective,
    }

    torch.manual_seed(42)
    episode = episode_builder(batch_dict)
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

    missing = set(expected) - set(actual)
    added = set(actual) - set(expected)
    assert not missing, f"keys missing from actual: {sorted(missing)}"
    assert not added, f"new keys in actual not in snapshot: {sorted(added)}"

    drifted = {
        k: (actual[k], expected[k])
        for k in expected
        if not math.isclose(actual[k], expected[k], rel_tol=RTOL, abs_tol=1e-8)
    }
    assert not drifted, (
        f"losses drifted beyond rel_tol={RTOL}: {drifted}\n"
        f"if intentional, refresh with `just update-snapshots`"
    )
