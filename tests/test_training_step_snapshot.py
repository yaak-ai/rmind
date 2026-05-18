"""Golden-snapshot regression test for the loss values produced by a deterministic
forward + compute_metrics pass on a fixed batch.

Parameters are filled with an RNG-free, index-based pattern (sin of the parameter
index in `named_parameters()` order) before the forward pass. This sidesteps
torch's RNG entirely, so weights are bit-identical across:
  - platforms (no SIMD/CPU/arch differences in init)
  - test ordering (no dependence on what consumed RNG before this test ran)
  - pytest sessions (no dependence on fixture cache state)

The snapshot lives at `tests/snapshots/training_step_losses.pt` and contains
the per-objective loss TensorDict. Refresh with:

    just update-snapshots

CPU-only. The recorded losses don't correspond to any "real" training
configuration — they're a stable function of the architecture and the input
shape. The test catches code regressions that change forward-pass arithmetic.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.nn import Module
from torch.testing import assert_close

from rmind.components.base import TensorTree
from rmind.components.episode import EpisodeBuilder
from rmind.components.objectives import (
    ForwardDynamicsPredictionObjective,
    InverseDynamicsPredictionObjective,
    MemoryExtractionObjective,
    PolicyObjective,
)
from tests.conftest import make_batch

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "training_step_losses.pt"
RTOL = 1e-3  # wide enough for cross-platform matmul jitter; real regressions are >>0.1%
ATOL = 1e-5

BATCH_SEED = 42
BATCH_B, BATCH_T = 2, 6


def _fresh_batch(device: torch.device) -> TensorTree:
    """Build a fresh batch with a fixed seed, independent of any shared fixture.

    The module-scoped conftest batch gets mutated in-place by transforms inside
    episode_builder.forward (e.g. in-place Normalize). Re-seeding and calling
    make_batch() directly ensures every run of this test starts from identical
    input bytes regardless of test ordering or fixture cache state.

    The seed is set inline here rather than in a session/function-scoped fixture
    because it must fire immediately before make_batch() — any intervening test
    that consumes RNG would shift the values and break the bit-identical guarantee.
    """
    pl.seed_everything(BATCH_SEED, workers=True, verbose=False)
    return make_batch(device, b=BATCH_B, t=BATCH_T).to_dict(retain_none=False)


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


def _compute_metrics(
    device: torch.device,
    episode_builder: EpisodeBuilder,
    encoder: Module,
    objectives: dict[str, Any],
) -> TensorDict:
    _fill_deterministic([episode_builder, encoder, *objectives.values()])
    batch = _fresh_batch(device)
    episode = episode_builder(batch)
    embedding = encoder(src=episode.embeddings_unpacked, mask=episode.attention_mask)
    metrics = TensorDict({
        name: obj.compute_metrics(episode=episode, embedding=embedding)
        for name, obj in objectives.items()
    })
    losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
    metrics["loss", "total"] = losses.sum(reduce=True)
    # Only keep scalar losses — metrics also contains _artifacts (large tensors).
    return TensorDict(
        {
            k: v
            for k, v in metrics.items(include_nested=True, leaves_only=True)
            if v.ndim == 0
        },
        batch_size=[],
    ).detach()


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

    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"snapshot {SNAPSHOT_PATH} missing; run `just update-snapshots` to create it"
        )

    objectives = {
        "inverse_dynamics": inverse_dynamics_prediction_objective,
        "forward_dynamics": forward_dynamics_prediction_objective,
        "memory_extraction": memory_extraction_objective,
        "policy_objective": policy_objective,
    }
    actual = _compute_metrics(device, episode_builder, encoder, objectives)
    expected = torch.load(SNAPSHOT_PATH, weights_only=False)

    assert_close(actual, expected, atol=ATOL, rtol=RTOL)
