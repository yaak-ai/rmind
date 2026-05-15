"""Golden-snapshot regression test for the loss values produced by a deterministic
forward + compute_metrics pass on a fixed batch.

The snapshot at `tests/snapshots/training_step_losses.json` contains:

- `structure`: sha256 over the model state_dict (keys + shapes + dtypes only — no
  tensor values) and the batch tree spec (paths + shapes + dtypes). Stable across
  platforms; flips only when architecture or input shape changes.
- `losses`: scalar per-objective losses produced from those inputs.

Refresh with:

    just update-snapshots

CPU-only. Tensor *values* are intentionally excluded from the structure hash:
PyTorch's vectorized init kernels produce bit-different weights on x86 vs ARM
even from the same seed, so a value-sensitive hash would force per-platform
baselines. The loss tolerance (rel_tol below) absorbs the resulting platform
jitter while staying tight enough to flag real regressions.
"""

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_flatten_with_path, treespec_dumps  # noqa: PLC2701

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
RESET_SEED = 42
FORWARD_SEED = 42


def _reset_all_parameters(*modules: Module) -> None:
    """Reseed and re-initialize every parameter that has a `reset_parameters` method.

    Module-scoped fixtures cache their weights, and the RNG state at the time those
    fixtures were *constructed* depends on what tests pytest ran first. To get
    bit-identical weights across machines we ignore the cached values and reset
    everything from a fresh seed right before the forward pass.
    """
    pl.seed_everything(RESET_SEED, workers=True, verbose=False)
    for m in modules:
        m.apply(
            lambda submodule: (
                submodule.reset_parameters()  # ty:ignore[possibly-unbound-attribute]
                if hasattr(submodule, "reset_parameters")
                and callable(submodule.reset_parameters)
                else None
            )
        )


def _flatten_scalars(td: TensorDict) -> dict[str, float]:
    return {
        "/".join(map(str, k)): v.item()
        for k, v in td.items(include_nested=True, leaves_only=True)
        if isinstance(v, Tensor) and v.ndim == 0
    }


def _hash_state_dict_structure(h: "hashlib._Hash", sd: Mapping[str, Tensor]) -> None:
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(repr(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())


def _hash_pytree_structure(h: "hashlib._Hash", tree: TensorTree) -> None:
    leaves, spec = tree_flatten_with_path(tree)
    h.update(treespec_dumps(spec).encode())
    for path, leaf in leaves:
        h.update(repr(path).encode())
        if isinstance(leaf, Tensor):
            h.update(repr(tuple(leaf.shape)).encode())
            h.update(str(leaf.dtype).encode())
        else:
            h.update(repr(leaf).encode())


def _structure_hash(modules: Mapping[str, Module], batch: TensorTree) -> str:
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(b"module:")
        h.update(name.encode())
        _hash_state_dict_structure(h, modules[name].state_dict())
    h.update(b"batch:")
    _hash_pytree_structure(h, batch)
    return f"sha256:{h.hexdigest()}"


def _diagnose(actual: dict, expected: dict) -> str:
    structure_changed = actual["structure"] != expected["structure"]
    drifted = {
        k: (actual["losses"][k], expected["losses"][k])
        for k in expected["losses"]
        if k in actual["losses"]
        and not math.isclose(
            actual["losses"][k], expected["losses"][k], rel_tol=RTOL, abs_tol=ATOL
        )
    }
    missing = set(expected["losses"]) - set(actual["losses"])
    added = set(actual["losses"]) - set(expected["losses"])

    if structure_changed and (drifted or missing or added):
        return (
            "structure AND losses differ — architecture / input shape was modified.\n"
            f"  expected structure: {expected['structure']}\n"
            f"  actual structure:   {actual['structure']}\n"
            "next step: if the change is intentional, refresh with "
            "`just update-snapshots`."
        )

    if structure_changed:
        return (
            "structure changed but losses still match — arch/input shape was modified "
            "in a way that happens to preserve the loss values. refresh with "
            "`just update-snapshots` to record the new structure."
        )

    # structure matches: architecture/inputs are unchanged → any loss diff is a regression
    parts = [
        (
            "structure matches snapshot — architecture and input shapes are unchanged, "
            "so the loss diff is a CODE REGRESSION. investigate before refreshing."
        )
    ]
    if drifted:
        parts.append(
            f"drifted losses beyond rel_tol={RTOL}, abs_tol={ATOL} "
            "(tolerance is wide enough to absorb cross-platform float jitter):"
        )
        for k, (a, e) in sorted(drifted.items()):
            parts.append(f"  {k}: actual={a!r} expected={e!r} delta={a - e:+.4g}")
    if missing:
        parts.append(f"missing from actual: {sorted(missing)}")
    if added:
        parts.append(f"new keys not in snapshot: {sorted(added)}")
    return "\n".join(parts)


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

    _reset_all_parameters(episode_builder, encoder, *objectives.values())

    structure = _structure_hash(
        {"episode_builder": episode_builder, "encoder": encoder, **objectives},
        batch_dict,
    )

    torch.manual_seed(FORWARD_SEED)
    episode = episode_builder(batch_dict)
    embedding = encoder(src=episode.embeddings_unpacked, mask=episode.attention_mask)

    metrics = TensorDict({  # ty:ignore[invalid-argument-type]
        name: obj.compute_metrics(episode=episode, embedding=embedding)
        for name, obj in objectives.items()
    })
    losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
    metrics["loss", "total"] = losses.sum(reduce=True)

    actual = {"structure": structure, "losses": _flatten_scalars(metrics.detach())}

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

    structure_match = actual["structure"] == expected["structure"]
    losses_match = actual["losses"].keys() == expected["losses"].keys() and all(
        math.isclose(
            actual["losses"][k], expected["losses"][k], rel_tol=RTOL, abs_tol=ATOL
        )
        for k in expected["losses"]
    )
    if structure_match and losses_match:
        return

    pytest.fail(_diagnose(actual, expected))
