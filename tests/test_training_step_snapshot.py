"""Golden-snapshot regression test for the loss values produced by a deterministic
forward + compute_metrics pass on a fixed batch.

The snapshot lives at `tests/snapshots/training_step_losses.json` and contains:

- `fingerprint`: sha256 over the model state_dict (keys + shapes + values) and the
  input batch tensors. Captures everything that determines the loss landscape.
- `losses`: scalar per-objective losses produced from those inputs.

Refresh with:

    just update-snapshots

CPU-only on purpose: floating-point results across CUDA/MPS would not match a
single recorded baseline.
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
RTOL = 1e-5
ATOL = 1e-8


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
        if isinstance(v, Tensor) and v.ndim == 0
    }


def _tensor_bytes(t: Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def _hash_state_dict(h: "hashlib._Hash", sd: Mapping[str, Tensor]) -> None:
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(repr(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(_tensor_bytes(v))


def _hash_pytree(h: "hashlib._Hash", tree: TensorTree) -> None:
    leaves, spec = tree_flatten_with_path(tree)
    h.update(treespec_dumps(spec).encode())
    for path, leaf in leaves:
        h.update(repr(path).encode())
        if isinstance(leaf, Tensor):
            h.update(repr(tuple(leaf.shape)).encode())
            h.update(str(leaf.dtype).encode())
            h.update(_tensor_bytes(leaf))
        else:
            h.update(repr(leaf).encode())


def _fingerprint(modules: Mapping[str, Module], batch: TensorTree) -> str:
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(b"module:")
        h.update(name.encode())
        _hash_state_dict(h, modules[name].state_dict())
    h.update(b"batch:")
    _hash_pytree(h, batch)
    return f"sha256:{h.hexdigest()}"


def _diagnose(actual: dict, expected: dict) -> str:
    fp_changed = actual["fingerprint"] != expected["fingerprint"]
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

    if fp_changed and (drifted or missing or added):
        return (
            "fingerprint AND losses differ from snapshot — inputs (state_dict / batch / "
            "fixtures) were modified.\n"
            "  expected fingerprint: {exp}\n"
            "  actual fingerprint:   {act}\n"
            "next step: if the input change is intentional, refresh with "
            "`just update-snapshots`."
        ).format(exp=expected["fingerprint"], act=actual["fingerprint"])

    if fp_changed:
        return (
            "fingerprint changed but losses still match — inputs were modified in a way "
            "that happens to preserve the loss values. refresh with "
            "`just update-snapshots` to record the new fingerprint."
        )

    # fingerprint matches: inputs identical → any loss diff is a code regression
    parts = [
        (
            "fingerprint matches snapshot — inputs are identical, so the loss diff is "
            "a CODE REGRESSION. investigate before refreshing the snapshot."
        )
    ]
    if drifted:
        parts.append(f"drifted losses (rel_tol={RTOL}, abs_tol={ATOL}):")
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

    fingerprint = _fingerprint(
        {"episode_builder": episode_builder, "encoder": encoder, **objectives},
        batch_dict,
    )

    torch.manual_seed(42)
    episode = episode_builder(batch_dict)
    embedding = encoder(src=episode.embeddings_unpacked, mask=episode.attention_mask)

    metrics = TensorDict({  # ty:ignore[invalid-argument-type]
        name: obj.compute_metrics(episode=episode, embedding=embedding)
        for name, obj in objectives.items()
    })
    losses = metrics.select(*((k, "loss") for k in metrics.keys()))  # noqa: SIM118
    metrics["loss", "total"] = losses.sum(reduce=True)

    actual = {"fingerprint": fingerprint, "losses": _flatten_scalars(metrics.detach())}

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

    fp_match = actual["fingerprint"] == expected["fingerprint"]
    losses_match = actual["losses"].keys() == expected["losses"].keys() and all(
        math.isclose(
            actual["losses"][k], expected["losses"][k], rel_tol=RTOL, abs_tol=ATOL
        )
        for k in expected["losses"]
    )
    if fp_match and losses_match:
        return

    pytest.fail(_diagnose(actual, expected))
