"""Standalone script to refresh tests/snapshots/training_step_losses.pt.

Run via:
    just update-snapshots

or directly:
    uv run python -m tests.scripts.update_snapshots

This is intentionally NOT a pytest test — it has the side effect of writing
a file, which tests should never do.
"""

import torch

from tests.conftest import build_snapshot_modules
from tests.test_training_step_snapshot import (
    SNAPSHOT_PATH,
    _compute_metrics,
    _fresh_batch,
)


def main() -> None:
    torch.set_float32_matmul_precision("high")

    device = torch.device("cpu")
    modules = build_snapshot_modules(device)

    metrics = _compute_metrics(modules.model, _fresh_batch(device))

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, SNAPSHOT_PATH)

    print(f"Snapshot written to {SNAPSHOT_PATH}")  # noqa: T201
    for k, v in metrics.items(include_nested=True, leaves_only=True):
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            print(f"  {'/'.join(map(str, k))}: {v.item():.6g}")  # noqa: T201


if __name__ == "__main__":
    main()
