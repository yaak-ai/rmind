"""Standalone script to refresh tests/snapshots/training_step_losses.json.

Run via:
    just update-snapshots

or directly:
    uv run python -m tests.scripts.update_snapshots

This is intentionally NOT a pytest test — it has the side effect of writing
a file, which tests should never do.
"""

import json

import torch

from tests.conftest import build_snapshot_modules
from tests.test_training_step_snapshot import (
    SNAPSHOT_PATH,
    _compute_metrics,
    _fresh_batch,
)


def main() -> None:
    torch.set_float32_matmul_precision("high")

    # Fail fast if the destination dir can't be created — don't waste a minute
    # building modules and computing metrics only to discover we can't write.
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    modules = build_snapshot_modules(device)

    metrics = _compute_metrics(modules.model, _fresh_batch(device))

    try:
        with SNAPSHOT_PATH.open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
    except BaseException:
        # Don't leave a half-written snapshot behind.
        SNAPSHOT_PATH.unlink(missing_ok=True)
        raise

    print(f"Snapshot written to {SNAPSHOT_PATH}")  # noqa: T201
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.6g}")  # noqa: T201


if __name__ == "__main__":
    main()
