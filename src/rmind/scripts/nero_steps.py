"""The real optimizer step count of a nero experiment: len(train_dataloader) x max_epochs.

    python -m rmind.scripts.nero_steps yaak/nero_robot/bimanual_hand_off [--override K=V ...]

The cosine schedule (`get_cosine_schedule_with_warmup`) has no clamp: past
`lr_total_steps` the LR swings back up. So every nero experiment's
`lr_total_steps` must equal the step count the trainer will really take, which
depends on the split (window count), the windowing and the batch size. This
builds the experiment's TRAIN loader exactly as training does (shuffle,
drop_last, batch_size; no image streams -- they do not change the count) and
prints the count next to the configured `lr_total_steps`. Exits 1 on a mismatch.
Single device, no gradient accumulation (the nero trainer configs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def compose(experiment: str, overrides: list[str] | None = None) -> Any:
    from hydra import compose as hydra_compose  # noqa: PLC0415
    from hydra import initialize_config_dir  # noqa: PLC0415

    import rmind  # noqa: F401, PLC0415  (registers the `eval` resolver)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return hydra_compose(
            config_name="train",
            overrides=[f"experiment={experiment}", *(overrides or [])],
        )


def step_count(experiment: str, overrides: list[str] | None = None) -> dict[str, int]:
    """`{train_windows, batch_size, train_batches, max_epochs, steps, configured}`.

    Raises:
        ValueError: if the trainer is multi-device or accumulates gradients.
    """
    from hydra.utils import instantiate  # noqa: PLC0415

    cfg = compose(
        experiment, [*(overrides or []), "datamodule.train.dataset.streams=null"]
    )
    if (
        int(cfg.trainer.get("accumulate_grad_batches", 1)) != 1
        or int(cfg.trainer.get("devices", 1)) != 1
    ):
        msg = "nero_steps assumes one device and no gradient accumulation"
        raise ValueError(msg)
    loader = instantiate(cfg.datamodule.train)
    batches = len(loader)
    epochs = int(cfg.trainer.max_epochs)
    return {
        "train_windows": len(loader.dataset),
        "batch_size": int(cfg.batch_size),
        "train_batches": batches,
        "max_epochs": epochs,
        "steps": batches * epochs,
        "configured": int(cfg.lr_total_steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    result = step_count(args.experiment, args.override)
    print(json.dumps(result, indent=1))  # noqa: T201
    if result["steps"] != result["configured"]:
        print(  # noqa: T201
            f"lr_total_steps={result['configured']} but the trainer takes "
            f"{result['steps']} steps: set lr_total_steps: {result['steps']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
