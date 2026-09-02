"""Emit a random-weight nero policy checkpoint and print its serving I/O contract.

For building a serving app **before any model is trained**. The weights are
random and the outputs are meaningless -- what is real, and what a serving app
needs to be written against, is the *shape* of the interface: which tensors go
in, at what dtypes and sizes, and what comes back.

    uv run python -m rmind.scripts.nero_serving_stub --out /tmp/nero-serving

Writes `policy_random.ckpt` (loadable with `torch.load(..., weights_only=False)`)
and `io_contract.json` next to it.

Nothing here trains, and nothing here should be benchmarked for accuracy. It is
a shape and plumbing fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate

from rmind.datamodules.nero_random import nero_random_batch
from rmind.scripts.nero_smoke import _cfg


def _describe(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively map a nested batch/output to `{key: {shape, dtype}}`."""
    out: dict[str, Any] = {}
    items = obj.items() if hasattr(obj, "items") else []
    for key, value in items:
        name = f"{prefix}{key}"
        if hasattr(value, "items"):
            out.update(_describe(value, prefix=f"{name}."))
        elif isinstance(value, torch.Tensor):
            out[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            out[name] = {"type": type(value).__name__}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--experiment", default="yaak/nero_arms/causal")
    _ = parser.add_argument("--out", default="/tmp/nero-serving")  # noqa: S108
    _ = parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    cfg = _cfg(args.experiment)
    model = instantiate(cfg.model).eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch = nero_random_batch(batch_size=args.batch_size)
    with torch.no_grad():
        output = model(batch)

    contract = {
        "note": "RANDOM WEIGHTS. Shapes are real; values are meaningless.",
        "experiment": args.experiment,
        "batch_size": args.batch_size,
        "parameters": {"total": total, "trainable": trainable},
        "inputs": _describe(batch),
        "outputs": _describe(output),
    }

    ckpt = out / "policy_random.ckpt"
    torch.save({"state_dict": model.state_dict(), "experiment": args.experiment}, ckpt)
    (out / "io_contract.json").write_text(json.dumps(contract, indent=2))

    print(json.dumps(contract, indent=2))  # noqa: T201
    print(f"\ncheckpoint: {ckpt}")  # noqa: T201
    print(f"contract:   {out / 'io_contract.json'}")  # noqa: T201


if __name__ == "__main__":
    main()
