"""Phase-2 offline eval dump (eval_v0): one datamodule spin, many conditions.

Runs multiple PatchPolicy checkpoints on the SAME real val batches under the
standing max_speed_override sweep (rmind.scripts.map_probe seam functions)
plus an explicit all-NaN max_speed condition (UNKNOWN flood), and dumps
per-sample last-frame code logits / argmax codes / decoded action chunks to a
compressed .npz for offline analysis:

- per-override decoded gas/brake/steer means + deltas vs the None baseline
- speed-conditioned contrasts (e.g. override=30 vs 100 on v > 50 km/h frames)
- arm-vs-parent warm-start fidelity under missing map input (None condition)

Map-less checkpoints (no max_speed_tokenizer, e.g. the warm-start parent) are
run under the None condition only.

Usage (box worktree, PYTHONPATH pointing at it):
  python -m rmind.scripts.eval_v0_dump \
      --config-dir /abs/path/to/config \
      --ckpt armM=artifacts/model-<id>:v0/model.ckpt [--ckpt ...] \
      [--experiment yaak/patch_policy/dinov2_dinowm_maxspeed_warmstart] \
      [--batches 24] [--batch-cache <pt>] \
      --out diag_results/eval_v0/probe_dump.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from rmind.models.patch_policy import PatchPolicy
from rmind.scripts.map_probe import (
    DEFAULT_OVERRIDES,
    _to_device,
    override_name,
    run_override,
)

MAX_SPEED_KEY = "meta/MapContext/max_speed"


def collect_batches(args: argparse.Namespace) -> list[dict]:
    """Real val batches, optionally cached to disk so re-runs skip the
    ~19 min datamodule spin."""
    if args.batch_cache and Path(args.batch_cache).exists():
        batches = torch.load(args.batch_cache, weights_only=False)
        print(f"loaded {len(batches)} cached val batches", flush=True)
        return batches

    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(
            config_name="train", overrides=[f"experiment={args.experiment}"]
        )
    datamodule = instantiate(cfg.datamodule)
    loader = datamodule.val_dataloader()
    batches: list[dict] = []
    for batch in loader:
        batches.append(batch)
        print(f"collected batch {len(batches)}/{args.batches}", flush=True)
        if len(batches) >= args.batches:
            break
    if args.batch_cache:
        Path(args.batch_cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(batches, args.batch_cache)
    return batches


def main() -> None:  # noqa: PLR0914
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument(
        "--experiment",
        default="yaak/patch_policy/dinov2_dinowm_maxspeed_warmstart",
    )
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument(
        "--ckpt",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeatable; NAME keys the npz entries",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch-cache", default=None, help="optional .pt batch cache")
    ap.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="split each cached batch into chunks of this many samples for the "
        "forward pass (0 = whole batch); lowers peak VRAM ~linearly so the "
        "probe fits on busy shared GPUs",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    batches = collect_batches(args)

    arrays: dict[str, np.ndarray] = {}
    # per-sample last-frame vehicle speed (the conditioning variable for the
    # speed>50 contrast)
    arrays["speed_last"] = (
        torch.cat(
            [b["data"]["meta/VehicleMotion/speed"][:, -1].reshape(-1) for b in batches]
        )
        .float()
        .numpy()
    )
    if MAX_SPEED_KEY in batches[0]["data"]:
        last = [b["data"][MAX_SPEED_KEY][:, -1] for b in batches]
        arrays["max_speed_last"] = (
            torch.cat([x.reshape(x.shape[0], -1)[:, 0] for x in last])
            .float()
            .numpy()
        )
        print("batches DO carry", MAX_SPEED_KEY, flush=True)
    else:
        print("batches do NOT carry", MAX_SPEED_KEY, flush=True)

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model = PatchPolicy.load_from_checkpoint(
            path, map_location="cpu", weights_only=False
        )
        model.sample_codes = False  # deterministic argmax decoding
        model = model.to(device).eval()
        is_map = model.max_speed_tokenizer is not None

        conds: list[tuple[str, float | None, bool]] = [("None", None, False)]
        if is_map:
            conds += [(override_name(v), v, False) for v in DEFAULT_OVERRIDES[1:]]
            conds += [("NaNflood", None, True)]

        def _slice_tree(node, s, e):
            if isinstance(node, dict):
                return {k: _slice_tree(v, s, e) for k, v in node.items()}
            if isinstance(node, torch.Tensor):
                return node[s:e]
            return node

        for cname, ov, nanflood in conds:
            outs: dict[str, list] = {"logits": [], "codes": [], "chunk": []}
            for batch in batches:
                bsz = batch["data"]["meta/VehicleMotion/speed"].shape[0]
                step = args.micro_batch or bsz
                for s in range(0, bsz, step):
                    chunk_cpu = _slice_tree(batch, s, s + step)
                    batch_d = _to_device(chunk_cpu, device)
                    if nanflood:
                        speed = batch_d["data"]["meta/VehicleMotion/speed"]
                        b2 = dict(batch_d)
                        b2["data"] = dict(batch_d["data"])
                        b2["data"][MAX_SPEED_KEY] = torch.full(
                            (*speed.shape[:2], 1), float("nan"), device=device
                        )
                        batch_d = b2
                    out = run_override(model, batch_d, ov)
                    for k in outs:
                        outs[k].append(out[k])
                    del batch_d
                torch.cuda.empty_cache()
            for k, v in outs.items():
                arrays[f"{name}/{cname}/{k}"] = torch.cat(v).numpy()
            print(f"{name} {cname} done", flush=True)

        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    n = arrays["speed_last"].shape[0]
    print(f"saved {args.out} ({len(arrays)} arrays, {n} samples/condition)")


if __name__ == "__main__":
    main()
