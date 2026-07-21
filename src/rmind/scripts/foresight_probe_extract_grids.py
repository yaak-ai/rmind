"""Extract UNpooled foresight/GT patch grids per the GRID-CACHE SCHEMA CONTRACT.

Phase-1 (P1 oracle-bits / P2 K-head trajectory-WTA) needs patch-level features;
this script reuses the validated machinery of foresight_probe_extract.py
(checkpoint load + key audit, val dataloader compose, FD parity checks, the
TF32 6-frame-window trick for bit-exact standalone future DINO embeddings,
OOM-halving retry, forkserver hygiene) and writes shards of 2048 samples:

per shard (fp16 unless noted):
  ctx_foresight_grid   (n,256,384)   encoder outputs at the 256 foresight slots
                                     of timestep 5, UNpooled (FD-head context
                                     BEFORE the 768->384 projection concat)
  ctx_action_summary   (n,384)       encoder output at summary/action_summary @ t5
  gt_grid              (n,3,256,384) GT raw DINO patch grids of clip frames
                                     6, 8, 10 -> horizons H=1,3,5 in that order
  gt_pooled            (n,5,384)     pooled GT DINO of clip frames 6..10 (H=1..5)
  ctx_foresight_pooled (n,384)       = mean(ctx_foresight_grid, dim=1) computed
                                     from the STORED fp16 grid (fp32 accum on
                                     CPU, cast back to fp16) so the identity
                                     mean(grid,1)==pooled holds exactly
  speed/gas/brake/steer (n,11)       batch meta signals over the full clip
  sample_id (n,) int64; frame_idx (n,) int64 (camera frame_idx of clip frame 5)
  input_id  list[str]

plus meta.json {n_samples, n_shards, dataset, notes}.

Datasets:
  --dataset val          val.yaml via experiment=pretrain, shuffle=False,
                         all 18833 samples (dataloader order == existing
                         foresight_mm/cache/val pooled cache order)
  --dataset train096101  foresight_mm/train_096_101.yaml grafted into the
                         composed pretrain train datamodule node
                         (cfg.datamodule.train.dataset = OmegaConf.load(...),
                         reuse_map.md §2 / vjepa_extract.py pattern),
                         shuffle=True with a FIXED torch.Generator seed passed
                         to TorchDataNodeDataLoader (-> RandomSampler; the
                         node IS seedable), in_order=True, so the sample
                         sequence is fully determined by --seed. The emitted
                         sample_id list additionally defines the subset.

Disk safety: aborts before starting and between shard flushes if the target
filesystem has <20 GB free; prints running cache size per flush.

Smoke:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
      src/rmind/scripts/foresight_probe_extract_grids.py \
      --dataset val --max-samples 256 --out foresight_mm/cache/smoke_g
Full runs (aboutblank; see foresight_mm/notes/extract_grids.md):
  ... --dataset val --out foresight_mm/cache/val_g
  ... --dataset train096101 --max-samples 45000 --out foresight_mm/cache/train096101_g
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import Tensor

from rmind.scripts.foresight_probe_extract import (
    CONFIG_DIR,
    CUR,
    DEFAULT_CKPT,
    FORESIGHT_KEY,
    FRAME_IDX_KEY,
    IMAGE_KEY,
    REPO_ROOT,
    SIGNAL_KEYS,
    _slice_batch,
    _to_device,
    build_dataloader,
    embed_frames,
    load_model,
    run_parity_checks,
    shutdown_dataloader,
)

SHARD_SIZE = 2048
MIN_FREE_GB = 20.0
GT_GRID_HORIZONS = (1, 3, 5)  # -> clip frames 6, 8, 10; indices 0,2,4 of window
TRAIN_YAML = REPO_ROOT / "foresight_mm" / "train_096_101.yaml"

# subset of SIGNAL_KEYS per the grid-cache contract (no `turn`)
GRID_SIGNAL_KEYS = {k: SIGNAL_KEYS[k] for k in ("speed", "gas", "brake", "steer")}


def build_train096101_dataloader(
    batch_size: int, num_workers: int, seed: int
) -> Any:
    """Pretrain train node with foresight_mm/train_096_101.yaml grafted in,
    shuffle=True seeded via an explicit torch.Generator.

    Determinism: TorchDataNodeDataLoader builds a RandomSampler(generator=...)
    when shuffle=True (rbyte/dataloader.py) and its ParallelMapper defaults to
    in_order=True, so the sample order is fully determined by `seed`
    (hydra.utils.instantiate kwargs-override passes the live Generator object).
    """
    import rmind  # noqa: F401  (registers the "eval" OmegaConf resolver)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not TRAIN_YAML.exists():
        raise SystemExit(f"missing {TRAIN_YAML} (copy from rmind-rqv scratchpad)")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=yaak/control_transformer/pretrain"],
        )
    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")
    cfg.datamodule.train.dataset = OmegaConf.load(TRAIN_YAML)

    node = cfg.datamodule.train
    node.batch_size = batch_size
    node.num_workers = num_workers
    node.shuffle = True
    node.drop_last = False

    gen = torch.Generator().manual_seed(seed)
    try:
        return instantiate(node, generator=gen)
    except Exception as e:  # noqa: BLE001
        print(f"[extract-g] dataloader build failed ({e!r}); retrying once", flush=True)
        time.sleep(10)
        return instantiate(node, generator=gen)


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def _dir_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("shard_*.pt")) / 2**30


def _check_disk(out: Path) -> None:
    free = _free_gb(out)
    if free < MIN_FREE_GB:
        raise SystemExit(
            f"ABORT: only {free:.1f} GB free on target filesystem "
            f"(< {MIN_FREE_GB} GB); cache so far kept at {out}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=["val", "train096101"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42, help="train096101 shuffle seed")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} non-empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    _check_disk(out)

    import pytorch_lightning as pl

    pl.seed_everything(args.seed, workers=True)
    dev = torch.device(args.device)

    model = load_model(args.ckpt, args.device)

    if args.dataset == "val":
        loader = build_dataloader(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        dataset_note = (
            "config/dataset/yaak/val.yaml via experiment=yaak/control_transformer/"
            "pretrain, shuffle=False (order == foresight_mm/cache/val pooled cache)"
        )
    else:
        loader = build_train096101_dataloader(
            batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
        )
        dataset_note = (
            f"foresight_mm/train_096_101.yaml grafted into pretrain train node; "
            f"shuffle=True, RandomSampler(torch.Generator().manual_seed({args.seed})), "
            f"in_order=True -> order fully determined by seed; sample_id defines "
            f"the subset"
        )
    n_ds = len(loader.dataset)
    print(
        f"[extract-g] dataset={args.dataset}: {n_ds} samples; "
        f"max={args.max_samples}; free={_free_gb(out):.1f} GB",
        flush=True,
    )

    buffers: dict[str, list[Tensor]] = {}
    input_ids: list[str] = []
    parity: dict[str, Any] = {}
    n_total = n_buf = n_shards = 0
    t0 = time.perf_counter()

    def _flush(idx: int) -> None:
        shard = {k: torch.cat(v) for k, v in buffers.items()}
        shard["input_id"] = list(input_ids)
        torch.save(shard, out / f"shard_{idx:05d}.pt")
        for v in buffers.values():
            v.clear()
        input_ids.clear()
        print(
            f"[extract-g] shard {idx:05d} written; cache {_dir_gb(out):.2f} GB, "
            f"free {_free_gb(out):.1f} GB",
            flush=True,
        )
        _check_disk(out)

    def _process(
        batch: dict[str, Any], batch_idx: int
    ) -> tuple[dict[str, Any], list[str], int]:
        nonlocal parity
        episode = model.episode_builder(batch)
        embedding = model.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        b = embedding.shape[0]

        if batch_idx == 0 and not parity and not args.skip_parity:
            parity = run_parity_checks(model, episode, embedding, batch)
            print(f"[extract-g] parity checks: {parity}", flush=True)

        from rmind.components.base import Modality, SummaryToken

        # encoder outputs at timestep 5 (raw 384-d per slot, pre-FD-projection)
        idx_last = episode.index[-1]
        ctx_foresight = (
            idx_last.select(FORESIGHT_KEY).parse(embedding).get(FORESIGHT_KEY)
        )  # (b, 256, 384)
        ctx_action_summary = (
            idx_last
            .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )[:, 0]  # (b, 384)

        # GT DINO grids: frames 5..10 embedded as ONE 6-frame window so the
        # flattened batch shape matches the episode's (TF32 kernel selection is
        # batch-shape dependent; see foresight_probe_extract parity C note).
        gt_grid_cur = episode.input_embeddings.get(IMAGE_KEY)[:, -1]
        frames = batch["data"]["cam_front_left"]  # (b, 11, 324, 576, 3) uint8
        win = embed_frames(model, frames[:, CUR : CUR + 6])  # (b, 6, 256, 384)
        anchor = (win[:, 0] - gt_grid_cur).abs().max().item()
        assert anchor < 1e-4, f"per-batch frame-5 anchor drifted: {anchor}"
        fut_grids = win[:, 1:]  # clip frames 6..10 -> (b, 5, 256, 384)

        # store the fp16 grid FIRST, then derive pooled from the stored values
        # (fp32 accumulation on CPU) so mean(ctx_foresight_grid,1) ==
        # ctx_foresight_pooled holds bit-exactly for cache consumers.
        ctx_grid16 = ctx_foresight.half().cpu()
        md = batch["data"]
        rec: dict[str, Tensor] = {
            "ctx_foresight_grid": ctx_grid16,
            "ctx_action_summary": ctx_action_summary.half().cpu(),
            "gt_grid": fut_grids[:, [0, 2, 4]].half().cpu(),  # H=1,3,5
            "gt_pooled": fut_grids.mean(dim=2).half().cpu(),  # H=1..5
            "ctx_foresight_pooled": ctx_grid16.float().mean(dim=1).half(),
            "sample_id": md["meta/sample_id"].to(torch.int64).cpu(),
            "frame_idx": md[FRAME_IDX_KEY][:, CUR].to(torch.int64).cpu(),
        }
        for name, key in GRID_SIGNAL_KEYS.items():
            rec[name] = md[key].half().cpu()
        iid = [str(x) for x in batch["meta"]["input_id"]]
        return rec, iid, b

    try:
        with torch.inference_mode():
            for bi, batch_cpu in enumerate(loader):
                batch = _to_device(batch_cpu, dev)
                try:
                    rec, iid, b = _process(batch, bi)
                except torch.OutOfMemoryError:
                    print("[extract-g] OOM; retrying with half batch", flush=True)
                    torch.cuda.empty_cache()
                    nb = len(list(batch["meta"]["input_id"]))
                    half = max(1, nb // 2)
                    rec1, iid1, b1 = _process(_slice_batch(batch, slice(0, half)), bi)
                    rec2, iid2, b2 = _process(_slice_batch(batch, slice(half, nb)), bi)
                    rec = {k: torch.cat([rec1[k], rec2[k]]) for k in rec1}
                    iid, b = iid1 + iid2, b1 + b2

                if args.max_samples is not None and n_total + b > args.max_samples:
                    keep = args.max_samples - n_total
                    rec = {k: v[:keep] for k, v in rec.items()}
                    iid = iid[:keep]
                    b = keep

                for k, v in rec.items():
                    buffers.setdefault(k, []).append(v)
                input_ids.extend(iid)

                n_total += b
                n_buf += b
                if n_buf >= SHARD_SIZE:
                    _flush(n_shards)
                    n_shards += 1
                    n_buf = 0

                if (bi + 1) % 20 == 0:
                    r = n_total / (time.perf_counter() - t0)
                    print(
                        f"[extract-g] {n_total} samples ({bi + 1} batches, "
                        f"{r:.2f}/s)",
                        flush=True,
                    )
                if args.max_samples is not None and n_total >= args.max_samples:
                    break
    finally:
        shutdown_dataloader(loader)

    if n_buf > 0:
        _flush(n_shards)
        n_shards += 1

    meta = {
        "n_samples": n_total,
        "n_shards": n_shards,
        "dataset": args.dataset,
        "notes": {
            "ckpt": str(args.ckpt),
            "dataset_detail": dataset_note,
            "seed": args.seed,
            "current_frame": CUR,
            "gt_grid_frames": "clip 6, 8, 10 (H=1,3,5 in that order)",
            "gt_pooled_frames": "clip 6..10 (H=1..5)",
            "ctx_foresight_grid": (
                "encoder output at foresight slots 267..522 of timestep 5, raw "
                "384-d per slot (pre-projection FD context); pooled key derived "
                "from the stored fp16 grid (fp32 accum) so mean(grid,1)==pooled"
            ),
            "frame_idx": (
                "camera frame_idx (meta/ImageMetadata.cam_front_left/frame_idx) "
                "of clip frame 5"
            ),
            "shard_size": SHARD_SIZE,
            "parity": parity,
            "elapsed_s": time.perf_counter() - t0,
            "batch_size": args.batch_size,
            "cache_gb": _dir_gb(out),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[extract-g] done: {n_total} samples, {n_shards} shards, "
        f"{meta['notes']['cache_gb']:.2f} GB -> {out}; "
        f"{meta['notes']['elapsed_s']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    import multiprocessing as mp
    import sys

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
