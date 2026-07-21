"""Extract foresight-probe features from the PT ControlTransformer over yaak/val.

One pass over the val set (config/dataset/yaak/val.yaml, clip_length=11,
episode = clip frames 0..5, strictly-future frames 6..10 = horizons H=1..5)
with the pretrained checkpoint (run t5cmm8om / unique-sun-340), producing a
shard cache per the foresight_mm cache schema contract:

per shard (4096 samples, dataloader order, shuffle=False, fp16 unless noted):
  ctx_foresight_pooled  (n,384)   encoder output at foresight slots, timestep 5, mean over 256 slots
  ctx_action_summary    (n,384)   encoder output at summary/action_summary, timestep 5
  ctx_obs_summary       (n,384)   encoder output at summary/observation_summary, timestep 5
  ctx_obs_history       (n,384)   encoder output at summary/observation_history, timestep 5
  img_dino_pooled_cur   (n,384)   episode.input_embeddings image/cam_front_left @ t5, mean over 256 patches
  fut_dino_pooled       (n,5,384) pooled DINO embeddings of clip frames 6..10 (episode-parity path)
  pred_fd_pooled_h1     (n,384)   FD-head prediction of frame 6 from timestep-5 context, mean over patches
  speed/gas/brake/steer/turn (n,11) batch meta signals over the full clip
  sample_id (n,) int64; frame_idx (n,) int64 (camera frame_idx of clip frame 5);
  input_id list[str]

plus ONE sibling file `<out>_grids.pt` for the FIRST `--grid-samples` (2048)
dataloader samples: pred_fd_grid_h1 / gt_grid_cur / gt_grid_h1 / gt_grid_h5
(each (n,256,384) fp16) + sample_id + input_id.

Mandatory parity checks run on the first batch (all assert):
  A. FD replica at index[:-1] reproduces compute_metrics logits (< 1e-6)
  B. objective FD foresight loss on first --loss-check-batches val batches is
     order-of-magnitude ~10 (run val metric: 10.01)
  C. standalone image path (episode builder's own transform + DINO backbone)
     on clip frame 5 reproduces episode.input_embeddings image @ t5 (< 1e-5 fp32)
  D. empirical episode<->clip frame mapping (expect timestep i == clip frame i, i=0..5)

Usage (smoke):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
      src/rmind/scripts/foresight_probe_extract.py \
      --out foresight_mm/cache/smoke --max-samples 512
Full run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \
      src/rmind/scripts/foresight_probe_extract.py \
      --out foresight_mm/cache/val
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from einops import pack, repeat
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CKPT = REPO_ROOT / "artifacts" / "model-t5cmm8om_latest" / "model.ckpt"

SHARD_SIZE = 4096
CUR = 5  # current observation frame index within the 11-frame clip
FUT = slice(6, 11)  # strictly-future frames, horizons H=1..5
N_HORIZONS = 5

FORESIGHT_KEY = ("foresight", "cam_front_left")
IMAGE_KEY = ("image", "cam_front_left")

SIGNAL_KEYS = {
    "speed": "meta/VehicleMotion/speed",
    "gas": "meta/VehicleMotion/gas_pedal_normalized",
    "brake": "meta/VehicleMotion/brake_pedal_normalized",
    "steer": "meta/VehicleMotion/steering_angle_normalized",
    "turn": "meta/VehicleState/turn_signal",
}
FRAME_IDX_KEY = "meta/ImageMetadata.cam_front_left/frame_idx"


# --------------------------------------------------------------------------- #
# dataloader (adapted from rmind-rqv offset_diag.py, pretrain experiment)
# --------------------------------------------------------------------------- #
def build_dataloader(
    batch_size: int = 12, num_workers: int = 3
) -> Any:
    """Instantiate the PRETRAIN experiment's val dataloader, shuffle=False."""
    import rmind  # noqa: F401  (registers the "eval" OmegaConf resolver)
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiment=yaak/control_transformer/pretrain"],
        )

    # make the cache path independent of the caller's working directory
    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")

    node = cfg.datamodule.val
    node.batch_size = batch_size
    node.num_workers = num_workers
    node.shuffle = False

    # the rbyte sample build's forkserver pool is occasionally flaky
    # (BrokenProcessPool on a shared box); one retry is cheap insurance
    try:
        return instantiate(node)
    except Exception as e:  # noqa: BLE001
        print(f"[extract] dataloader build failed ({e!r}); retrying once", flush=True)
        time.sleep(10)
        return instantiate(node)


def shutdown_dataloader(dataloader: Any) -> None:
    """Best-effort shutdown of TorchDataNodeDataLoader worker threads."""
    root = getattr(getattr(dataloader, "_loader", None), "root", None)
    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        shutdown = getattr(node, "_shutdown", None)
        if callable(shutdown):
            shutdown()
        stack.extend(getattr(node, attr, None) for attr in ("_it", "source", "root"))


def _to_device(batch: Any, device: torch.device) -> Any:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
        batch,
    )


def _slice_batch(batch: Any, sl: slice) -> Any:
    """Slice every per-sample leaf of a batch dict along dim 0 (for OOM retry)."""
    if isinstance(batch, dict):
        return {k: _slice_batch(v, sl) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch[sl]
    if isinstance(batch, (list, tuple)):
        return list(batch)[sl.start : sl.stop]
    return batch


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str | Path, device: str) -> Any:
    """Load the PT ControlTransformer; report state-dict key mismatches."""
    from rmind.models.control_transformer import ControlTransformer

    model = ControlTransformer.load_from_checkpoint(
        Path(ckpt_path), map_location="cpu", weights_only=False
    )
    # explicit key audit (strict load already succeeded if we got here, but the
    # checkpoint is from a diverged branch, so report it explicitly)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    print(
        f"[extract] checkpoint key audit: missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )
    if missing:
        print(f"[extract]   missing: {missing[:20]}", flush=True)
    if unexpected:
        print(f"[extract]   unexpected: {unexpected[:20]}", flush=True)
    del ckpt

    model = model.to(torch.device(device)).eval().requires_grad_(requires_grad=False)
    fd = model.objectives["forward_dynamics"]
    print(f"[extract] FD head paths: {fd.heads.tree_paths()}", flush=True)

    # Pre-ad93596 checkpoint: FD also has a ('continuous','speed') head whose
    # LogitBiasCrossEntropyLoss.logit_bias is populated by the LogitBiasSetter
    # trainer callback at fit start and is None after a bare load, which makes
    # compute_metrics crash. Zero it so compute_metrics runs (plain CE for the
    # speed leaf; the foresight GramAnchoringLoss we care about is unaffected).
    from rmind.components.loss import HasLogitBias

    for name, module in fd.losses.named_modules():
        if isinstance(module, HasLogitBias) and module.logit_bias is None:
            module.logit_bias = torch.tensor(0.0, device=torch.device(device))
            print(f"[extract] set losses.{name}.logit_bias = 0.0", flush=True)
    return model


# --------------------------------------------------------------------------- #
# FD replica (mirrors ForwardDynamicsPredictionObjective.compute_metrics L49-87,
# but with a caller-supplied timestep index)
# --------------------------------------------------------------------------- #
def fd_foresight_logits(
    fd: Any, episode: Any, embedding: Tensor, *, index: Any, mask_slice: slice
) -> Tensor:
    """FD-head foresight logits for the given timestep index.

    `index=episode.index[:-1]`, `mask_slice=slice(1, None)` reproduces
    compute_metrics exactly; `index=episode.index[-1:]`,
    `mask_slice=slice(-1, None)` is the serving anchor (timestep 5 -> frame 6).
    The (UTILITY, 'mask') embedding is a single learned vector constant across
    timesteps, so the mask_slice choice only sets the number of query steps.
    Returns (b, t_idx, 256, 384).
    """
    from rmind.components.base import Modality, SummaryToken

    if fd.norm is not None:
        embedding = fd.norm(embedding)

    observation_keys = fd.heads.tree_paths()
    observations = index.select(*observation_keys).parse(embedding)
    action_summary = (
        index
        .select(k := (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY))
        .parse(embedding)
        .get(k)
    )
    features = observations.apply(
        lambda obs: pack([obs, action_summary.broadcast_to(obs.shape)], "b t p *")[0]
    )
    features_projected = fd.projections(features.to_dict())
    _, _, n_patches, _ = episode.embeddings.get((
        Modality.IMAGE,
        "cam_front_left",
    )).shape
    mask_tokens = repeat(
        episode.embeddings.get((Modality.UTILITY, "mask"))[:, mask_slice],
        "b t 1 d -> b t n d",
        n=n_patches,
    )
    if fd.patch_pos_embed is not None:
        mask_tokens = fd.patch_pos_embed(mask_tokens)
    features_projected[Modality.FORESIGHT] = tree_map(
        lambda x: {"query": mask_tokens, "context": x},
        features_projected[Modality.FORESIGHT],
    )
    logits = fd.heads(
        features_projected,
        is_leaf=lambda x: isinstance(x, dict) and "query" in x and "context" in x,
    )
    return logits[Modality.FORESIGHT]["cam_front_left"]


# --------------------------------------------------------------------------- #
# standalone image path (the episode builder's own transform + DINO backbone)
# --------------------------------------------------------------------------- #
def embed_frames(model: Any, frames_u8: Tensor) -> Tensor:
    """(b, k, 324, 576, 3) uint8 -> (b, k, 256, 384) raw DINO patch embeddings.

    Uses the episode builder's own image input-transform (Rearrange ->
    CenterCrop[320,576] -> Resize[256,256] -> ToDtype(f32,scale) -> ImageNet
    Normalize) and image embedding module (TimmBackbone
    vit_small_patch16_dinov3.lvd1689m out_indices=[10] + Rearrange), i.e. the
    exact modules that produce episode.input_embeddings image/cam_front_left.
    """
    image_transform = model.episode_builder.input_transform[2].get("image")
    image_embedding = model.episode_builder.embeddings.get("image")
    return image_embedding(image_transform(frames_u8))


# --------------------------------------------------------------------------- #
# parity checks
# --------------------------------------------------------------------------- #
def run_parity_checks(
    model: Any, episode: Any, embedding: Tensor, batch: dict[str, Any]
) -> dict[str, Any]:
    from rmind.components.base import Modality

    fd = model.objectives["forward_dynamics"]
    results: dict[str, Any] = {}

    # A. replica vs compute_metrics logits (bit-exact expected)
    metrics = fd.compute_metrics(episode=episode, embedding=embedding)
    ref = metrics["_artifacts"]["last_embeddings"][Modality.FORESIGHT][
        "cam_front_left"
    ]
    mine = fd_foresight_logits(
        fd, episode, embedding, index=episode.index[:-1], mask_slice=slice(1, None)
    )
    diff_a = (mine - ref).abs().max().item()
    results["parity_A_replica_vs_compute_metrics_maxabs"] = diff_a
    assert diff_a < 1e-6, f"parity A failed: max abs diff {diff_a}"

    # C. standalone image path vs episode.input_embeddings @ t5.
    # NOTE: the standalone path embeds frames 5..10 as ONE 6-frame window so the
    # flattened conv/matmul batch shape (b*6) matches the episode's — cuDNN TF32
    # kernel selection is batch-shape dependent (frame 5 alone differs by up to
    # ~0.5 abs); with matching shapes the two paths are bit-exact.
    frames = batch["data"]["cam_front_left"]  # (b, 11, 324, 576, 3) uint8
    standalone_cur = embed_frames(model, frames[:, CUR : CUR + 6])[:, 0].float()
    ep_cur = episode.input_embeddings.get(IMAGE_KEY)[:, -1].float()
    diff_c = (standalone_cur - ep_cur).abs().max().item()
    results["parity_C_standalone_frame5_vs_episode_maxabs"] = diff_c
    assert diff_c < 1e-5, f"parity C failed: max abs diff {diff_c}"

    # D. which clip frames does the episode consume? (expect 0..5). Compared
    # across batch shapes, so matched-frame diffs are TF32-kernel noise (~0.5)
    # vs >40 for mismatched frames; assert the argmin mapping + a margin.
    standalone_all = embed_frames(model, frames).float()  # (b, 11, 256, 384)
    ep_all = episode.input_embeddings.get(IMAGE_KEY).float()  # (b, 6, 256, 384)
    mapping, matched, runner_up = [], [], []
    for t in range(ep_all.shape[1]):
        diffs = (standalone_all - ep_all[:, t : t + 1]).abs().amax(dim=(0, 2, 3))
        order = diffs.argsort()
        mapping.append(int(order[0].item()))
        matched.append(float(diffs[order[0]].item()))
        runner_up.append(float(diffs[order[1]].item()))
    results["parity_D_timestep_to_clip_frame"] = mapping
    results["parity_D_matched_maxabs"] = matched
    results["parity_D_runner_up_maxabs"] = runner_up
    assert mapping == list(range(6)), f"parity D failed: mapping {mapping}"
    assert all(
        m < 2.0 and r > 10.0 * m for m, r in zip(matched, runner_up, strict=True)
    ), f"parity D margin failed: matched {matched} runner-up {runner_up}"

    return results


# --------------------------------------------------------------------------- #
# main extraction
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid-samples", type=int, default=2048)
    ap.add_argument("--loss-check-batches", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} non-empty; refusing to overwrite")
    grids_path = out.parent / f"{out.name}_grids.pt"
    if grids_path.exists():
        raise SystemExit(f"{grids_path} exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    import pytorch_lightning as pl

    pl.seed_everything(42, workers=True)
    dev = torch.device(args.device)

    model = load_model(args.ckpt, args.device)
    fd = model.objectives["forward_dynamics"]

    loader = build_dataloader(batch_size=args.batch_size, num_workers=args.num_workers)
    n_ds = len(loader.dataset)
    print(
        f"[extract] val dataset: {n_ds} samples; max={args.max_samples}", flush=True
    )

    from rmind.components.base import Modality, SummaryToken

    SUMMARY = Modality.SUMMARY
    buffers: dict[str, list[Tensor]] = {}
    input_ids: list[str] = []
    grid_buffers: dict[str, list[Tensor]] = {}
    grid_input_ids: list[str] = []
    parity: dict[str, Any] = {}
    fd_losses: list[float] = []
    n_total = n_buf = n_shards = n_grid = 0
    t0 = time.perf_counter()

    def _flush(idx: int) -> None:
        shard = {k: torch.cat(v) for k, v in buffers.items()}
        shard["input_id"] = list(input_ids)
        torch.save(shard, out / f"shard_{idx:05d}.pt")
        for v in buffers.values():
            v.clear()
        input_ids.clear()

    def _process(batch: dict[str, Any], batch_idx: int) -> tuple[dict[str, Tensor], list[str], int]:
        nonlocal parity
        episode = model.episode_builder(batch)
        embedding = model.encoder(
            src=episode.embeddings_flattened, mask=episode.attention_mask
        )
        b = embedding.shape[0]

        if batch_idx == 0 and not parity:
            parity = run_parity_checks(model, episode, embedding, batch)
            print(f"[extract] parity checks: {parity}", flush=True)

        # B. FD foresight loss vicinity check (first --loss-check-batches)
        if len(fd_losses) < args.loss_check_batches:
            metrics = fd.compute_metrics(episode=episode, embedding=embedding)
            fd_losses.append(
                float(metrics["loss"][Modality.FORESIGHT]["cam_front_left"])
            )

        # encoder outputs at timestep 5
        idx_last = episode.index[-1]
        ctx_foresight = (
            idx_last.select(FORESIGHT_KEY).parse(embedding).get(FORESIGHT_KEY)
        )  # (b, 256, 384)
        summaries = idx_last.select(SUMMARY).parse(embedding)
        ctx_action_summary = summaries.get((SUMMARY, SummaryToken.ACTION_SUMMARY))[
            :, 0
        ]
        ctx_obs_summary = summaries.get((SUMMARY, SummaryToken.OBSERVATION_SUMMARY))[
            :, 0
        ]
        ctx_obs_history = summaries.get((SUMMARY, SummaryToken.OBSERVATION_HISTORY))[
            :, 0
        ]

        # raw DINO embeddings: current frame from the episode, futures standalone.
        # Frames 5..10 are embedded as one 6-frame window so the flattened batch
        # shape matches the episode's (bit-exact vs training targets; see
        # parity C note). Frame 5 of the window doubles as a per-batch anchor.
        gt_grid_cur = episode.input_embeddings.get(IMAGE_KEY)[:, -1]  # (b, 256, 384)
        frames = batch["data"]["cam_front_left"]  # (b, 11, 324, 576, 3) uint8
        win = embed_frames(model, frames[:, CUR : CUR + 6])  # (b, 6, 256, 384)
        anchor = (win[:, 0] - gt_grid_cur).abs().max().item()
        assert anchor < 1e-4, f"per-batch frame-5 anchor drifted: {anchor}"
        fut_grids = win[:, 1:]  # frames 6..10 -> (b, 5, 256, 384)

        # FD prediction at the serving anchor (timestep-5 context -> frame 6)
        pred_grid_h1 = fd_foresight_logits(
            fd, episode, embedding, index=episode.index[-1:], mask_slice=slice(-1, None)
        )[:, 0]  # (b, 256, 384)

        md = batch["data"]
        rec: dict[str, Tensor] = {
            "ctx_foresight_pooled": ctx_foresight.mean(dim=1).half(),
            "ctx_action_summary": ctx_action_summary.half(),
            "ctx_obs_summary": ctx_obs_summary.half(),
            "ctx_obs_history": ctx_obs_history.half(),
            "img_dino_pooled_cur": gt_grid_cur.mean(dim=1).half(),
            "fut_dino_pooled": fut_grids.mean(dim=2).half(),
            "pred_fd_pooled_h1": pred_grid_h1.mean(dim=1).half(),
            "sample_id": md["meta/sample_id"].to(torch.int64),
            "frame_idx": md[FRAME_IDX_KEY][:, CUR].to(torch.int64),
        }
        for name, key in SIGNAL_KEYS.items():
            rec[name] = md[key].half()

        grids: dict[str, Tensor] = {
            "pred_fd_grid_h1": pred_grid_h1.half(),
            "gt_grid_cur": gt_grid_cur.half(),
            "gt_grid_h1": fut_grids[:, 0].half(),
            "gt_grid_h5": fut_grids[:, 4].half(),
            "sample_id": rec["sample_id"],
        }
        rec.update({f"__grid__{k}": v for k, v in grids.items() if k != "sample_id"})
        rec["__grid__sample_id"] = grids["sample_id"]
        iid = [str(x) for x in batch["meta"]["input_id"]]
        return rec, iid, b

    try:
        with torch.inference_mode():
            for bi, batch_cpu in enumerate(loader):
                batch = _to_device(batch_cpu, dev)
                try:
                    rec, iid, b = _process(batch, bi)
                except torch.OutOfMemoryError:
                    print("[extract] OOM; retrying with half batch", flush=True)
                    torch.cuda.empty_cache()
                    nb = len(list(batch["meta"]["input_id"]))
                    half = max(1, nb // 2)
                    rec1, iid1, b1 = _process(_slice_batch(batch, slice(0, half)), bi)
                    rec2, iid2, b2 = _process(_slice_batch(batch, slice(half, nb)), bi)
                    rec = {k: torch.cat([rec1[k], rec2[k]]) for k in rec1}
                    iid, b = iid1 + iid2, b1 + b2

                grid_rec = {
                    k.removeprefix("__grid__"): rec.pop(k)
                    for k in list(rec)
                    if k.startswith("__grid__")
                }

                if args.max_samples is not None and n_total + b > args.max_samples:
                    keep = args.max_samples - n_total
                    rec = {k: v[:keep] for k, v in rec.items()}
                    grid_rec = {k: v[:keep] for k, v in grid_rec.items()}
                    iid = iid[:keep]
                    b = keep

                for k, v in rec.items():
                    buffers.setdefault(k, []).append(v.cpu())
                input_ids.extend(iid)

                if n_grid < args.grid_samples:
                    keep_g = min(b, args.grid_samples - n_grid)
                    for k, v in grid_rec.items():
                        grid_buffers.setdefault(k, []).append(v[:keep_g].cpu())
                    grid_input_ids.extend(iid[:keep_g])
                    n_grid += keep_g

                n_total += b
                n_buf += b
                if n_buf >= SHARD_SIZE:
                    _flush(n_shards)
                    n_shards += 1
                    n_buf = 0

                if (bi + 1) % 20 == 0:
                    r = n_total / (time.perf_counter() - t0)
                    print(
                        f"[extract] {n_total} samples ({bi + 1} batches, {r:.2f}/s)",
                        flush=True,
                    )
                if args.max_samples is not None and n_total >= args.max_samples:
                    break
    finally:
        shutdown_dataloader(loader)

    if n_buf > 0:
        _flush(n_shards)
        n_shards += 1

    grids_file = {k: torch.cat(v) for k, v in grid_buffers.items()}
    grids_file["input_id"] = grid_input_ids
    torch.save(grids_file, grids_path)

    fd_loss_mean = sum(fd_losses) / max(len(fd_losses), 1)
    # B. order-of-magnitude check vs run val metric (~10.01)
    assert 2.0 < fd_loss_mean < 50.0, f"FD foresight loss {fd_loss_mean} out of range"

    meta = {
        "n_samples": n_total,
        "n_shards": n_shards,
        "fd_loss_check": {
            "mean_foresight_loss": fd_loss_mean,
            "n_batches": len(fd_losses),
            "run_val_reference": 10.01,
        },
        "notes": {
            "ckpt": str(args.ckpt),
            "dataset": "config/dataset/yaak/val.yaml via experiment=yaak/control_transformer/pretrain, shuffle=False",
            "current_frame": CUR,
            "future_frames": "clip 6..10 (H=1..5)",
            "frame_idx": "camera frame_idx (meta/ImageMetadata.cam_front_left/frame_idx) of clip frame 5",
            "grid_samples": n_grid,
            "parity": parity,
            "elapsed_s": time.perf_counter() - t0,
            "batch_size": args.batch_size,
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[extract] done: {n_total} samples, {n_shards} shards -> {out}; "
        f"grids ({n_grid}) -> {grids_path}; "
        f"fd foresight loss {fd_loss_mean:.3f} over {len(fd_losses)} batches; "
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
