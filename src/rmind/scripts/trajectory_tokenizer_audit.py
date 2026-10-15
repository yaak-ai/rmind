"""Codebook audit for a trained `TrajectoryTokenizer`
(docs/trajectory_tokenizer_codebook_plan.md, step 1).

Runs a checkpoint over the whole trajectory val split in eval mode and writes
one JSON with:

- `reconstruction`: val ADE/FDE/heading error, both as a mean over all clips
  and as a mean of per-batch means (what training logs as `val/ade`).
- `usage`: per level, the full code histogram, the number of codes with share
  >0.1% / >1%, and perplexity over the whole split (plus the mean per-batch
  perplexity, comparable to the logged `perplexity/q{i}`).
- `level0_class_quality` (decoder-free): ADE/FDE from each clip to the mean
  ground-truth trajectory of the clips sharing its level-0 code -- how good
  the coarse classes a VQ-BeT head would classify between are on their own.
  `global_mean` is the same with one class, for scale.
- `prefix`: ADE/FDE decoding from the first k levels only (later levels set
  to -1, so `get_output_from_indices` zeroes them). Diagnostic only: the
  decoder was trained on full 4-level sums, so k < 4 is out of distribution.
- `level0_modes`: per level-0 code, share, mean speed, signed final heading
  (mean / p10 / p90), the mean ground-truth trajectory, and the decoded
  codebook vector.

Only `cfg.datamodule.val` is instantiated: instantiating the full datamodule
makes pipefunc wipe the train sample store.

Usage (in the training docker image, with the rbyte cache mounted):
    python -m rmind.scripts.trajectory_tokenizer_audit \
        --artifact yaak/rmind/model-403b4rrh:v26 --out audit_v26.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.trajectory_tokenizer import TrajectoryTokenizer, _wrap
from rmind.scripts.offset_diag import shutdown_dataloader

if TYPE_CHECKING:
    from rbyte.dataloader import TorchDataNodeDataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
EXPERIMENT = "yaak/trajectory_tokenizer/pretrain"

#: camera frame rate the trajectory step is expressed in
FPS = 30.0
USAGE_THRESHOLDS = (0.001, 0.01)


def load_tokenizer(
    artifact: str, device: torch.device
) -> tuple[TrajectoryTokenizer, dict[str, Any]]:
    ckpt_path = TrajectoryTokenizer.download_wandb_artifact(artifact)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = {
        "artifact": artifact,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }
    tokenizer = TrajectoryTokenizer.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )
    tokenizer = tokenizer.to(device).eval().requires_grad_(requires_grad=False)
    return tokenizer, info


def build_val_dataloader(
    num_workers: int = 2,
) -> tuple[TorchDataNodeDataLoader[dict[str, Any]], float]:
    """The trajectory val dataloader and the pose spacing in seconds."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={EXPERIMENT}",
                "++datamodule.val.dataset.samples.resume=true",
            ],
        )

    # make the cache path independent of the caller's working directory
    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")

    node = cfg.datamodule.val
    node.num_workers = num_workers
    node.shuffle = False
    return instantiate(node), cfg.trajectory_step / FPS


def _xy_errors(pred: Tensor, target: Tensor) -> dict[str, float]:
    err = (pred[..., :2] - target[..., :2]).norm(dim=-1)  # (n, P)
    return {"ade": err.mean().item(), "fde": err[..., -1].mean().item()}


def _usage(codes: Tensor, codebook_size: int) -> dict[str, Any]:
    counts = torch.bincount(codes, minlength=codebook_size)
    share = counts.double() / counts.sum()
    entropy = -(share * share.clamp_min(1e-12).log()).sum()
    return {
        "counts": counts.tolist(),
        "share": [round(s, 6) for s in share.tolist()],
        **{
            f"num_codes_share_gt_{t:g}": int((share > t).sum())
            for t in USAGE_THRESHOLDS
        },
        "num_codes_used": int((counts > 0).sum()),
        "perplexity": entropy.exp().item(),
    }


def _class_quality(poses: Tensor, labels: Tensor, num_classes: int) -> dict[str, Any]:
    """ADE/FDE from each clip to its class's mean ground-truth trajectory."""
    flat = poses[..., :2].flatten(-2, -1)  # (n, P * 2)
    sums = torch.zeros(num_classes, flat.shape[-1], dtype=torch.float64)
    sums.index_add_(0, labels, flat.double())
    counts = torch.bincount(labels, minlength=num_classes).clamp_min(1)
    means = (sums / counts[:, None]).float()
    return _xy_errors(means[labels].unflatten(-1, (-1, 2)), poses)


@torch.no_grad()
def audit(  # noqa: PLR0914
    tokenizer: TrajectoryTokenizer,
    dataloader: TorchDataNodeDataLoader[dict[str, Any]],
    pose_dt: float,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    quantizer = tokenizer.quantizer
    num_levels, codebook_size = quantizer.num_quantizers, quantizer.codebook_size

    all_poses: list[Tensor] = []
    all_codes: list[Tensor] = []
    all_prefix: list[list[Tensor]] = [[] for _ in range(num_levels)]
    batch_metrics: list[dict[str, Any]] = []

    try:
        for i, raw_batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            batch = tree_map(
                lambda x: x.to(device) if isinstance(x, Tensor) else x, raw_batch
            )
            poses = tokenizer._gather_poses(batch)  # noqa: SLF001
            x = tokenizer._normalize(tokenizer.to_representation(poses))  # noqa: SLF001
            codes, _, _ = quantizer(tokenizer.encoder(x))

            for k in range(1, num_levels + 1):
                masked = codes.clone()
                masked[..., k:] = -1
                all_prefix[k - 1].append(
                    tokenizer.decode(quantizer.lookup(masked)).cpu()
                )

            full = all_prefix[-1][-1]
            poses = poses.cpu()
            heading_err = _wrap(full[..., 2] - poses[..., 2]).abs()
            batch_metrics.append({
                **_xy_errors(full, poses),
                "heading_err_deg": math.degrees(heading_err.mean().item()),
                "perplexity": quantizer.perplexity(codes).tolist(),
            })
            all_poses.append(poses)
            all_codes.append(codes.cpu())
    finally:
        shutdown_dataloader(dataloader)

    poses = torch.cat(all_poses)
    codes = torch.cat(all_codes)
    prefix = [torch.cat(p) for p in all_prefix]
    full = prefix[-1]
    n = poses.shape[0]

    heading_err = _wrap(full[..., 2] - poses[..., 2]).abs()
    reconstruction = {
        **_xy_errors(full, poses),
        "heading_err_deg": math.degrees(heading_err.mean().item()),
        "final_heading_err_deg": math.degrees(heading_err[..., -1].mean().item()),
        "mean_of_batch_means": {
            key: sum(m[key] for m in batch_metrics) / len(batch_metrics)
            for key in ("ade", "fde", "heading_err_deg")
        },
    }

    batch_perplexity = torch.tensor([m["perplexity"] for m in batch_metrics])
    usage = {
        f"q{q}": _usage(codes[:, q], codebook_size)
        | {"mean_batch_perplexity": batch_perplexity[:, q].mean().item()}
        for q in range(num_levels)
    }

    level0 = codes[:, 0]
    class_quality = {
        "level0": _class_quality(poses, level0, codebook_size),
        "levels01": _class_quality(
            poses, level0 * codebook_size + codes[:, 1], codebook_size**2
        ),
        "global_mean": _class_quality(poses, torch.zeros_like(level0), 1),
    }

    prefix_metrics = {f"k{k + 1}": _xy_errors(p, poses) for k, p in enumerate(prefix)}

    horizon_s = tokenizer.num_poses * pose_dt
    decoded_codebook = tokenizer.decode(quantizer.codebook(0)).cpu()
    modes = []
    for c in range(codebook_size):
        mask = level0 == c
        count = int(mask.sum())
        entry: dict[str, Any] = {"code": c, "count": count, "share": count / n}
        if count:
            p = poses[mask]
            final_heading = torch.rad2deg(p[:, -1, 2])
            entry |= {
                "mean_speed_mps": (p[:, -1, 1] / horizon_s).mean().item(),
                "final_heading_deg": {
                    "mean": final_heading.mean().item(),
                    "p10": final_heading.quantile(0.1).item(),
                    "p90": final_heading.quantile(0.9).item(),
                },
                "mean_gt_trajectory": [
                    [round(v, 4) for v in pose] for pose in p.mean(0).tolist()
                ],
            }
        entry["decoded_codebook_vector"] = [
            [round(v, 4) for v in pose] for pose in decoded_codebook[c].tolist()
        ]
        modes.append(entry)
    modes.sort(key=lambda m: -m["count"])

    return {
        "num_clips": n,
        "num_batches": len(batch_metrics),
        "horizon_s": horizon_s,
        "reconstruction": reconstruction,
        "usage": usage,
        "level0_class_quality": class_quality,
        "prefix": prefix_metrics,
        "level0_modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="model-<run>:v<N>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, info = load_tokenizer(args.artifact, device)
    dataloader, pose_dt = build_val_dataloader(args.num_workers)
    result = info | audit(tokenizer, dataloader, pose_dt, device, args.max_batches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("epoch", "num_clips", "reconstruction")}))  # noqa: T201


if __name__ == "__main__":
    main()
