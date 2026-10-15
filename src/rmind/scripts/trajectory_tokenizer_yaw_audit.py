"""Yaw-augmentation robustness of a frozen `TrajectoryTokenizer`
(docs/trajectory_vqbet_head_plan.md, step 0).

`YawShiftCrop` rotates the per-frame `trajectory_target` by a yaw `psi` on
half the training clips; the tokenizer never saw rotated targets. This runs
the trajectory val split through the tokenizer twice -- as recorded, and
rotated with `YawShiftCrop.rotate_labels` -- and writes one JSON with:

- `fixed_psi`: per |psi| in a grid (both signs pooled), full-decode
  ADE/FDE/final-heading error against the ROTATED target, the share of clips
  whose level-0 code changes, and the share of straight clips (recorded
  |final heading| < 4 deg) that land in a turn or glitch level-0 code.
- `training_marginal`: the same for psi drawn as `YawShiftCrop.sample_psi`
  draws it (random walk over `--frames` frames, clipped to the crop margin,
  realised to whole pixels), one random frame per clip, every clip perturbed;
  also split by speed band and |psi| band.

`--mask-degenerate` zeroes, at the encoder input, every standardized feature
whose training std is ~0. For model-403b4rrh:v26 that is pose 0's `x`:
identically 0 in the dead-reckoned target, so its std is clamped to 1e-6
and any rotation (x0 = -y0 sin psi) explodes it to z ~ 1e3-1e5.

Only `cfg.datamodule.val` is instantiated (see trajectory_tokenizer_audit).

Usage (in the training docker image, with the rbyte cache mounted):
    python -m rmind.scripts.trajectory_tokenizer_yaw_audit \
        --artifact yaak/rmind/model-403b4rrh:v26 --out yaw_v26.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.augment import YawShiftCrop
from rmind.models.trajectory_tokenizer import TrajectoryTokenizer, _wrap
from rmind.scripts.offset_diag import shutdown_dataloader
from rmind.scripts.trajectory_tokenizer_audit import (
    build_val_dataloader,
    load_tokenizer,
)

#: level-0 codes of model-403b4rrh:v26 (docs/trajectory_tokenizer_codebook_plan.md §10)
TURN_CODES = (0, 5, 6, 11)
GLITCH_CODES = (7, 10, 12)
STRAIGHT_MAX_DEG = 4.0
PSI_GRID_DEG = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.71)
SPEED_BANDS_MPS = ((0.0, 2.0), (2.0, 8.0), (8.0, 15.0), (15.0, 40.0))
PSI_BANDS_DEG = ((0.0, 1.0), (1.0, 2.5), (2.5, 4.0), (4.0, 5.0))

# r1a_yaw experiment values (config/experiment/yaak/patch_policy/
# dinov2_dinowm_causal_readout_dropout_depth_anything_v2_r1a_yaw.yaml)
YAW_FX = 389.4
YAW_CROP_WIDTH = 512
YAW_FRAME_WIDTH = 576


#: training std at or below this marks a feature as degenerate (never varied)
DEGENERATE_STD = 1e-5


def mask_degenerate_features(tokenizer: TrajectoryTokenizer) -> list[int]:
    """Zero the standardized encoder input of every zero-variance feature, in
    place on this instance; returns their flat indices."""
    _, std = tokenizer._norm_stats  # noqa: SLF001
    keep = (std > DEGENERATE_STD).float()
    normalize = tokenizer._normalize  # noqa: SLF001
    tokenizer._normalize = lambda x: normalize(x) * keep  # ty:ignore[invalid-assignment]  # noqa: SLF001
    return (keep == 0).nonzero().flatten().tolist()


def _collect_poses(
    tokenizer: TrajectoryTokenizer, dataloader: Any, device: torch.device
) -> Tensor:
    all_poses: list[Tensor] = []
    try:
        for raw_batch in dataloader:
            batch = tree_map(
                lambda x: x.to(device) if isinstance(x, Tensor) else x, raw_batch
            )
            all_poses.append(tokenizer._gather_poses(batch).cpu())  # noqa: SLF001
    finally:
        shutdown_dataloader(dataloader)
    return torch.cat(all_poses)


def _rotate(yaw: YawShiftCrop, poses: Tensor, psi: Tensor) -> Tensor:
    """`(n, P, 3)` poses rotated by the per-clip `(n,)` yaw, via the exact
    label path `YawShiftCrop` uses in training."""
    tree = {"context": {"trajectory_target": poses[:, None]}}
    return yaw.rotate_labels(tree, psi[:, None])["context"]["trajectory_target"][:, 0]


@torch.no_grad()
def _encode_decode(
    tokenizer: TrajectoryTokenizer, poses: Tensor, chunk: int = 8192
) -> tuple[Tensor, Tensor]:
    codes, recon = [], []
    for part in poses.split(chunk):
        c = tokenizer(part)
        codes.append(c)
        recon.append(tokenizer.invert(c))
    return torch.cat(codes), torch.cat(recon)


def _summary(  # noqa: PLR0913, PLR0917
    poses: Tensor,
    recon: Tensor,
    codes: Tensor,
    base_codes: Tensor,
    straight: Tensor,
    mask: Tensor | None = None,
) -> dict[str, Any]:
    if mask is not None:
        poses, recon, codes, base_codes, straight = (
            t[mask] for t in (poses, recon, codes, base_codes, straight)
        )
    n = poses.shape[0]
    if n == 0:
        return {"n": 0}
    err = (recon[..., :2] - poses[..., :2]).norm(dim=-1)
    heading_err = _wrap(recon[..., 2] - poses[..., 2]).abs()
    level0 = codes[:, 0]
    turn = torch.isin(level0, torch.tensor(TURN_CODES, device=level0.device))
    glitch = torch.isin(level0, torch.tensor(GLITCH_CODES, device=level0.device))
    n_straight = int(straight.sum())
    return {
        "n": n,
        "ade": err.mean().item(),
        "fde": err[..., -1].mean().item(),
        "ade_p95": err.mean(-1).quantile(0.95).item(),
        "final_heading_err_deg": math.degrees(heading_err[..., -1].mean().item()),
        "level0_changed": (level0 != base_codes[:, 0]).float().mean().item(),
        "codes_changed_any_level": (codes != base_codes).any(-1).float().mean().item(),
        "n_straight": n_straight,
        "straight_to_turn": (turn & straight).sum().item() / max(n_straight, 1),
        "straight_to_glitch": (glitch & straight).sum().item() / max(n_straight, 1),
    }


def audit(  # noqa: PLR0913, PLR0914
    tokenizer: TrajectoryTokenizer,
    poses: Tensor,
    pose_dt: float,
    device: torch.device,
    *,
    frames: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    poses = poses.to(device)
    n = poses.shape[0]
    yaw = YawShiftCrop(
        crop_width=YAW_CROP_WIDTH, fx=YAW_FX, cameras=["cam"], probability=1.0
    )
    margin_px = (YAW_FRAME_WIDTH - YAW_CROP_WIDTH) // 2
    max_rad = margin_px / YAW_FX

    base_codes, base_recon = _encode_decode(tokenizer, poses)
    straight = torch.rad2deg(poses[:, -1, 2]).abs() < STRAIGHT_MAX_DEG
    speed = poses[:, -1, 1] / (tokenizer.num_poses * pose_dt)

    result: dict[str, Any] = {
        "num_clips": n,
        "max_psi_deg": math.degrees(max_rad),
        "unrotated": _summary(poses, base_recon, base_codes, base_codes, straight),
        "fixed_psi": {},
    }

    for psi_deg in PSI_GRID_DEG:
        sign = torch.randint(0, 2, (n,), generator=generator).to(device) * 2 - 1
        psi = sign * math.radians(psi_deg)
        rotated = _rotate(yaw, poses, psi)
        codes, recon = _encode_decode(tokenizer, rotated)
        result["fixed_psi"][f"{psi_deg:g}"] = _summary(
            rotated, recon, codes, base_codes, straight
        )

    # the training distribution: YawShiftCrop.sample_psi over `frames` frames,
    # one random frame per clip, realised to whole pixels as forward() does
    walk = yaw.sample_psi(n, frames, max_rad=max_rad, device=device)
    frame = torch.randint(0, frames, (n,), generator=generator).to(device)
    psi = walk.gather(1, frame[:, None])[:, 0]
    psi = (psi * YAW_FX).round().clamp(-margin_px, margin_px) / YAW_FX
    rotated = _rotate(yaw, poses, psi)
    codes, recon = _encode_decode(tokenizer, rotated)
    abs_psi_deg = torch.rad2deg(psi).abs()

    marginal: dict[str, Any] = {
        "frames": frames,
        "abs_psi_deg_quantiles": {
            f"p{int(q * 100)}": abs_psi_deg.quantile(q).item() for q in (0.1, 0.5, 0.9)
        },
        "all": _summary(rotated, recon, codes, base_codes, straight),
        "by_speed_mps": {},
        "by_abs_psi_deg": {},
    }
    for lo, hi in SPEED_BANDS_MPS:
        mask = (speed >= lo) & (speed < hi)
        marginal["by_speed_mps"][f"{lo:g}-{hi:g}"] = {
            "unrotated": _summary(
                poses, base_recon, base_codes, base_codes, straight, mask
            ),
            "rotated": _summary(rotated, recon, codes, base_codes, straight, mask),
        }
    for lo, hi in PSI_BANDS_DEG:
        mask = (abs_psi_deg >= lo) & (abs_psi_deg < hi)
        marginal["by_abs_psi_deg"][f"{lo:g}-{hi:g}"] = _summary(
            rotated, recon, codes, base_codes, straight, mask
        )
    result["training_marginal"] = marginal
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="model-<run>:v<N>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--frames", type=int, default=32, help="episode length of the policy arm"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-degenerate", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, info = load_tokenizer(args.artifact, device)
    if args.mask_degenerate:
        info["masked_features"] = mask_degenerate_features(tokenizer)
    dataloader, pose_dt = build_val_dataloader(args.num_workers)
    poses = _collect_poses(tokenizer, dataloader, device)
    result = info | audit(
        tokenizer, poses, pose_dt, device, frames=args.frames, seed=args.seed
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["training_marginal"]["all"]))  # noqa: T201


if __name__ == "__main__":
    main()
