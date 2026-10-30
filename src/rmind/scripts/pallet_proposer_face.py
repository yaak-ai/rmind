"""Proposer head trained on entry-face boxes instead of Grounding DINO boxes.

Same head and features as `pallet_proposer_head` (frozen policy DINOv3 ViT-S layer 10
on the 256x144 frame; CenterNet on a 32x18 grid), different targets: one box per
pallet around its base and entry face, the box convention of the multi-pallet test
set (`labels/multi_labels_v1.json`).

Targets per cached frame (every 10th frame of each recording):
    positive   the entry face of the pallet the keypoint net locked on: the box of its
               confident points (conf > 0.3), kept when the floor pose agrees
               (>= `--min-inliers` RANSAC inliers, mean confidence >= `--min-conf`);
               hand-labelled keypoints replace the prediction where they exist
    ignore     every Grounding DINO box (>= 0.25) that does not overlap the positive:
               possibly another pallet, which the keypoint net does not label
    negative   everything else
Frames where the keypoint net found no consistent pallet keep only ignore regions.

Evaluation is on the hand-labelled multi-pallet set: recall of every sure floor
pallet with a visible face (IoU >= 0.5, and "some box covers >= 70 %"), precision
(a predicted box hits any labelled pallet at IoU >= 0.3), and how many pallets per
multi-pallet frame get their own box; Grounding DINO is scored the same way.
Test frames must not leak into training: `--holdout recording` drops every
recording that has a test frame, `--holdout window` drops frames within
`--window` frames of one (the warehouse drive then still trains).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from structlog import get_logger

from rmind.pallet.correspondence import NAMES
from rmind.pallet.tracking import log_model, start, use_labels
from rmind.scripts.pallet_proposer_head import (
    ENCODER,
    LAYER,
    ProposerHead,
    decode,
    focal_loss,
    iou,
    targets,
)

logger = get_logger(__name__)

WORKSPACE = Path("/nasa/team-space/nikita/pallet-detector")


def face_box(points: np.ndarray, keep: np.ndarray) -> list[float] | None:
    if keep.sum() < 4:
        return None
    p = points[keep]
    return [*p.min(axis=0).tolist(), *p.max(axis=0).tolist()]


def build_targets(
    args: argparse.Namespace, caches: list[dict]
) -> dict[tuple[str, int], list[list[float]]]:
    """(job, frame) -> boxes as [x0, y0, x1, y1, s] with s = 1 positive, 0 ignore."""
    pred = pl.concat(
        [pl.read_parquet(p) for p in sorted(args.pred.glob("*.parquet"))],
        how="diagonal_relaxed",
    )
    hand = json.loads(args.entry_labels.read_text())
    out: dict[tuple[str, int], list[list[float]]] = {}
    for d in caches:
        rows = {
            r["frame"]: r for r in pred.filter(pl.col("job") == d["job"]).to_dicts()
        }
        for frame, dets in zip(d["frame"], d["boxes"], strict=True):
            r, name = rows.get(frame), f"{d['job']}_{frame:06d}.jpg"
            pos = None
            if name in hand:
                pos = face_box(
                    np.array([hand[name][k] for k in NAMES if k in hand[name]]),
                    np.ones(len(hand[name]), bool),
                )
            elif (
                r is not None
                and r.get("pose") is not None
                and r["inliers"] >= args.min_inliers
                and r["conf_mean"] >= args.min_conf
            ):
                pos = face_box(np.array(r["kp"]), np.array(r["conf"]) > 0.3)
            boxes = [[*pos, 1.0]] if pos is not None else []
            boxes += [
                [*b[:4], 0.0] for b in dets if pos is None or iou(b[:4], pos) < 0.1
            ]
            out[d["job"], frame] = boxes
    return out


def split(
    keys: list[tuple[str, int]], test: dict[str, list], mode: str, window: int
) -> list[bool]:
    tests: dict[str, list[int]] = {}
    for n in test:
        job, frame = n[:-4].rsplit("_", 1)
        tests.setdefault(job, []).append(int(frame))
    if mode == "recording":
        return [job not in tests for job, _ in keys]
    return [all(abs(f - t) > window for t in tests.get(job, [])) for job, f in keys]


def evaluate(pred: dict[str, list], test: dict[str, list]) -> dict[str, float]:
    """Head (or detector) boxes vs the hand-labelled multi-pallet set."""
    n = hit = cov = n_pred = good = 0
    own = []
    for name, pallets in test.items():
        p = pred.get(name, [])
        sure = [
            q["box"]
            for q in pallets
            if q["floor"] and q["face"] and not q.get("unsure")
        ]
        n += len(sure)
        hit += sum(any(iou(g, b) >= 0.5 for b in p) for g in sure)
        cov += sum(any(_cover(g, b) >= 0.7 for b in p) for g in sure)
        n_pred += len(p)
        good += sum(any(iou(q["box"], b) >= 0.3 for q in pallets) for b in p)
        if len(sure) >= 2:
            own.append(sum(any(iou(g, b) >= 0.3 for b in p) for g in sure) / len(sure))
    return {
        "recall50": hit / max(n, 1),
        "cover70": cov / max(n, 1),
        "precision30": good / max(n_pred, 1),
        "own_box_multi": float(np.mean(own)) if own else 0.0,
        "boxes_per_frame": n_pred / max(len(test), 1),
        "pallets": n,
    }


def _cover(gt: list[float], det: list[float]) -> float:
    ix = max(0.0, min(gt[2], det[2]) - max(gt[0], det[0]))
    iy = max(0.0, min(gt[3], det[3]) - max(gt[1], det[1]))
    return ix * iy / ((gt[2] - gt[0]) * (gt[3] - gt[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/nasa/team-space/nikita/pallet-detector-ab/proposer-cache-20261027/cache"),
    )
    parser.add_argument("--pred", type=Path, default=WORKSPACE / "predictions/pred_v2")
    parser.add_argument(
        "--entry-labels", type=Path, default=WORKSPACE / "labels/entry_labels_v2.json"
    )
    parser.add_argument(
        "--test-labels", type=Path, default=WORKSPACE / "labels/multi_labels_v1.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", choices=["recording", "window"], default="window")
    parser.add_argument(
        "--window",
        type=int,
        default=300,
        help="frames (10 s) around each test frame left out",
    )
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--min-conf", type=float, default=0.6)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = start(
        "train-proposer",
        vars(args),
        [
            "proposer",
            "dinov3-head",
            "entry-face-boxes",
            f"holdout:{args.holdout}",
            *args.tags,
        ],
        notes=f"CenterNet head on frozen {ENCODER} layer {LAYER}, entry-face boxes from keypoint predictions, tested on multi-pallet labels",
        enabled=not args.no_wandb,
    )
    use_labels(run, args.entry_labels, "entry-labels")
    use_labels(run, args.test_labels, "multi-labels")

    test = json.loads(args.test_labels.read_text())
    caches = [
        torch.load(f, weights_only=False, map_location="cpu")
        for f in sorted(args.cache.glob("*.pt"))
    ]
    tgt = build_targets(args, caches)
    toks = [d["tokens"] for d in caches]
    flips = [d["tokens_flip"] for d in caches]
    keys = [(d["job"], fr) for d in caches for fr in d["frame"]]
    cached = {
        f"{d['job']}_{fr:06d}.jpg": b
        for d in caches
        for fr, b in zip(d["frame"], d["boxes"], strict=True)
    }
    del caches
    toks_all, flips_all = torch.cat(toks), torch.cat(flips)
    boxes = [tgt[k] for k in keys]
    train_mask = split(keys, test, args.holdout, args.window)
    tr = [i for i, m in enumerate(train_mask) if m]
    test_idx = {
        f"{j}_{fr:06d}.jpg": i
        for i, (j, fr) in enumerate(keys)
        if f"{j}_{fr:06d}.jpg" in test
    }
    missing = set(test) - set(test_idx)
    logger.info(
        "data",
        train=len(tr),
        with_positive=sum(any(b[4] > 0 for b in boxes[i]) for i in tr),
        test=len(test_idx),
        test_missing=len(missing),
    )

    t_plain = [targets(boxes[i], 0.5, flip=False) for i in tr]
    t_flip = [targets(boxes[i], 0.5, flip=True) for i in tr]
    target_all = {
        k: torch.cat([
            torch.stack([t[k] for t in t_plain]),
            torch.stack([t[k] for t in t_flip]),
        ])
        for k in t_plain[0]
    }
    x_all = torch.cat([toks_all[tr], flips_all[tr]])
    test_x = toks_all[list(test_idx.values())]

    baseline = {
        f"gdino@{t}": evaluate(
            {n: [b for b in cached.get(n, []) if b[4] >= t] for n in test}, test
        )
        for t in (0.25, 0.35, 0.45)
    }
    for k, m in baseline.items():
        logger.info("baseline", detector=k, **{a: round(v, 3) for a, v in m.items()})

    head = ProposerHead().to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(x_all) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.05
    )
    best, best_metrics = -1.0, {}
    for epoch in range(args.epochs):
        head.train()
        perm, total = torch.randperm(len(x_all)), 0.0
        for i in range(0, len(x_all) - args.batch + 1, args.batch):
            b = perm[i : i + args.batch]
            t = {k: v[b].to(args.device) for k, v in target_all.items()}
            heat, size, offset = head(x_all[b].to(args.device).float())
            m = t["mask"][:, None].expand_as(size)
            norm = m.sum().clamp(min=1)
            loss = (
                focal_loss(heat, t["heat"], t["ignore"])
                + 0.2 * F.l1_loss(size[m], t["size"][m], reduction="sum") / norm
                + 2 * F.l1_loss(offset[m], t["offset"][m], reduction="sum") / norm
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        head.eval()
        with torch.no_grad():
            out = head(test_x.to(args.device).float())
        metrics = {}
        for thr in args.thresholds:
            pred = dict(zip(test_idx, decode(*out, thr), strict=True))
            metrics |= {
                f"{k}@{thr}": v
                for k, v in evaluate(pred, test).items()
                if k != "pallets"
            }
        score = max(
            metrics[f"recall50@{t}"] + metrics[f"precision30@{t}"]
            for t in args.thresholds
        )
        logger.info(
            "epoch",
            epoch=epoch,
            loss=round(total / max(len(x_all) // args.batch, 1), 4),
            **{
                k: round(v, 3)
                for k, v in metrics.items()
                if k.startswith(("recall50", "precision30", "own_box"))
            },
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": total / max(len(x_all) // args.batch, 1),
                **{f"test/{k}": v for k, v in metrics.items()},
            })
        if score > best:
            best, best_metrics = score, metrics
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "encoder": ENCODER,
                    "layer": LAYER,
                    "metrics": metrics,
                    "targets": "entry-face",
                },
                args.out / "best.pt",
            )
    result = {"head": best_metrics, **baseline}
    (args.out / "metrics.json").write_text(json.dumps(result, indent=1))
    logger.info(
        "done",
        **{
            k: round(v, 3)
            for k, v in best_metrics.items()
            if k.startswith(("recall50", "cover70", "precision30", "own_box"))
        },
    )
    if run is not None:
        run.summary.update(
            {f"best/{k}": v for k, v in best_metrics.items()}
            | {f"{d}/{k}": v for d, m in baseline.items() for k, v in m.items()}
        )
        log_model(
            run,
            args.out / "best.pt",
            "proposer-head",
            {"targets": "entry-face", "holdout": args.holdout, **best_metrics},
        )
        run.finish()


if __name__ == "__main__":
    main()
