"""Pallet proposer: a detection head on the policy's frozen DINOv3 features.

Replaces Grounding DINO (a separate ~170M model on the full 1920x1080 frame) by a
small head on features the D12 patch policy computes anyway: frozen
`vit_small_patch16_dinov3.lvd1689m`, layer 10 (pre-norm, `TimmBackbone`
`out_indices=[10]`), on the 256x144 frames of `extract_d12_frames.sh`, ImageNet
normalisation -> a 16x9 grid of 384-d patch tokens (one token ~120 full-res px).

The head is CenterNet-style: tokens -> a few convs -> 2x upsample to a 32x18 grid
-> per cell a pallet-centre heatmap, box size and sub-cell offset. Several
pallets come out as several peaks.

Two stages:

    cache   per job, every `--stride`-th frame: Grounding DINO on the full-res
            frame (ALL boxes >= its 0.25 threshold, fork tips removed) as the
            pseudo-label, and the frozen tokens of the 256x144 policy frame and
            of its mirror image (flip augmentation without re-running the ViT).
    train   CenterNet on the cache; boxes with score >= `--pos` are pallets,
            boxes in [0.25, --pos) are ignored (masked out of the loss).
            Validation is by recording. Reports, against Grounding DINO
            boxes >= --pos on the held-out recordings: recall / precision at
            IoU 0.5 and recall of the policy's target (the box `pick_target`
            chose). With `--kp-checkpoint`, also end to end on the hand-labelled
            held-out frames: keypoint error and pose success with the crop from
            the head's box vs from Grounding DINO's.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from structlog import get_logger
from torch import Tensor, nn

from rmind.pallet.tracking import log_model, start, use_labels

logger = get_logger(__name__)

ENCODER = "vit_small_patch16_dinov3.lvd1689m"
LAYER = 10
POLICY_H, POLICY_W = 144, 256
GRID_H, GRID_W = 18, 32  # tokens 9x16, upsampled 2x
WIDTH, HEIGHT = 1920, 1080
CELL = WIDTH / GRID_W  # 60 full-res px per heatmap cell
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
VAL_JOBS = ("2026-09-10--15-17-38", "2026-10-02--15-48-24", "2026-10-22--09-50-09")
# byte-identical copy of 2026-10-22--09-50-09: never cache it
DUPLICATES = ("2026-10-22--09-49-39",)


# --- cache ---------------------------------------------------------------------------


def policy_frame(root: Path, job: str, index: int, full: np.ndarray) -> np.ndarray:
    """The 256x144 frame the policy sees; resized from the full frame if not extracted."""
    from PIL import Image

    path = (
        root
        / job
        / "cam_fork"
        / "frames"
        / f"{POLICY_W}x{POLICY_H}"
        / f"{index:06d}.jpg"
    )
    if path.is_file():
        return np.array(Image.open(path).convert("RGB"))
    return np.array(
        Image.fromarray(full).resize((POLICY_W, POLICY_H), Image.Resampling.BILINEAR)
    )


def encoder(device: str) -> nn.Module:
    from rmind.components.timm_backbone import TimmBackbone

    return (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[POLICY_H, POLICY_W])
        .to(device)
        .eval()
    )


def tokens(enc: nn.Module, images: np.ndarray, device: str) -> Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        return enc(((x - MEAN) / STD).to(device)).half().cpu()


def cache(args: argparse.Namespace) -> None:
    from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, Proposer
    from rmind.scripts.pallet_predict_entry import frames

    args.out.mkdir(parents=True, exist_ok=True)
    proposer, enc = (
        Proposer(args.device, "IDEA-Research/grounding-dino-tiny"),
        encoder(args.device),
    )
    for job in args.jobs:
        path = args.out / f"{job}.pt"
        if path.exists() or job in DUPLICATES:
            continue
        idx_all, toks, toks_flip, boxes = [], [], [], []
        for idx, images in frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", args.stride, args.batch
        ):
            small = np.stack([
                policy_frame(D12_ROOT, job, i, im)
                for i, im in zip(idx, images, strict=True)
            ])
            toks.append(tokens(enc, small, args.device))
            toks_flip.append(
                tokens(enc, np.ascontiguousarray(small[:, :, ::-1]), args.device)
            )
            boxes += [[[*b, s] for b, s in c] for c in proposer.detect_all(images)]
            idx_all += idx
        torch.save(
            {
                "job": job,
                "frame": idx_all,
                "tokens": torch.cat(toks),
                "tokens_flip": torch.cat(toks_flip),
                "boxes": boxes,
            },
            path,
        )
        logger.info(
            "cached", job=job, frames=len(idx_all), with_box=sum(bool(b) for b in boxes)
        )


# --- head ----------------------------------------------------------------------------


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(16, cout),
        nn.GELU(),
    )


class ProposerHead(nn.Module):
    """(B, 384, 9, 16) tokens -> heatmap logits (B, 1, 18, 32), log box size (B, 2, ..), offset (B, 2, ..)."""

    def __init__(self, dim: int = 384, width: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(dim, width, 1),
            _block(width, width),
            _block(width, width),
            _block(width, width),
        )
        self.up = _block(width, 128)
        self.heat = nn.Conv2d(128, 1, 3, padding=1)
        self.size = nn.Conv2d(128, 2, 3, padding=1)
        self.offset = nn.Conv2d(128, 2, 3, padding=1)
        nn.init.constant_(self.heat.bias, -2.19)  # CenterNet prior: sigmoid = 0.1

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.up(
            F.interpolate(
                self.trunk(x),
                size=(GRID_H, GRID_W),
                mode="bilinear",
                align_corners=False,
            )
        )
        return self.heat(x), self.size(x), self.offset(x)


def targets(boxes: list[list[float]], pos: float, flip: bool) -> dict[str, Tensor]:  # ruff: ignore[boolean-type-hint-positional-argument]
    """CenterNet targets on the 32x18 grid; boxes in [0.25, pos) are an ignore region."""
    heat = torch.zeros(GRID_H, GRID_W)
    ignore = torch.zeros(GRID_H, GRID_W, dtype=torch.bool)
    size, offset, mask = (
        torch.zeros(2, GRID_H, GRID_W),
        torch.zeros(2, GRID_H, GRID_W),
        torch.zeros(GRID_H, GRID_W, dtype=torch.bool),
    )
    ys, xs = torch.meshgrid(
        torch.arange(GRID_H) + 0.5, torch.arange(GRID_W) + 0.5, indexing="ij"
    )
    for x0, y0, x1, y1, s in boxes:
        if flip:
            x0, x1 = WIDTH - x1, WIDTH - x0  # ruff: ignore[redefined-loop-name]
        if s < pos:
            ignore[
                int(y0 / CELL) : int(np.ceil(y1 / CELL)),
                int(x0 / CELL) : int(np.ceil(x1 / CELL)),
            ] = True
            continue
        cx, cy = (x0 + x1) / 2 / CELL, (y0 + y1) / 2 / CELL
        w, h = (x1 - x0) / CELL, (y1 - y0) / CELL
        sigma = max(0.5, min(w, h) / 6)
        heat = torch.maximum(
            heat, torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
        )
        i, j = min(int(cy), GRID_H - 1), min(int(cx), GRID_W - 1)
        heat[i, j] = 1.0
        size[:, i, j] = torch.tensor([np.log(w), np.log(h)])
        offset[:, i, j] = torch.tensor([cx - j, cy - i])
        mask[i, j] = True
    ignore &= heat < 0.5  # a real pallet wins over an overlapping ignore box
    return {
        "heat": heat[None],
        "ignore": ignore[None],
        "size": size,
        "offset": offset,
        "mask": mask,
    }


def focal_loss(logits: Tensor, target: Tensor, ignore: Tensor) -> Tensor:
    p = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = (1 - pos) * (~ignore).float()
    loss = -(
        pos * (1 - p) ** 2 * p.log() + neg * (1 - target) ** 4 * p**2 * (1 - p).log()
    )
    return loss.sum() / pos.sum().clamp(min=1)


def decode(
    heat: Tensor, size: Tensor, offset: Tensor, threshold: float, k: int = 10
) -> list[list[list[float]]]:
    """Peaks (3x3 max) above `threshold` -> full-res (x0, y0, x1, y1, score) per image."""
    p = heat.sigmoid()
    peaks = p * (F.max_pool2d(p, 3, stride=1, padding=1) == p)
    out = []
    for b in range(p.shape[0]):
        scores, flat = peaks[b, 0].flatten().topk(k)
        boxes = []
        for s, f in zip(scores.tolist(), flat.tolist(), strict=True):
            if s < threshold:
                break
            i, j = divmod(f, GRID_W)
            cx, cy = j + offset[b, 0, i, j].item(), i + offset[b, 1, i, j].item()
            w, h = size[b, 0, i, j].exp().item(), size[b, 1, i, j].exp().item()
            boxes.append([
                (cx - w / 2) * CELL,
                (cy - h / 2) * CELL,
                (cx + w / 2) * CELL,
                (cy + h / 2) * CELL,
                s,
            ])
        out.append(boxes)
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


# --- train / eval --------------------------------------------------------------------


def load(
    paths: list[Path],
) -> tuple[Tensor, Tensor, list[list[list[float]]], list[tuple[str, int]]]:
    toks, flips, boxes, keys = [], [], [], []
    for p in paths:
        d = torch.load(p, weights_only=False)
        toks.append(d["tokens"])
        flips.append(d["tokens_flip"])
        boxes += d["boxes"]
        keys += [(d["job"], f) for f in d["frame"]]
    return torch.cat(toks), torch.cat(flips), boxes, keys


def detection_metrics(
    pred: list[list[list[float]]], gt: list[list[list[float]]], pos: float
) -> dict[str, float]:
    tp = n_pred = n_gt = 0
    for p, g in zip(pred, gt, strict=True):
        g = [b for b in g if b[4] >= pos]  # ruff: ignore[redefined-loop-name]
        n_pred, n_gt = n_pred + len(p), n_gt + len(g)
        used = set()
        for b in p:
            best = max(range(len(g)), key=lambda i: iou(b, g[i]), default=None)
            if best is not None and best not in used and iou(b, g[best]) >= 0.5:
                used.add(best)
                tp += 1
    return {
        "recall50": tp / max(n_gt, 1),
        "precision50": tp / max(n_pred, 1),
        "gt": n_gt,
        "pred": n_pred,
    }


def target_recall(
    pred: list[list[list[float]]], gt: list[list[list[float]]], pos: float
) -> dict[str, float]:
    """How often the head has a box on the pallet the old pipeline would have targeted."""
    from rmind.scripts.pallet_pipeline import pick_target

    ious = []
    for p, g in zip(pred, gt, strict=True):
        target = pick_target([(b[:4], b[4]) for b in g if b[4] >= pos], None)
        if target is not None:
            ious.append(max((iou(b, target[0]) for b in p), default=0.0))
    a = np.array(ious)
    return {
        "target_n": len(a),
        "target_recall50": float((a >= 0.5).mean()),
        "target_recall70": float((a >= 0.7).mean()),
        "target_iou_median": float(np.median(a)),
    }


def predict(
    head: nn.Module, toks: Tensor, device: str, threshold: float
) -> list[list[list[float]]]:
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(toks), 256):
            out += decode(*head(toks[i : i + 256].to(device).float()), threshold)
    return out


def end_to_end(head: nn.Module, args: argparse.Namespace) -> dict[str, float]:
    """Keypoint error / pose success on the hand-labelled held-out frames: head box vs Grounding DINO box."""
    from PIL import Image

    from rmind.pallet.correspondence import NAMES, POINTS_3D
    from rmind.pallet.entry import MEAN as KMEAN
    from rmind.pallet.entry import STD as KSTD
    from rmind.pallet.entry import crop_for_box, load_labels
    from rmind.pallet.geometry import CAM_FORK
    from rmind.pallet.keypoints import PalletKeypointNet
    from rmind.pallet.keypoints import decode as kdecode
    from rmind.pallet.pose import solve
    from rmind.scripts.pallet_poc_zeroshot import D12_ROOT

    ck = torch.load(args.kp_checkpoint, map_location="cpu", weights_only=True)
    net = PalletKeypointNet(ck["backbone"], num_keypoints=len(NAMES), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net = net.to(args.device).eval()
    labels = load_labels(args.kp_labels)
    gdino = json.loads(args.kp_labels.with_suffix(".boxes.json").read_text())
    enc = encoder(args.device)
    names = [n for n in sorted(labels) if n.startswith(VAL_JOBS)]
    res: dict[str, list] = {"head": [], "gdino": []}
    pose_ok: dict[str, list] = {"head": [], "gdino": []}
    found: dict[str, list] = {"head": [], "gdino": []}
    for name in names:
        full = np.array(Image.open(args.images / name).convert("RGB"))
        job, frame = name[:-4].rsplit("_", 1)
        small = policy_frame(D12_ROOT, job, int(frame), full)
        boxes = predict(
            head, tokens(enc, small[None], args.device), args.device, args.threshold
        )[0]
        gt, lab = labels[name]
        centre = np.nanmean(gt[lab], axis=0)
        # the head box that holds the labelled pallet: nearest centre among its boxes
        head_box = min(
            boxes,
            key=lambda b: np.hypot(
                (b[0] + b[2]) / 2 - centre[0], (b[1] + b[3]) / 2 - centre[1]
            ),
            default=None,
        )
        for tag, box in (
            ("head", head_box[:4] if head_box else None),
            ("gdino", gdino.get(name)),
        ):
            found[tag].append(box is not None)
            if box is None:
                pose_ok[tag].append(False)
                continue
            crop = crop_for_box(np.asarray(box), None)
            patch = (
                crop.extract(torch.from_numpy(full).permute(2, 0, 1).float() / 255)
                - KMEAN
            ) / KSTD
            with torch.no_grad():
                px, conf = kdecode(net(patch[None].to(args.device)))
            kp, conf = crop.to_frame(px[0].cpu().numpy()), conf[0].cpu().numpy()
            res[tag].append(np.linalg.norm(kp - gt, axis=1)[lab])
            pose_ok[tag].append(
                solve(
                    CAM_FORK,
                    kp,
                    conf > 0.3,
                    confidence=conf,
                    model=POINTS_3D,
                    sigma_px=3.0,
                )
                is not None
            )
    out = {"e2e_frames": len(names)}
    for tag in res:  # ruff: ignore[dict-index-missing-items]
        e = np.concatenate(res[tag]) if res[tag] else np.array([np.inf])
        # keypoint error over the frames where a box was found; misses count in `found` and `pose`
        out |= {
            f"e2e_{tag}_found": float(np.mean(found[tag])),
            f"e2e_{tag}_median_px": float(np.median(e)),
            f"e2e_{tag}_pck10": float((e < 10).mean()),
            f"e2e_{tag}_pose": float(np.mean(pose_ok[tag])),
        }
    return out


def train(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    run = start(
        "train-proposer",
        vars(args),
        ["proposer", "dinov3-head", f"pos:{args.pos}", *args.tags],
        notes=f"CenterNet head on frozen {ENCODER} layer {LAYER}, Grounding DINO pseudo-labels >= {args.pos}",
        enabled=not args.no_wandb,
    )
    if args.kp_checkpoint is not None:
        use_labels(run, args.kp_labels, "entry-labels")
    paths = sorted(args.cache.glob("*.pt"))
    val_paths = [p for p in paths if p.stem.startswith(VAL_JOBS)]
    tr_toks, tr_flip, tr_boxes, _ = load([p for p in paths if p not in val_paths])
    va_toks, _, va_boxes, _ = load(val_paths)
    logger.info(
        "data",
        train=len(tr_toks),
        val=len(va_toks),
        train_with_pallet=sum(any(b[4] >= args.pos for b in bs) for bs in tr_boxes),
    )
    tgt = [targets(b, args.pos, flip=False) for b in tr_boxes]
    tgt_flip = [targets(b, args.pos, flip=True) for b in tr_boxes]
    stack = lambda ts, k: torch.stack([t[k] for t in ts])  # ruff: ignore[lambda-assignment]
    targets_all = {k: torch.cat([stack(tgt, k), stack(tgt_flip, k)]) for k in tgt[0]}
    x_all = torch.cat([tr_toks, tr_flip])
    head = ProposerHead().to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(x_all) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.05
    )
    best = -1.0
    for epoch in range(args.epochs):
        head.train()
        perm, total = torch.randperm(len(x_all)), 0.0
        for i in range(0, len(x_all) - args.batch + 1, args.batch):
            b = perm[i : i + args.batch]
            x = x_all[b].to(args.device).float()
            t = {k: v[b].to(args.device) for k, v in targets_all.items()}
            heat, size, offset = head(x)
            m = t["mask"][:, None].expand_as(size)
            loss = (
                focal_loss(heat, t["heat"], t["ignore"])
                + 0.1
                * F.l1_loss(size[m], t["size"][m], reduction="sum")
                / m.sum().clamp(min=1)
                * 2
                + F.l1_loss(offset[m], t["offset"][m], reduction="sum")
                / m.sum().clamp(min=1)
                * 2
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        pred = predict(head, va_toks, args.device, args.threshold)
        m = detection_metrics(pred, va_boxes, args.pos) | target_recall(
            pred, va_boxes, args.pos
        )
        logger.info(
            "epoch",
            epoch=epoch,
            loss=round(total / max(len(x_all) // args.batch, 1), 4),
            **{k: round(v, 3) for k, v in m.items()},
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": total / max(len(x_all) // args.batch, 1),
                **{f"val/{k}": v for k, v in m.items()},
            })
        score = m["target_recall50"] + m["recall50"]
        if score > best:
            best = score
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "encoder": ENCODER,
                    "layer": LAYER,
                    "metrics": m,
                    "pos": args.pos,
                },
                args.out / "best.pt",
            )
    head.load_state_dict(
        torch.load(args.out / "best.pt", weights_only=False)["state_dict"]
    )
    final = torch.load(args.out / "best.pt", weights_only=False)["metrics"]
    if args.kp_checkpoint is not None:
        final |= end_to_end(head, args)
    (args.out / "metrics.json").write_text(json.dumps(final, indent=1))
    logger.info(
        "done",
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in final.items()},
    )
    if run is not None:
        run.summary.update({f"best/{k}": v for k, v in final.items()})
        log_model(
            run,
            args.out / "best.pt",
            "proposer-head",
            {"encoder": ENCODER, "layer": LAYER, **final},
        )
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--jobs", nargs="+", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--stride", type=int, default=10)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--device", default="cuda:0")
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--pos",
        type=float,
        default=0.45,
        help="Grounding DINO score from which a box is a pallet",
    )
    t.add_argument(
        "--threshold", type=float, default=0.3, help="head peak score for a detection"
    )
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--kp-checkpoint", type=Path, default=None)
    t.add_argument(
        "--kp-labels",
        type=Path,
        default=Path(
            "/nasa/team-space/nikita/pallet-detector/labels/entry_labels_v2.json"
        ),
    )
    t.add_argument(
        "--images",
        type=Path,
        default=Path("/nasa/team-space/nikita/pallet-detector/images/entry"),
    )
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--tags", nargs="*", default=[], help="extra wandb tags")
    t.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    cache(args) if args.cmd == "cache" else train(args)


if __name__ == "__main__":
    main()
