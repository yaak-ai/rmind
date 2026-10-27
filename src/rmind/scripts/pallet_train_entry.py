"""Fine-tune a keypoint net on the hand-labelled entry-face frames (12 points).

Validation is by JOB (`--val-jobs`), never by frame: frames of one approach look
alike, a frame split would measure memory. Reports keypoint error in RAW image
pixels (crop error times the crop scale) on labelled, in-crop points: median,
p90, and PCK at 5 / 10 px. Keeps the best-on-val weights.

Detector boxes for the labelled frames are computed once (Grounding DINO) and
cached next to the labels, so crops at train time look like inference crops.

    CUDA_VISIBLE_DEVICES=1 uv run --extra pallet python -m rmind.scripts.pallet_train_entry \
        --labels /nasa/data/d12/yaak/pallet-labels/entry_labels_v1.json \
        --images /nasa/data/d12/yaak/pallet-labels/entry2 --backbone convnext_base.fb_in22k_ft_in1k --out ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from structlog import get_logger
from torch.utils.data import DataLoader, RandomSampler

from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import EntryKeypoints, load_labels
from rmind.pallet.keypoints import PalletKeypointNet, decode, heatmap_loss
from rmind.pallet.tracking import log_model, start, use_labels

logger = get_logger(__name__)
LOSS_SCALE = 1000.0


def detector_boxes(
    images: Path, names: list[str], cache: Path, device: str
) -> dict[str, list[float] | None]:
    boxes = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [n for n in names if n not in boxes]
    if todo:
        from PIL import Image

        from rmind.scripts.pallet_pipeline import DETECTOR_ID, pick_target
        from rmind.scripts.pallet_poc_zeroshot import Proposer

        proposer = Proposer(device, DETECTOR_ID)
        for n in todo:
            image = np.array(Image.open(images / n).convert("RGB"))
            target = pick_target(proposer.detect_all(image[None])[0], None)
            boxes[n] = target[0] if target is not None else None
        cache.write_text(json.dumps(boxes), encoding="utf-8")
    return boxes


def evaluate(
    net: PalletKeypointNet, loader: DataLoader, device: str
) -> dict[str, float]:
    net.eval()
    errors = []
    with torch.no_grad():
        for b in loader:
            pixels, _ = decode(net(b["image"].to(device)))
            err = (pixels.cpu() - b["keypoints"]).norm(dim=-1) * b["scale"][:, None]
            errors.append(err[b["visible"]])
    e = torch.cat(errors)
    return {
        "median_px": float(e.median()),
        "p90_px": float(e.quantile(0.9)),
        "pck5": float((e < 5).float().mean()),
        "pck10": float((e < 10).float().mean()),
        "n": len(e),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", default="convnext_base.fb_in22k_ft_in1k")
    parser.add_argument(
        "--val-jobs",
        nargs="+",
        default=[
            "2026-09-10--15-17-38",
            "2026-10-02--15-48-24",
            "2026-10-22--09-50-09_02",
        ],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=1600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--train-blocks",
        type=int,
        default=None,
        help="ViT only: fine-tune just the top N transformer blocks (+ head); None = all",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="checkpoint to start from (same keypoints)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tags", nargs="*", default=[], help="extra wandb tags")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run = start(
        "train-keypoints",
        vars(args),
        ["keypoints", "entry-face", args.backbone, *args.tags],
        notes=f"{args.backbone} on {args.labels.name}, val jobs {' '.join(args.val_jobs)}",
        enabled=not args.no_wandb,
    )
    use_labels(run, args.labels, "entry-labels")

    labels = load_labels(args.labels)
    boxes = detector_boxes(
        args.images, sorted(labels), args.labels.with_suffix(".boxes.json"), args.device
    )
    is_val = lambda n: any(n.startswith(j) for j in args.val_jobs)  # ruff: ignore[lambda-assignment]
    train_labels = {n: v for n, v in labels.items() if not is_val(n)}
    val_labels = {n: v for n, v in labels.items() if is_val(n)}
    train = EntryKeypoints(args.images, train_labels, boxes, train=True)
    val = EntryKeypoints(args.images, val_labels, boxes, train=False)
    logger.info("data", train=len(train), val=len(val), backbone=args.backbone)
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        sampler=RandomSampler(
            train, replacement=True, num_samples=args.samples_per_epoch
        ),
        num_workers=8,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=4)

    net = PalletKeypointNet(args.backbone, num_keypoints=len(NAMES)).to(args.device)
    if args.init is not None:
        net.load_state_dict(
            torch.load(args.init, map_location="cpu", weights_only=True)["state_dict"]
        )
    if args.train_blocks is not None:
        # freeze the patch embedding and all but the top N blocks; the multi-layer
        # fusion head reads layers 2/5/8/11, so frozen lower layers still feed it
        blocks = [
            n
            for n, _ in net.backbone.named_parameters()
            if ".blocks." in f".{n}" or n.startswith("blocks.")
        ]
        depth = 1 + max(int(n.split("blocks.")[1].split(".")[0]) for n in blocks)
        for n, p in net.backbone.named_parameters():
            idx = int(n.split("blocks.")[1].split(".")[0]) if "blocks." in n else -1
            p.requires_grad = idx >= depth - args.train_blocks
        logger.info(
            "freeze",
            depth=depth,
            trainable_blocks=args.train_blocks,
            trainable_backbone_params=sum(
                p.numel() for p in net.backbone.parameters() if p.requires_grad
            ),
        )
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in net.backbone.parameters() if p.requires_grad],
                "lr": args.lr,
            },
            {"params": head, "lr": 10 * args.lr},
        ],
        weight_decay=0.05,
    )
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, 10 * args.lr], total_steps=steps, pct_start=0.1
    )
    best = float("inf")
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for b in train_loader:
            loss = LOSS_SCALE * heatmap_loss(
                net(b["image"].to(args.device)),
                b["targets"].to(args.device),
                b["labelled"].to(args.device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += float(loss)
        m = evaluate(net, val_loader, args.device)
        logger.info(
            "epoch",
            epoch=epoch,
            loss=round(total / len(train_loader), 4),
            **{k: round(v, 3) for k, v in m.items()},
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                "train/loss": total / len(train_loader),
                **{f"val/{k}": v for k, v in m.items()},
            })
        if m["median_px"] < best:
            best = m["median_px"]
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "backbone": args.backbone,
                    "keypoints": NAMES,
                    "metrics": m,
                },
                args.out / "best.pt",
            )
    logger.info("done", best_median_px=round(best, 2))
    if run is not None:
        best_metrics = torch.load(
            args.out / "best.pt", map_location="cpu", weights_only=True
        )["metrics"]
        run.summary.update({f"best/{k}": v for k, v in best_metrics.items()})
        log_model(
            run,
            args.out / "best.pt",
            "keypoint-net",
            {"backbone": args.backbone, "labels": args.labels.name, **best_metrics},
        )
        run.finish()


if __name__ == "__main__":
    main()
