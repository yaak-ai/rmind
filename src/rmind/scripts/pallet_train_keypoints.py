"""Train `PalletKeypointNet` on box-corner keypoints: PalLoc6D, plus real pseudo-labels.

8 viewpoint-canonical box corners per pallet (`rmind.pallet.palloc6d`), square
crops around the pallet, Gaussian heatmaps at stride 4. Synthetic: the last
`--val` PalLoc6D frames are held out. Real (optional): `--real-train` /
`--real-val` are `scripts.pallet_pseudo_labels` outputs (silhouette-fit labels
on `cam_fork`, val from held-out jobs); each training batch draws
`--real-fraction` of its samples from them. Reports keypoint error in crop pixels
on each validation set and keeps the weights best on real val when given, else
on synthetic val.

    CUDA_VISIBLE_DEVICES=1 uv run --extra pallet python -m rmind.scripts.pallet_train_keypoints \
        --data /nasa/data/d12/yaak/palloc6d/images --out outputs/pallet_kp
"""

import argparse
from pathlib import Path

import torch
from structlog import get_logger
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from rmind.pallet.keypoints import CROP_SIZE, PalletKeypointNet, decode, heatmap_loss
from rmind.pallet.palloc6d import (
    BOX_CORNER_NAMES,
    KeypointCrops,
    palloc6d_items,
    pseudo_label_items,
)

logger = get_logger(__name__)

# heatmap MSE is tiny in absolute terms; scale it to a sensible loss magnitude
LOSS_SCALE = 1000.0


def evaluate(
    net: PalletKeypointNet, loader: DataLoader, device: str
) -> dict[str, float]:
    net.eval()
    errors, visible_all = [], []
    with torch.no_grad():
        for batch in loader:
            heatmaps = net(batch["image"].to(device))
            pixels, _ = decode(heatmaps)
            errors.append((pixels.cpu() - batch["keypoints"]).norm(dim=-1))
            visible_all.append(batch["visible"])
    error = torch.cat(errors)[torch.cat(visible_all)]
    return {
        "median_px": float(error.median()),
        "p90_px": float(error.quantile(0.9)),
        "pck_5pct": float((error < 0.05 * CROP_SIZE).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument(
        "--limit", type=int, default=None, help="use only the first N synthetic frames"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--real-train", type=Path, default=None)
    parser.add_argument("--real-val", type=Path, default=None)
    parser.add_argument("--real-fraction", type=float, default=0.5)
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.data.glob("[0-9]*.json"))[: args.limit]
    synthetic = KeypointCrops(palloc6d_items(files[: -args.val]), train=True)
    val_loaders = {
        "synthetic": DataLoader(
            KeypointCrops(palloc6d_items(files[-args.val :]), train=False),
            batch_size=args.batch,
            num_workers=8,
        )
    }
    parts, weights = [synthetic], [torch.full((len(synthetic),), 1.0 / len(synthetic))]
    if args.real_train is not None:
        real = KeypointCrops(pseudo_label_items(args.real_train), train=True, seed=1)
        parts.append(real)
        weights = [w * (1 - args.real_fraction) for w in weights] + [
            torch.full((len(real),), args.real_fraction / len(real))
        ]
    if args.real_val is not None:
        val_loaders["real"] = DataLoader(
            KeypointCrops(pseudo_label_items(args.real_val), train=False),
            batch_size=args.batch,
            num_workers=8,
        )
    train = ConcatDataset(parts)
    sampler = WeightedRandomSampler(
        torch.cat(weights), num_samples=args.samples_per_epoch, replacement=True
    )
    logger.info(
        "data",
        parts=[len(p) for p in parts],
        val={k: len(v.dataset) for k, v in val_loaders.items()},
        keypoints=BOX_CORNER_NAMES,
    )  # ty: ignore[invalid-argument-type]
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=8,
        drop_last=True,
        persistent_workers=True,
    )

    net = PalletKeypointNet(args.backbone, num_keypoints=len(BOX_CORNER_NAMES)).to(
        args.device
    )
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": net.backbone.parameters(), "lr": 2e-4},
            {"params": head, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-4, 1e-3],
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1,
    )
    best = float("inf")
    for epoch in range(args.epochs):
        net.train()
        total = 0.0
        for batch in train_loader:
            heatmaps = net(batch["image"].to(args.device))
            loss = LOSS_SCALE * heatmap_loss(
                heatmaps,
                batch["targets"].to(args.device),
                torch.ones_like(batch["visible"]).to(args.device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss)
        metrics = {
            f"{name}_{k}": v
            for name, loader in val_loaders.items()
            for k, v in evaluate(net, loader, args.device).items()
        }
        logger.info(
            "epoch",
            epoch=epoch,
            loss=round(total / len(train_loader), 4),
            **{k: round(v, 3) for k, v in metrics.items()},
        )
        score = metrics.get("real_median_px", metrics["synthetic_median_px"])
        if score < best:
            best = score
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "backbone": args.backbone,
                    "keypoints": BOX_CORNER_NAMES,
                    "metrics": metrics,
                },
                args.out / "best.pt",
            )
    logger.info("done", best_median_px=best, checkpoint=str(args.out / "best.pt"))


if __name__ == "__main__":
    main()
