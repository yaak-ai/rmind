"""Train the pallet teacher (`pallet.teacher`): keypoints + loaded + type + floor-free 3D pose.

Data, all as pallet crops:

    real hand labels (--labels)      keypoints (hand-label head), loaded where tagged
    pseudo-labels (--pseudo-labels)  keypoints (hand-label head), loaded where tagged
    Veigel renders (--synthetic)     keypoints (synthetic head), type, metric pose, loaded
                                     (trusted when "loaded", or "empty" on the floor: the
                                     generator's flag misses the warehouse's own rack cartons)

`--synthetic` is a `pallet_synthetic_veigel --labels-out` directory (`entry_train.json` +
`.boxes.json` + `.pose.json`). Sampling mass: hand 1, pseudo `--pseudo-weight`, synthetic
`--synthetic-weight` (each relative to the hand-label total), elevated (stacked / rack)
synthetic pallets x `--elevated-weight`, loaded real pallets x `--loaded-weight`.

Loss: `LOSS_SCALE` x (hand-head + synthetic-head + face-centre heatmap MSE) + `--aux-weight` x
(loaded and type BCE + depth and yaw L1). Validation is the real val jobs' hand-labelled keypoints
(the hand-label head), as in `pallet_train_entry`; `best.pt` is the best real median px,
`last.pt` the final weights. The pose head is scored after training by `pallet_eval_teacher`
on the held-out Veigel scene - never used to pick a checkpoint.

    direnv exec . uv run --no-sync python -m rmind.scripts.pallet_train_teacher \
        --labels .../entry_labels_v4.json --images .../images/entry \
        --synthetic .../teacher_v1/veigel_relaxed --out .../teacher_v1/hrnet_w18
"""

import argparse
import json
from pathlib import Path

import torch
from structlog import get_logger
from torch.utils.data import DataLoader, WeightedRandomSampler

from rmind.pallet.entry import EntryKeypoints, load_labels
from rmind.pallet.keypoints import decode, heatmap_loss
from rmind.pallet.teacher import (
    TYPES,
    PalletTeacherNet,
    TeacherCrops,
    aux_losses,
    centre_loss,
)
from rmind.pallet.tracking import log_model, start, use_labels

logger = get_logger(__name__)
LOSS_SCALE = 1000.0
LOAD = {"loaded": 1, "empty": 0}


def synthetic_meta(directory: Path, split: str) -> tuple[dict, dict, dict]:
    """Veigel labels, crop boxes and per-pallet teacher metadata for one split."""
    labels = load_labels(directory / f"entry_{split}.json")
    boxes = json.loads((directory / f"entry_{split}.boxes.json").read_text())
    poses = json.loads((directory / f"entry_{split}.pose.json").read_text())
    meta = {}
    for name, r in poses.items():
        m = {"synthetic": True, "type": TYPES.index(r["type"]), "pose": r}
        if r["loaded"] or r["floor"]:
            m["loaded"] = int(r["loaded"])
        meta[name] = m
    return labels, boxes, meta


def evaluate(
    net: PalletTeacherNet, loader: DataLoader, device: str
) -> dict[str, float]:
    """Hand-label-head keypoint error (raw px) and load accuracy on the real val crops."""
    net.eval()
    errors, hits, known = [], 0, 0
    with torch.no_grad():
        for b in loader:
            out = net.forward_all(b["image"].to(device), b["geometry"].to(device))
            pixels, _ = decode(out["heatmaps"])
            err = (pixels.cpu() - b["keypoints"]).norm(dim=-1) * b["scale"][:, None]
            errors.append(err[b["visible"]])
            mask = b["loaded"] >= 0
            hits += int(((out["loaded"].cpu() > 0).float() == b["loaded"])[mask].sum())
            known += int(mask.sum())
    e = torch.cat(errors)
    return {
        "median_px": float(e.median()),
        "p90_px": float(e.quantile(0.9)),
        "pck5": float((e < 5).float().mean()),
        "pck10": float((e < 10).float().mean()),
        "n": len(e),
        "loaded_acc": hits / max(known, 1),
        "loaded_n": known,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", default="hrnet_w18.ms_aug_in1k")
    parser.add_argument(
        "--val-jobs",
        nargs="+",
        default=[
            "2026-09-10--15-17-38",
            "2026-10-02--15-48-24",
            "2026-10-22--09-50-09_03",
            "2026-10-22--09-50-09_04",
            "2026-10-22--09-50-09_05",
        ],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-labels", type=Path, nargs="*", default=[])
    parser.add_argument("--pseudo-weight", type=float, default=3.0)
    parser.add_argument("--synthetic-weight", type=float, default=2.0)
    parser.add_argument("--elevated-weight", type=float, default=3.0)
    parser.add_argument("--loaded-weight", type=float, default=2.0)
    parser.add_argument("--boxes", type=Path, nargs="*", default=[])
    parser.add_argument("--quality-aug", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tags", nargs="*", default=[], help="extra wandb tags")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    run = start(
        "train-teacher",
        vars(args),
        ["teacher", "keypoints", "pose-3d", "entry-face", args.backbone, *args.tags],
        notes=f"teacher {args.backbone}: keypoints + loaded + type + floor-free pose; "
        f"{args.labels.name} + Veigel {args.synthetic.name}",
        enabled=not args.no_wandb,
    )
    use_labels(run, args.labels, "entry-labels")

    is_val = lambda n: any(n.startswith(j) for j in args.val_jobs)  # ruff: ignore[lambda-assignment]
    labels = load_labels(args.labels)
    boxes: dict = {}
    for f in args.boxes:
        boxes |= json.loads(f.read_text())
    load_tags: dict[str, str] = {}
    for f in [args.labels, *args.pseudo_labels]:
        if f.with_suffix(".load.json").exists():
            load_tags |= json.loads(f.with_suffix(".load.json").read_text())
    pseudo: dict = {}
    for extra in args.pseudo_labels:
        use_labels(run, extra, "pseudo-labels")
        pseudo |= {n: v for n, v in load_labels(extra).items() if not is_val(n)}
    syn_labels, syn_boxes, meta = synthetic_meta(args.synthetic, "train")
    boxes |= syn_boxes
    for name in [*labels, *pseudo]:
        if load_tags.get(name) in LOAD:
            meta[name] = {"loaded": LOAD[load_tags[name]]}
    train_labels = {n: v for n, v in labels.items() if not is_val(n)}
    n_hand = len(train_labels)
    train_labels |= pseudo | syn_labels
    val_labels = {n: v for n, v in labels.items() if is_val(n)}

    train = TeacherCrops(
        EntryKeypoints(
            args.images,
            train_labels,
            boxes,
            train=True,
            seed=args.seed,
            quality_aug=args.quality_aug,
        ),
        meta,
    )
    val = TeacherCrops(
        EntryKeypoints(args.images, val_labels, boxes, train=False), meta
    )

    def weight(name: str) -> float:
        if name in syn_labels:
            elevated = not meta[name]["pose"]["floor"]
            return (
                args.synthetic_weight
                * n_hand
                / len(syn_labels)
                * (args.elevated_weight if elevated else 1.0)
            )
        w = args.pseudo_weight * n_hand / max(len(pseudo), 1) if name in pseudo else 1.0
        return w * (args.loaded_weight if load_tags.get(name) == "loaded" else 1.0)

    weights = [weight(name) for name, *_ in train.base.items]
    logger.info(
        "data",
        hand=n_hand,
        pseudo=len(pseudo),
        synthetic=len(syn_labels),
        synthetic_elevated=sum(not meta[n]["pose"]["floor"] for n in syn_labels),
        val=len(val),
        backbone=args.backbone,
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        sampler=WeightedRandomSampler(
            weights, num_samples=args.samples_per_epoch, replacement=True
        ),
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=4)

    net = PalletTeacherNet(args.backbone).to(args.device)
    head = [p for n, p in net.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(net.backbone.parameters()), "lr": args.lr},
            {"params": head, "lr": 10 * args.lr},
        ],
        weight_decay=0.05,
    )
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, 10 * args.lr], total_steps=steps, pct_start=0.1
    )

    def save(path: Path, metrics: dict) -> None:
        torch.save(
            {
                "state_dict": net.state_dict(),
                "backbone": args.backbone,
                "kind": "teacher",
                "metrics": metrics,
            },
            path,
        )

    best = float("inf")
    for epoch in range(args.epochs):
        net.train()
        totals: dict[str, float] = {}
        for b in train_loader:
            b = {k: v.to(args.device) for k, v in b.items()}  # ruff: ignore[redefined-loop-name]
            out = net.forward_all(b["image"], b["geometry"])
            syn = b["synthetic"][:, None]
            parts = {
                "hm_real": LOSS_SCALE
                * heatmap_loss(out["heatmaps"], b["targets"], b["labelled"] & ~syn),
                "hm_syn": LOSS_SCALE
                * heatmap_loss(out["heatmaps_syn"], b["targets"], b["labelled"] & syn),
                "hm_centre": LOSS_SCALE * centre_loss(out, b),
                **aux_losses(out, b),
            }
            loss = (
                parts["hm_real"]
                + parts["hm_syn"]
                + parts["hm_centre"]
                + args.aux_weight
                * sum(v for k, v in parts.items() if not k.startswith("hm_"))
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v) / len(train_loader)
        m = evaluate(net, val_loader, args.device)
        logger.info(
            "epoch",
            epoch=epoch,
            **{k: round(v, 4) for k, v in totals.items()},
            **{k: round(v, 3) for k, v in m.items()},
        )
        if run is not None:
            run.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in totals.items()},
                **{f"val/{k}": v for k, v in m.items()},
            })
        if m["median_px"] < best:
            best = m["median_px"]
            save(args.out / "best.pt", m)
    save(args.out / "last.pt", m)
    logger.info("done", best_median_px=round(best, 2))
    if run is not None:
        run.summary.update({"best/median_px": best})
        log_model(
            run,
            args.out / "last.pt",
            "pallet-teacher",
            {"backbone": args.backbone, "labels": args.labels.name, **m},
        )
        run.finish()


if __name__ == "__main__":
    main()
