"""Linear probe: frozen DINOv2 features -> map-GT speed class (Phase-0).

Question: can the policy's vision stack READ speed limits at all at its
input resolution? Samples frames across the map-GT sidecar drives
(stratified by the 13-class max-speed vocabulary, UNKNOWN excluded),
extracts FROZEN features with the winner recipe -- timm
``vit_small_patch14_dinov2.lvd142m``, final-layer post-norm patch tokens
(DINO-WM's ``x_norm_patchtokens``) mean-pooled, after the model's exact
input transform (576x324 JPEG -> CenterCrop 320x576 -> Resize 224x224 ->
ImageNet normalize; see config/model/yaak/patch_policy/raw.yaml) -- and
fits a multinomial logistic regression to the speed class.

Reports held-out accuracy vs the majority-class baseline and per-class
recall. The split is BY DRIVE (deterministic hash), so near-duplicate
frames never straddle train/test.

Caveat: this upper-bounds "sees the sign" only loosely -- limits correlate
with scene type (motorway vs city), so part of the accuracy is scene
recognition. The per-class confusion (e.g. 50 vs 30 inside city) is the
sharper signal.

Needs torch/timm (GPU strongly recommended) + the rbyte-style frames dir:
  <data-root>/<vehicle>/<drive>/frames/cam_front_left.pii.mp4/576x324/%09d.jpg

Usage (box worktree, PYTHONPATH pointing at it):
  python -m rmind.scripts.map_gt.linear_probe \
      [--sidecar-root caches/map_gt] [--data-root /nasa/drives/yaak/data] \
      [--per-class 400] [--device cuda] \
      [--features-cache diag_results/map_probe/linear_probe_features.npz] \
      [--out diag_results/map_probe/linear_probe.md]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import Tensor

from rmind.components.map_context import (
    MAX_SPEED_NUM_SPECIAL,
    MAX_SPEED_UNKNOWN_ID,
    MAX_SPEED_UNLIMITED_ID,
    MAX_SPEED_VOCAB_KMH,
    MAX_SPEED_VOCAB_SIZE,
    MAX_SPEED_WALK_ID,
    MAX_SPEED_WALK_MAX_KMH,
)

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
IMAGE_RESIZE = (224, 224)
CENTER_CROP = (320, 576)
MIN_SPACING_S = 5.0
TEST_FRACTION_MOD = 5  # drive-hash % 5 == 0 -> test (~20 %)

CLASS_NAMES = ["UNKNOWN", "UNLIMITED", "WALK"] + [
    f"{v:g}" for v in MAX_SPEED_VOCAB_KMH
]


def speed_class(kmh: np.ndarray) -> np.ndarray:
    """Vectorized mirror of rmind.components.map_context.MaxSpeedTokenizer."""
    x = np.asarray(kmh, dtype=np.float64)
    table = np.asarray(MAX_SPEED_VOCAB_KMH)
    nearest = np.abs(x[..., None] - table).argmin(axis=-1) + MAX_SPEED_NUM_SPECIAL
    ids = np.where(x <= MAX_SPEED_WALK_MAX_KMH, MAX_SPEED_WALK_ID, nearest)
    ids = np.where(x < 0.0, MAX_SPEED_UNLIMITED_ID, ids)
    return np.where(np.isnan(x), MAX_SPEED_UNKNOWN_ID, ids).astype(np.int64)


def frame_path(data_root: Path, drive: str, frame_idx: int) -> Path:
    return (
        data_root
        / drive
        / "frames"
        / "cam_front_left.pii.mp4"
        / "576x324"
        / f"{frame_idx:09d}.jpg"
    )


def build_sample_table(
    sidecar_root: Path, data_root: Path, *, per_class: int, seed: int
) -> pl.DataFrame:
    """Stratified (class-balanced, drive-spread, >= 5 s spaced) frame sample."""
    rng = np.random.default_rng(seed)
    pools: dict[int, list[tuple[str, int]]] = {}
    for sc in sorted(sidecar_root.glob("*/*.parquet")):
        drive = f"{sc.parent.name}/{sc.stem}"
        df = (
            pl.read_parquet(
                sc, columns=["frame_idx", "time_stamp_us", "max_speed_kmh"]
            )
            .sort("time_stamp_us")
            .with_columns(
                pl.col("max_speed_kmh")
                .map_batches(
                    lambda s: pl.Series(speed_class(s.to_numpy())),
                    return_dtype=pl.Int64,
                )
                .alias("cls")
            )
            .filter(pl.col("cls") != MAX_SPEED_UNKNOWN_ID)
        )
        # thin to >= MIN_SPACING_S within each (drive, class)
        for (cls,), g in df.group_by("cls"):
            ts = g["time_stamp_us"].to_numpy()
            fi = g["frame_idx"].to_numpy()
            keep_ts = -np.inf
            pool = pools.setdefault(int(cls), [])
            for t, f in zip(ts, fi):
                if (t - keep_ts) / 1e6 >= MIN_SPACING_S:
                    pool.append((drive, int(f)))
                    keep_ts = t

    rows: list[tuple[str, int, int]] = []
    for cls, pool in sorted(pools.items()):
        # round-robin over drives so no single drive dominates a class
        by_drive: dict[str, list[tuple[str, int]]] = {}
        for item in pool:
            by_drive.setdefault(item[0], []).append(item)
        for items in by_drive.values():
            rng.shuffle(items)
        order = sorted(by_drive)
        picked: list[tuple[str, int]] = []
        budget = int(per_class * 1.3)  # oversample; missing jpgs filtered later
        i = 0
        while len(picked) < budget and any(by_drive[d] for d in order):
            d = order[i % len(order)]
            if by_drive[d]:
                picked.append(by_drive[d].pop())
            i += 1
        rows += [(drive, f, cls) for drive, f in picked]

    table = pl.DataFrame(
        rows, schema={"drive": pl.Utf8, "frame_idx": pl.Int64, "cls": pl.Int64},
        orient="row",
    )
    # drop rows whose jpg is missing, then trim back to per_class
    exists = [
        frame_path(data_root, r["drive"], r["frame_idx"]).exists()
        for r in table.iter_rows(named=True)
    ]
    table = (
        table.filter(pl.Series(exists))
        .group_by("cls", maintain_order=True)
        .head(per_class)
    )
    return table.with_columns(
        pl.col("drive")
        .map_elements(
            lambda d: int(hashlib.sha1(d.encode()).hexdigest(), 16)
            % TEST_FRACTION_MOD
            == 0,
            return_dtype=pl.Boolean,
        )
        .alias("is_test")
    )


@torch.no_grad()
def extract_features(
    table: pl.DataFrame, data_root: Path, *, device: torch.device, batch_size: int
) -> np.ndarray:
    from torchvision.transforms.v2 import (  # noqa: PLC0415
        CenterCrop,
        Compose,
        Normalize,
        Resize,
        ToDtype,
    )

    from rmind.components.timm_backbone import TimmBackbone  # noqa: PLC0415

    transform = Compose([
        CenterCrop(CENTER_CROP),
        Resize(list(IMAGE_RESIZE)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    backbone = (
        TimmBackbone(
            model_name=MODEL_NAME,
            img_size=list(IMAGE_RESIZE),
            norm_patch_tokens=True,
        )
        .to(device)
        .eval()
    )

    feats: list[Tensor] = []
    rows = list(table.iter_rows(named=True))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        imgs = np.stack([
            np.asarray(
                Image.open(frame_path(data_root, r["drive"], r["frame_idx"]))
            )
            for r in chunk
        ])  # (b, h, w, c) uint8
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
        x = transform(x)
        patches = backbone(x)  # (b, c, h, w) post-norm patch tokens
        feats.append(patches.mean(dim=(-2, -1)).float().cpu())
        if (start // batch_size) % 10 == 0:
            print(f"features {start + len(chunk)}/{len(rows)}")
    return torch.cat(feats).numpy()


def fit_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    num_classes: int,
    seed: int,
    device: torch.device,
    steps: int = 2000,
) -> np.ndarray:
    """Multinomial logistic regression (torch, full-batch); returns test preds."""
    torch.manual_seed(seed)
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    xt = torch.tensor((x_train - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, device=device)
    xv = torch.tensor((x_test - mu) / sd, dtype=torch.float32, device=device)

    # inverse-frequency class weights (the strata are only balanced pre-filter)
    counts = torch.bincount(yt, minlength=num_classes).float().clamp(min=1)
    weight = counts.sum() / (num_classes * counts)

    lin = torch.nn.Linear(xt.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(xt), yt, weight=weight)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(xv).argmax(dim=-1).cpu().numpy()


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar-root", type=Path, default=Path("caches/map_gt"))
    ap.add_argument("--data-root", type=Path, default=Path("/nasa/drives/yaak/data"))
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--features-cache",
        type=Path,
        default=Path("diag_results/map_probe/linear_probe_features.npz"),
        help="npz cache: reused when present (skips sampling + extraction)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("diag_results/map_probe/linear_probe.md")
    )
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.features_cache.exists():
        print(f"reusing features cache {args.features_cache}")
        z = np.load(args.features_cache, allow_pickle=False)
        feats, labels = z["features"], z["labels"]
        is_test, drives = z["is_test"], z["drives"]
    else:
        table = build_sample_table(
            args.sidecar_root, args.data_root, per_class=args.per_class,
            seed=args.seed,
        )
        print(table.group_by("cls", "is_test").len().sort("cls", "is_test"))
        feats = extract_features(
            table, args.data_root, device=device, batch_size=args.batch_size
        )
        labels = table["cls"].to_numpy().astype(np.int64)
        is_test = table["is_test"].to_numpy()
        drives = table["drive"].to_numpy().astype(str)
        args.features_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.features_cache,
            features=feats, labels=labels, is_test=is_test, drives=drives,
        )
        print(f"features cached -> {args.features_cache}")

    # collapse to classes present in BOTH splits (13-class ids kept in report)
    present = sorted(
        set(np.unique(labels[~is_test])) & set(np.unique(labels[is_test]))
    )
    keep = np.isin(labels, present)
    dropped = sorted(set(np.unique(labels)) - set(present))
    remap = {c: i for i, c in enumerate(present)}
    y = np.array([remap[c] for c in labels[keep]])
    x, test = feats[keep], is_test[keep]

    y_pred = fit_logreg(
        x[~test], y[~test], x[test],
        num_classes=len(present), seed=args.seed, device=device,
    )
    y_true = y[test]

    acc = float((y_pred == y_true).mean())
    train_majority = int(np.bincount(y[~test]).argmax())
    baseline = float((y_true == train_majority).mean())

    conf = np.zeros((len(present), len(present)), dtype=int)
    np.add.at(conf, (y_true, y_pred), 1)
    recalls = conf.diagonal() / conf.sum(1).clip(min=1)
    balanced_acc = float(recalls[conf.sum(1) > 0].mean())

    names = [CLASS_NAMES[c] for c in present]
    lines = [
        "# Linear probe: frozen DINOv2 features -> speed class",
        "",
        f"Backbone: `{MODEL_NAME}` @ {IMAGE_RESIZE[0]}x{IMAGE_RESIZE[1]} "
        "(center-crop 320x576 -> resize; final-layer post-norm patch tokens, "
        "mean-pooled) -- the dinov2_dinowm winner recipe.",
        f"Samples: {len(y)} frames / {len(np.unique(drives[keep]))} drives "
        f"({int((~test).sum())} train / {int(test.sum())} test, split BY "
        f"DRIVE); {len(present)}/{MAX_SPEED_VOCAB_SIZE} vocab classes present "
        f"in both splits"
        + (
            f" (dropped, missing from one split: "
            f"{[CLASS_NAMES[c] for c in dropped]})"
            if dropped
            else ""
        )
        + ".",
        "",
        f"- held-out accuracy: **{acc:.1%}**",
        f"- majority-class baseline (train majority = {names[train_majority]}): "
        f"**{baseline:.1%}**",
        f"- balanced accuracy (mean per-class recall): **{balanced_acc:.1%}** "
        f"(chance {1 / len(present):.1%})",
        "",
        "## Per-class recall (test)",
        "",
        "| class | n test | recall |",
        "|---|---|---|",
    ]
    for i, name in enumerate(names):
        lines.append(f"| {name} | {conf[i].sum()} | {recalls[i]:.1%} |")

    lines += [
        "",
        "## Confusion matrix (rows = true, cols = pred)",
        "",
        "| true \\ pred | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for i, name in enumerate(names):
        lines.append(
            f"| {name} | " + " | ".join(str(v) for v in conf[i]) + " |"
        )
    lines += [
        "",
        "Caveat: limits correlate with scene type, so scene recognition "
        "contributes; confusion WITHIN a scene type (e.g. 30 vs 50 city) is "
        "the sharper 'reads the sign' signal.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
