"""Labelling tools for the pallet detector: render a view, change a label, look again.

Built for labelling by eye from rendered images: absolute coordinates read off a
picture are 10-30 px off, relative moves ("3 px left") are not, so keypoints are
placed coarsely and then nudged on a zoomed, ticked view centred on each point.

    grid IMAGE [X0 Y0 X1 Y1]   gridded view of a frame (full frame: lines every 50 px;
                               region: zoomed, lines every 10 px), with the frame's
                               box labels drawn on top
    nudge IMAGE [OFFSETS]      keypoints: apply {"kp": [dx, dy] | "drop"} to the
                               frame's keypoint labels, render one zoomed tile per point
    boxes IMAGE JSON           box labels: a list of pallets
                               {"box": [x0, y0, x1, y1], "floor", "face", "trunc", "occl", "unsure"},
                               or {"edit": {i: [dx0, dy0, dx1, dy1]}}, or {"drop": [i, ...]}
    eval-boxes                 a detector's cached boxes vs the box labels: recall at
                               IoU 0.5 and "some box covers >= 70 % of the pallet"

Labels, images and renders live in the workspace (`--workspace`):
`labels/entry_labels_*.json` (keypoints), `labels/multi_labels_*.json` (boxes),
`images/entry`, `images/multi`, renders in `renders/`.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path))
    if im is None:
        msg = f"cannot read {path}"
        raise FileNotFoundError(msg)
    return im


WORKSPACE = Path("/nasa/team-space/nikita/pallet-detector")
TICK_R, TICK_Z = 36, 5  # nudge tiles: +-36 px around the point at 5x


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def grid(args: argparse.Namespace) -> Path:
    import cv2

    im = _read(args.images / args.image)
    labels = _load(args.labels).get(args.image, [])
    full = args.region is None
    x0, y0, x1, y1 = args.region or (0, 0, im.shape[1], im.shape[0])
    step, major_every = (50, 100) if full else (10, 50)
    z = 1600 / (x1 - x0)
    for p in labels:
        b = p["box"]
        cv2.rectangle(
            im,
            (int(b[0]), int(b[1])),
            (int(b[2]), int(b[3])),
            (0, 255, 0),
            2 if full else 1,
        )
    c = cv2.resize(
        im[y0:y1, x0:x1],
        None,
        fx=z,
        fy=z,
        interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA,
    )
    for x in range((x0 // step + 1) * step, x1, step):
        p, major = int((x - x0) * z), x % major_every == 0
        cv2.line(
            c, (p, 0), (p, c.shape[0]), (0, 255, 255) if major else (0, 110, 110), 1
        )
        if major:
            cv2.putText(c, str(x), (p + 2, 14), 0, 0.45, (0, 255, 255), 1)
    for y in range((y0 // step + 1) * step, y1, step):
        p, major = int((y - y0) * z), y % major_every == 0
        cv2.line(
            c, (0, p), (c.shape[1], p), (0, 255, 255) if major else (0, 110, 110), 1
        )
        if major:
            cv2.putText(c, str(y), (2, p - 3), 0, 0.45, (0, 255, 255), 1)
    for i, p in enumerate(labels):
        q = (int((p["box"][0] - x0) * z) + 3, int((p["box"][1] - y0) * z) + 18)
        cv2.putText(c, str(i), q, 0, 0.7, (0, 0, 0), 4)
        cv2.putText(c, str(i), q, 0, 0.7, (0, 255, 0), 2)
    out = args.workspace / "renders" / f"grid_{args.image}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), c)
    return out


def nudge(args: argparse.Namespace) -> Path:
    import cv2

    labels = _load(args.labels)
    pts = {k: list(map(float, v)) for k, v in labels.get(args.image, {}).items()}
    for k, d in (json.loads(args.offsets) if args.offsets else {}).items():
        if d == "drop":
            pts.pop(k, None)
        else:
            u, v = pts.get(k, (0.0, 0.0))
            pts[k] = [u + d[0], v + d[1]]
    labels[args.image] = {k: [round(a, 1), round(b, 1)] for k, (a, b) in pts.items()}
    args.labels.write_text(json.dumps(labels, indent=1))

    r, z = TICK_R, TICK_Z
    im = cv2.copyMakeBorder(
        _read(args.images / args.image), r + 2, r + 2, r + 2, r + 2, cv2.BORDER_CONSTANT
    )
    tiles = []
    for k, (u, v) in pts.items():
        # sub-pixel affine crop so the point sits exactly at the tile centre
        m = np.array(
            [[z, 0, -(u + 2) * z + 0.5 * z], [0, z, -(v + 2) * z + 0.5 * z]],
            dtype=np.float32,
        )
        tile = cv2.warpAffine(im, m, (2 * r * z, 2 * r * z), flags=cv2.INTER_CUBIC)
        c = r * z
        for d in range(-r, r + 1, 2):
            p, major = c + d * z, d % 10 == 0
            col = (0, 255, 255) if major else (0, 90, 90)
            cv2.line(tile, (p, 0), (p, 16 if major else 8), col, 1)
            cv2.line(tile, (0, p), (16 if major else 8, p), col, 1)
            if major and d:
                cv2.putText(tile, f"{d:+d}", (p - 12, 30), 0, 0.45, (0, 255, 255), 1)
                cv2.putText(tile, f"{d:+d}", (18, p + 5), 0, 0.45, (0, 255, 255), 1)
        cv2.line(tile, (c - 14, c), (c + 14, c), (0, 0, 255), 1)
        cv2.line(tile, (c, c - 14), (c, c + 14), (0, 0, 255), 1)
        cv2.putText(tile, k, (4, 2 * c - 8), 0, 0.55, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)])
    out = args.workspace / "renders" / f"nudge_{args.image}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(sheet, None, fx=0.6, fy=0.6))
    return out


def boxes(args: argparse.Namespace) -> int:
    labels = _load(args.labels)
    cur, arg = labels.get(args.image, []), json.loads(args.json)
    if isinstance(arg, list):
        cur = arg
    elif "edit" in arg:
        for i, d in arg["edit"].items():
            cur[int(i)]["box"] = [
                a + b for a, b in zip(cur[int(i)]["box"], d, strict=True)
            ]
    elif "drop" in arg:
        cur = [p for i, p in enumerate(cur) if i not in arg["drop"]]
    labels[args.image] = cur
    args.labels.write_text(json.dumps(labels, indent=1))
    return len(cur)


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _cover(gt: list[float], det: list[float]) -> float:
    """Share of the labelled box a detection covers (detectors often box the whole load)."""
    ix = max(0.0, min(gt[2], det[2]) - max(gt[0], det[0]))
    iy = max(0.0, min(gt[3], det[3]) - max(gt[1], det[1]))
    return ix * iy / ((gt[2] - gt[0]) * (gt[3] - gt[1]))


def eval_boxes(args: argparse.Namespace) -> dict[str, float]:
    """Detections from a proposer cache (`pallet_proposer_head cache`) vs the box labels."""
    import torch

    labels = _load(args.labels)
    dets: dict[str, list] = {}
    for job in {n.rsplit("_", 1)[0] for n in labels}:
        d = torch.load(args.cache / f"{job}.pt", weights_only=False, map_location="cpu")
        dets |= {
            f"{job}_{f:06d}.jpg": b for f, b in zip(d["frame"], d["boxes"], strict=True)
        }
    out = {}
    for thr in args.thresholds:
        n = hit = cov = 0
        for name, pallets in labels.items():
            det = [b for b in dets.get(name, []) if b[4] >= thr]
            gts = [
                p["box"]
                for p in pallets
                if p["floor"] and p["face"] and not p.get("unsure")
            ]
            n += len(gts)
            hit += sum(any(_iou(g, d) >= 0.5 for d in det) for g in gts)
            cov += sum(any(_cover(g, d) >= 0.7 for d in det) for g in gts)
        out |= {f"recall50@{thr}": hit / max(n, 1), f"cover70@{thr}": cov / max(n, 1)}
    out["pallets"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("image")
    g.add_argument("region", nargs="*", type=int)
    g.add_argument("--set", default="multi", choices=["multi", "entry"])
    g.add_argument("--labels-version", default="v1")
    k = sub.add_parser("nudge")
    k.add_argument("image")
    k.add_argument("offsets", nargs="?")
    k.add_argument("--labels-version", default="v2")
    b = sub.add_parser("boxes")
    b.add_argument("image")
    b.add_argument("json")
    b.add_argument("--labels-version", default="v1")
    e = sub.add_parser("eval-boxes")
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.45])
    e.add_argument("--labels-version", default="v1")
    args = parser.parse_args()

    kind = (
        "entry"
        if args.cmd == "nudge" or getattr(args, "set", "multi") == "entry"
        else "multi"
    )
    args.labels = (
        args.workspace / "labels" / f"{kind}_labels_{args.labels_version}.json"
    )
    args.images = args.workspace / "images" / kind
    if args.cmd == "grid":
        args.region = args.region or None
        print(grid(args))  # ruff: ignore[print]
    elif args.cmd == "nudge":
        print(nudge(args))  # ruff: ignore[print]
    elif args.cmd == "boxes":
        print(args.image, boxes(args), "pallets")  # ruff: ignore[print]


if __name__ == "__main__":
    main()
