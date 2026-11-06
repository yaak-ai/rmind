"""Frame-to-frame jitter of keypoint nets on real consecutive frames.

Around each labelled pallet of `--labels` (keys `<job>_<frame>.jpg[#i]`, filtered by
`--keys-prefix`) the `--half-window` frames before and after are decoded from the
recording. The DINOv3 detector (`--head`) finds boxes on every frame; the labelled pallet's
box is tracked through the clip (largest IoU with the previous frame), so every keypoint net
gets the same crops. Jitter per point is the second difference |p(t-1) - 2 p(t) + p(t+1)|
in raw frame px: at 30 fps the real motion of an approach is smooth, so this is mostly the
net's frame-to-frame noise. Reported overall, relative to the face width, and per face-width
bucket. Points with confidence < 0.3 in any of the three frames are skipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from structlog import get_logger

from rmind.components.timm_backbone import TimmBackbone
from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import MEAN as KP_MEAN
from rmind.pallet.entry import STD as KP_STD
from rmind.pallet.entry import load_labels
from rmind.pallet.keypoints import CROP_SIZE, Crop, PalletKeypointNet, RectCrop
from rmind.pallet.keypoints import decode as decode_kp
from rmind.scripts.pallet_label_export import read_frames
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import decode, iou, make_head

logger = get_logger(__name__)
MIN_IOU = 0.3


def track(dets: list[list], start: int, box: list[float]) -> list[list[float] | None]:
    """The box of one pallet in every frame: from `start`, follow the largest IoU."""
    out: list[list[float] | None] = [None] * len(dets)
    out[start] = box
    for step in (1, -1):
        prev, t = box, start + step
        while 0 <= t < len(dets):
            best = max(dets[t], key=lambda b: iou(prev, b[:4]), default=None)
            if best is not None and iou(prev, best[:4]) >= MIN_IOU:
                prev = best[:4]
                out[t] = prev
            t += step
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--keys-prefix", nargs="*", default=[])
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--half-window", type=int, default=30)
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    labels = {
        k: v
        for k, v in load_labels(args.labels).items()
        if not args.keys_prefix or k.startswith(tuple(args.keys_prefix))
    }
    # one clip per labelled frame (several pallets of a frame share it)
    clips: dict[str, list[str]] = {}
    for k in sorted(labels):
        clips.setdefault(k.split("#")[0], []).append(k)
    clips = dict(list(clips.items())[: args.max_clips])
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(args.device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )
    nets = {}
    for path in args.checkpoints:
        c = torch.load(path, map_location="cpu", weights_only=False)
        net = PalletKeypointNet(
            c["backbone"], num_keypoints=len(NAMES), pretrained=False
        )
        net.load_state_dict(c["state_dict"])
        nets[f"{path.parent.name}"] = (
            net.to(args.device).eval(),
            c.get("crop_shape"),
            c.get("crop_margin", 0.3),
        )
    jitter: dict[str, list[tuple[float, float]]] = {n: [] for n in nets}
    for image, keys in clips.items():
        job, frame = image[:-4].rsplit("_", 1)
        f0 = int(frame)
        idx = list(range(max(0, f0 - args.half_window), f0 + args.half_window + 1))
        video = D12_ROOT / job / "cam_fork" / "video.mp4"
        try:
            frames = read_frames(video, idx)
        except Exception as e:  # ruff: ignore[blind-except] - clip past the end of a recording
            logger.warning("clip skipped", image=image, error=str(e)[:80])
            continue
        imgs = [
            torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255
            for f in frames
        ]
        dets = []
        with torch.no_grad():
            for i in range(0, len(imgs), 16):
                x = torch.stack(imgs[i : i + 16]).to(args.device)
                x = F.interpolate(
                    x,
                    size=(res[1], res[0]),
                    mode="bilinear",
                    antialias=True,
                    align_corners=False,
                )
                dets += decode(
                    head(enc((x - MEAN.to(args.device)) / STD.to(args.device))), 0.3
                )
        centre = idx.index(f0)
        for key in keys:
            xy, ok = labels[key]
            lb = [*xy[ok].min(0), *xy[ok].max(0)]
            first = max(dets[centre], key=lambda b: iou(lb, b[:4]), default=None)
            if first is None or iou(lb, first[:4]) < MIN_IOU / 2:
                continue
            boxes = track(dets, centre, first[:4])
            for name, (net, shape, margin) in nets.items():
                pts, conf, widths = [], [], []
                with torch.no_grad():
                    for img, box in zip(imgs, boxes, strict=True):
                        if box is None:
                            pts.append(None)
                            conf.append(None)
                            widths.append(None)
                            continue
                        if shape:
                            crop = RectCrop.around(tuple(box), margin)
                            patch = crop.extract(img, tuple(shape))
                            scale = np.array([crop.w / shape[0], crop.h / shape[1]])
                        else:
                            crop = Crop.around(tuple(box))
                            patch = crop.extract(img)
                            scale = np.array([crop.side / CROP_SIZE] * 2)
                        px, cf = decode_kp(
                            net(((patch - KP_MEAN) / KP_STD)[None].to(args.device))
                        )
                        pts.append(
                            px[0].cpu().numpy() * scale + np.array([crop.x0, crop.y0])
                        )
                        conf.append(cf[0].cpu().numpy())
                        widths.append(box[2] - box[0])
                for t in range(1, len(pts) - 1):
                    if pts[t - 1] is None or pts[t] is None or pts[t + 1] is None:
                        continue
                    good = (conf[t - 1] > 0.3) & (conf[t] > 0.3) & (conf[t + 1] > 0.3)
                    d2 = np.linalg.norm(pts[t - 1] - 2 * pts[t] + pts[t + 1], axis=1)[
                        good
                    ]
                    jitter[name] += [(float(v), widths[t]) for v in d2]
        logger.info("clip", image=image, pallets=len(keys))
    results = {}
    for name, rows in jitter.items():
        if not rows:
            continue
        v, w = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        res_n = {
            "median_px": float(np.median(v)),
            "p90_px": float(np.quantile(v, 0.9)),
            "median_rel_face_pct": float(np.median(v / w) * 100),
            "n": len(v),
        }
        for lo, hi in ((0, 150), (150, 400), (400, 4000)):
            m = (w >= lo) & (w < hi)
            if m.any():
                res_n[f"median_px_w{lo}-{hi}"] = float(np.median(v[m]))
        results[name] = res_n
        logger.info(
            "jitter",
            model=name,
            **{k: round(x, 3) if isinstance(x, float) else x for k, x in res_n.items()},
        )
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
