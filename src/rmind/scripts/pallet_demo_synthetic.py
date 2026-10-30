"""Detector + keypoint net on labelled synthetic frames, drawn against the ground truth.

    uv run python -m rmind.scripts.pallet_demo_synthetic --head .../fold0.pt \
        --keypoints .../best.pt --labels .../multi_test.json --out .../demo --title "..."

Per frame (1280x720 JPEG in `--out`): ground-truth pallet boxes thin white (dashed grey when
`unsure`: tiny or mostly hidden), ground-truth keypoints as white rings; detections coloured by
confidence (red at the threshold -> green at 1) with LOADED / empty from the load head, and the
keypoint net's 12 points on each detection (entry face outline + both pockets). Top left: how
many ground-truth pallets the detector found (centre inside a box). Stitch with ffmpeg.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from structlog import get_logger

from rmind.components.timm_backbone import TimmBackbone
from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import MEAN, STD, crop_for_box
from rmind.pallet.keypoints import PalletKeypointNet
from rmind.pallet.keypoints import decode as decode_keypoints
from rmind.scripts.pallet_demo_detector import (
    OUT_H,
    OUT_W,
    WHITE,
    confidence_colour,
    draw_legend,
)
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER
from rmind.scripts.pallet_proposer_multi import Frames, decode, is_positive, make_head

logger = get_logger(__name__)
SCALE = OUT_W / 1920
# entry face outline and the two pocket openings
LOOPS = (
    (
        "pallet_bottom_left",
        "pallet_bottom_right",
        "pallet_top_right",
        "pallet_top_left",
    ),
    (
        "pocket_l_bottom_left",
        "pocket_l_bottom_right",
        "pocket_l_top_right",
        "pocket_l_top_left",
    ),
    (
        "pocket_r_bottom_left",
        "pocket_r_bottom_right",
        "pocket_r_top_right",
        "pocket_r_top_left",
    ),
)


def pt(xy: np.ndarray) -> tuple[int, int]:
    return int(xy[0] * SCALE), int(xy[1] * SCALE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="multi_*.json")
    parser.add_argument(
        "--entry",
        type=Path,
        default=None,
        help="entry_*.json (default: next to labels)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text())
    entry_path = args.entry or args.labels.with_name(
        args.labels.name.replace("multi_", "entry_")
    )
    gt_points: dict[str, list] = {}
    for key, pts in json.loads(entry_path.read_text()).items():
        gt_points.setdefault(key.split("#")[0], []).extend(pts.values())
    names = sorted(labels)
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(args.device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )
    kk = torch.load(args.keypoints, map_location="cpu", weights_only=False)
    net = PalletKeypointNet(kk["backbone"], num_keypoints=len(NAMES), pretrained=False)
    net.load_state_dict(kk["state_dict"])
    net = net.to(args.device).eval()
    frames = Frames(Path("/"), names, res)
    found = total = 0
    for n, name in enumerate(names):
        x, _ = frames.batch([name], labels, None)
        image = frames.image(name)
        with torch.no_grad():
            dets = decode(head(enc(x.to(args.device))), args.threshold)[0]
        view = cv2.resize(
            (image.permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8),
            (OUT_W, OUT_H),
        )
        positives = [p for p in labels[name] if is_positive(p)]
        for p in labels[name]:
            b = np.array(p["box"]).reshape(2, 2)
            colour = WHITE if is_positive(p) else (120, 120, 120)
            cv2.rectangle(view, pt(b[0]), pt(b[1]), colour, 1)
        for u, v in gt_points.get(name, []):
            cv2.circle(view, pt((u, v)), 4, WHITE, 1)
        hit = sum(
            any(
                b[0] <= (p["box"][0] + p["box"][2]) / 2 <= b[2]
                and b[1] <= (p["box"][1] + p["box"][3]) / 2 <= b[3]
                for b in dets
            )
            for p in positives
        )
        found, total = found + hit, total + len(positives)
        if dets:
            crops = [crop_for_box(np.asarray(b[:4]), None) for b in dets]
            patches = torch.stack([(c.extract(image) - MEAN) / STD for c in crops])
            with torch.no_grad():
                px, conf = decode_keypoints(net(patches.to(args.device)))
            for b, c, p, cf in zip(
                dets, crops, px.cpu().numpy(), conf.cpu().numpy(), strict=True
            ):
                colour = confidence_colour(b[4], args.threshold)
                cv2.rectangle(view, pt(b[:2]), pt(b[2:4]), colour, 2)
                tag = f"{b[4]:.2f}"
                if len(b) > 5:
                    tag = f"{'LOADED' if b[5] >= 0.5 else 'empty'} {tag}"
                cv2.putText(
                    view, tag, (pt(b[:2])[0], pt(b[:2])[1] - 5), 0, 0.45, colour, 1
                )
                kp = c.to_frame(p)
                for loop in LOOPS:
                    idx = [NAMES.index(k) for k in loop]
                    for i, j in zip(idx, idx[1:] + idx[:1], strict=True):
                        if cf[i] > 0.3 and cf[j] > 0.3:
                            cv2.line(view, pt(kp[i]), pt(kp[j]), (255, 0, 255), 2)
                for q, s in zip(kp, cf, strict=True):
                    if s > 0.3:
                        cv2.circle(view, pt(q), 3, (255, 0, 255), -1)
        cv2.rectangle(view, (0, 0), (OUT_W, 52), (0, 0, 0), -1)
        cv2.putText(view, args.title, (10, 20), 0, 0.6, WHITE, 1)
        cv2.putText(
            view,
            f"{Path(name).stem}   found {hit}/{len(positives)} labelled pallets   "
            "white = ground truth, magenta = predicted keypoints",
            (10, 42),
            0,
            0.5,
            WHITE,
            1,
        )
        draw_legend(view, args.threshold)
        cv2.imwrite(
            str(args.out / f"{n:04d}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
    logger.info("done", frames=len(names), found=found, labelled=total)


if __name__ == "__main__":
    main()
