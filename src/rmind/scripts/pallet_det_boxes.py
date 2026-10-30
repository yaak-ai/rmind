"""Crop boxes from the DINOv3 pallet detector for keypoint-labelled pallets.

The keypoint net should train and be scored on the crops it will get at runtime: those of
the multi-pallet detector on the policy's frozen DINOv3 (`pallet_proposer_multi` heads), not
Grounding DINO's or the label's own. For every image of the given label files the detector
runs once; each labelled pallet takes the detection that best covers its keypoints (largest
IoU with the keypoint bounds, or containing their bottom centre). Writes `{key: box | null}`
next to `--out`; null = the detector missed this pallet.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from structlog import get_logger

from rmind.components.timm_backbone import TimmBackbone
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import decode, iou, make_head

logger = get_logger(__name__)


def match(points: np.ndarray, dets: list[list[float]]) -> list[float] | None:
    kb = [*points.min(0).tolist(), *points.max(0).tolist()]
    base = ((kb[0] + kb[2]) / 2, kb[3])
    best = max(dets, key=lambda b: iou(kb, b[:4]), default=None)
    if best is not None and iou(kb, best[:4]) > 0.2:
        return best[:4]
    inside = [
        b for b in dets if b[0] <= base[0] <= b[2] and b[1] - 20 <= base[1] <= b[3] + 20
    ]
    return (
        min(inside, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4] if inside else None
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="image folder for keys that are file names",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(args.device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )
    labels: dict[str, dict] = {}
    for f in args.labels:
        labels |= json.loads(f.read_text())
    by_image: dict[str, list[str]] = {}
    for key in labels:
        by_image.setdefault(key.split("#")[0], []).append(key)
    out, missed = {}, 0
    for n, (image, keys) in enumerate(sorted(by_image.items())):
        path = Path(image) if image.startswith("/") else args.images / image
        x = (
            torch
            .from_numpy(np.array(Image.open(path).convert("RGB")))
            .permute(2, 0, 1)
            .float()[None]
            .to(args.device)
            / 255
        )
        x = F.interpolate(
            x,
            size=(res[1], res[0]),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        with torch.no_grad():
            dets = decode(
                head(enc((x - MEAN.to(args.device)) / STD.to(args.device))),
                args.threshold,
            )[0]
        for key in keys:
            box = match(np.array(list(labels[key].values())), dets)
            out[key] = box
            missed += box is None
        if n % 200 == 0:
            logger.info("progress", images=n, of=len(by_image))
    args.out.write_text(json.dumps(out))
    logger.info("done", pallets=len(out), missed=missed)


if __name__ == "__main__":
    main()
