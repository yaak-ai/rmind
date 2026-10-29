"""Demo videos of the multi-pallet proposer head (`pallet_proposer_multi` k-fold checkpoints).

Each frame: the frozen policy DINOv3 ViT-S at the head's resolution -> CenterNet head ->
every pallet box above `--threshold`, drawn with its score. The k-fold heads never saw
their own test block; per frame the fold whose test frames are nearest in time (same
recording) is used, so labelled stretches are shown by a model that did not train on
them. Recordings without labels use fold 0.

    uv run --extra pallet python -m rmind.scripts.pallet_demo_detector \
        --heads .../proposer_multi_512_v2_all --out .../demos/detector \
        --jobs 2026-10-22--09-50-09:30900:31800
"""

import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from rmind.components.timm_backbone import TimmBackbone
from rmind.scripts.pallet_pipeline import VideoWriter
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, iter_frames
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import Head, decode

OUT_W, OUT_H = 1280, 720
RED, WHITE = (60, 60, 235), (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--heads",
        type=Path,
        required=True,
        help="output dir of a pallet_proposer_multi run",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True, help="job:first:last")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = []
    for p in sorted(args.heads.glob("fold*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        head = Head().to(args.device).eval()
        head.load_state_dict(ck["state_dict"])
        tests = [
            (n[:-4].rsplit("_", 1)[0], int(n[:-4].rsplit("_", 1)[1]))
            for n in ck["test"]
        ]
        folds.append((head, tests, tuple(ck["res"])))
    res = folds[0][2]
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )

    def fold_for(job: str, frame: int) -> int:
        best = min(
            (
                (abs(f - frame), k)
                for k, (_, tests, _) in enumerate(folds)
                for j, f in tests
                if j == job
            ),
            default=(0, 0),
        )
        return best[1]

    for spec in args.jobs:
        job, first, last = spec.split(":")
        first, last = int(first), int(last)
        writer = VideoWriter(
            args.out / f"{job}_{first:06d}-{last:06d}.mp4",
            OUT_W,
            OUT_H,
            fps=30 / args.stride,
        )
        index = first
        for images in iter_frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", first, last, args.stride, 8
        ):
            x = (
                torch.from_numpy(images).permute(0, 3, 1, 2).float().to(args.device)
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
                feats = enc((x - MEAN.to(args.device)) / STD.to(args.device))
            for b, image in enumerate(images):
                k = fold_for(job, index)
                with torch.no_grad():
                    boxes = decode(*folds[k][0](feats[b : b + 1]), args.threshold)[0]
                view = image[:, :, ::-1].copy()
                for x0, y0, x1, y1, s in boxes:
                    cv2.rectangle(view, (int(x0), int(y0)), (int(x1), int(y1)), RED, 3)
                    cv2.putText(
                        view,
                        f"{s:.2f}",
                        (int(x0), max(int(y0) - 8, 24)),
                        0,
                        1.0,
                        RED,
                        2,
                    )
                view = cv2.resize(view, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
                text = f"{job}  frame {index}   pallets: {len(boxes)}   (head fold {k}, {res[0]}x{res[1]}, thr {args.threshold})"
                cv2.putText(view, text, (14, 34), 0, 0.8, (0, 0, 0), 4)
                cv2.putText(view, text, (14, 34), 0, 0.8, WHITE, 2)
                writer.write(view)
                index += args.stride
        writer.close()
        print("done", spec, flush=True)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
