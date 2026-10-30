"""Run a proposer head (`final.pt` of `pallet_proposer_multi --final`) over whole recordings.

Every `--stride`-th frame: frozen policy DINOv3 ViT-S at the head's resolution -> every
pallet box with score >= `--threshold`. Writes one parquet per job: frame, boxes
(x0, y0, x1, y1, score in full-res pixels, then P(loaded) for heads that predict it). Input for pre-labelling and for tracking.
"""

import argparse
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F

from rmind.components.timm_backbone import TimmBackbone
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT
from rmind.scripts.pallet_predict_entry import frames
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import decode, make_head


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.head, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(args.device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )
    jobs = args.jobs or sorted(
        p.name
        for p in D12_ROOT.iterdir()
        if p.name.startswith("20") and (p / "cam_fork" / "video.mp4").is_file()
    )
    for job in jobs:
        if (
            job == "2026-10-22--09-49-39" or (args.out / f"{job}.parquet").exists()
        ):  # byte-identical copy of 09-50-09
            continue
        rows = []
        for idx, images in frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", args.stride, 16
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
                dets = decode(
                    head(enc((x - MEAN.to(args.device)) / STD.to(args.device))),
                    args.threshold,
                )
            rows += [
                {"job": job, "frame": f, "boxes": d}
                for f, d in zip(idx, dets, strict=True)
            ]
        pl.DataFrame(
            rows,
            schema={
                "job": pl.String,
                "frame": pl.Int64,
                "boxes": pl.List(pl.List(pl.Float64)),
            },
        ).write_parquet(args.out / f"{job}.parquet")
        print(job, len(rows), sum(len(r["boxes"]) for r in rows), flush=True)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
