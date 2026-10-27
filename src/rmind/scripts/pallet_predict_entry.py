"""Predict entry-face keypoints on every recording (every `--stride`-th frame).

Per frame: Grounding DINO pallet box -> crop -> keypoint net -> 12 points (raw
1920x1080 px, 720p drives upscaled) + per-point confidence -> floor-constrained
pose (`pose.solve`, free pitch) for a geometric consistency check. Writes one
parquet per job with, per frame: box, points, confidence, pose, RANSAC inliers,
reprojection RMS, and a 256-d crop embedding (PCA-free: the backbone's pooled
last stage, L2-normalised, randomly projected) for diversity-aware sampling.

"Hard" signals for active labelling: low mean confidence, few inliers, high
reprojection RMS, no pose; diversity from the embeddings.
"""

import argparse
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch

from rmind.pallet.correspondence import NAMES, POINTS_3D
from rmind.pallet.entry import MEAN, STD, crop_for_box
from rmind.pallet.geometry import CAM_FORK
from rmind.pallet.keypoints import PalletKeypointNet, decode
from rmind.pallet.pose import solve
from rmind.scripts.pallet_pipeline import DETECTOR_ID, pick_target
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, Proposer

W, H = 1920, 1080


def frames(
    video: Path, stride: int, batch: int
) -> Iterator[tuple[list[int], np.ndarray]]:
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    cmd = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=not(mod(n\\,{stride})),scale={W}:{H}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE) as p:  # ruff: ignore[subprocess-without-shell-equals-true]
        assert p.stdout is not None  # ruff: ignore[assert]
        idx, buf, n = [], [], 0
        while raw := p.stdout.read(W * H * 3):
            buf.append(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
            idx.append(n * stride)
            n += 1
            if len(buf) == batch:
                yield idx, np.stack(buf)
                idx, buf = [], []
        if buf:
            yield idx, np.stack(buf)


class EmbedNet(torch.nn.Module):
    """Keypoint net that also returns a pooled embedding of its deepest features."""

    def __init__(self, net: PalletKeypointNet) -> None:
        super().__init__()
        self.net = net
        self.proj: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.net.backbone(x)
        pooled = torch.nn.functional.normalize(feats[-1].mean(dim=(-2, -1)), dim=-1)
        if self.proj is None:
            g = torch.Generator(device="cpu").manual_seed(0)
            self.proj = (torch.randn(pooled.shape[-1], 256, generator=g) / 16).to(
                pooled
            )
        return self.net(x), pooled @ self.proj


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = PalletKeypointNet(
        ckpt["backbone"], num_keypoints=len(NAMES), pretrained=False
    )
    net.load_state_dict(ckpt["state_dict"])
    model = EmbedNet(net).to(args.device).eval()
    proposer = Proposer(args.device, DETECTOR_ID)
    jobs = args.jobs or sorted(
        p.name
        for p in D12_ROOT.iterdir()
        if p.name.startswith("20") and (p / "cam_fork" / "video.mp4").is_file()
    )
    for job in jobs:
        if (args.out / f"{job}.parquet").exists():
            continue
        rows = []
        for idx, images in frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", args.stride, args.batch
        ):
            cands = proposer.detect_all(images)
            for f, image, c in zip(idx, images, cands, strict=True):
                target = pick_target(c, None)
                row: dict = {"job": job, "frame": f, "has_box": target is not None}
                if target is not None:
                    box = np.asarray(target[0])
                    crop = crop_for_box(box, None)
                    patch = (
                        crop.extract(
                            torch.from_numpy(image).permute(2, 0, 1).float() / 255
                        )
                        - MEAN
                    ) / STD
                    with torch.no_grad():
                        heat, emb = model(patch[None].to(args.device))
                        px, conf = decode(heat)
                    raw = crop.to_frame(px[0].cpu().numpy())
                    conf = conf[0].cpu().numpy()
                    est = solve(
                        CAM_FORK,
                        raw,
                        conf > 0.3,
                        confidence=conf,
                        model=POINTS_3D,
                        sigma_px=3.0,
                    )
                    row |= {
                        "box": box.tolist(),
                        "box_score": target[1],
                        "kp": raw.tolist(),
                        "conf": conf.tolist(),
                        "conf_mean": float(conf.mean()),
                        "emb": emb[0].cpu().numpy().tolist(),
                        "pose": None if est is None else list(est.pose),
                        "inliers": 0 if est is None else int(est.inliers.sum()),
                        "rms_px": None if est is None else est.rms_px,
                    }
                rows.append(row)
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            args.out / f"{job}.parquet"
        )
        print(job, len(rows), flush=True)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
