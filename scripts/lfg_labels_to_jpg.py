"""Render pilot LFG per-patch labels to JPEGs for a visual sanity check (task §7.1).

For each labelled frame under a drive's lfg_labels directory, decode the packed
`(4, 16, 16)` uint8 blob (see `src/rmind/utils/lfg_labels.py`) and the corresponding
source JPEG, then write four PNGs/JPEGs per frame:

  {frame_idx}_00_seg_overlay.jpg   - seg_label (argmax class), colour-coded, blended over the
                                      224x224 model input
  {frame_idx}_01_seg_purity.jpg    - seg_purity heatmap
  {frame_idx}_02_motion.jpg        - motion heatmap
  {frame_idx}_03_confidence.jpg    - confidence heatmap

Usage:
    nix develop --command uv run python scripts/lfg_labels_to_jpg.py \
        --drive-frames /nasa/drives/yaak/data/Niro096-HQ/2023-01-11--13-47-36/frames/cam_front_left.pii.mp4/576x324 \
        --drive-labels /nasa/drives/yaak/lfg_labels/v1/Niro096-HQ/2023-01-11--13-47-36 \
        --out /home/alex/test_labels \
        --num-frames 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import simplejpeg
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rmind.utils.lfg_labels import decode_lfg_label

# 7 LFG segmentation classes -> distinct RGB colours (arbitrary but stable palette).
SEG_COLORS = np.array(
    [
        [220, 20, 60],  # 0 - red
        [70, 130, 180],  # 1 - steel blue (often sky)
        [107, 142, 35],  # 2 - olive
        [128, 64, 128],  # 3 - purple (often road)
        [244, 164, 96],  # 4 - sandy brown
        [255, 215, 0],  # 5 - gold
        [64, 64, 64],  # 6 - dark gray
    ],
    dtype=np.uint8,
)

MODEL_INPUT_SIZE = 224
CENTER_CROP = (320, 576)


def load_model_input(jpg_path: Path) -> np.ndarray:
    """Reproduce `config/model/yaak/patch_policy/raw.yaml`'s crop+resize, as uint8 HWC RGB."""
    rgb = simplejpeg.decode_jpeg(
        jpg_path.read_bytes(), colorspace="rgb", fastdct=True, fastupsample=True
    )
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.uint8)  # (3, 324, 576)
    x = TF.center_crop(x, list(CENTER_CROP))  # (3, 320, 576)
    x = TF.resize(
        x, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE], antialias=True
    )  # (3, 224, 224)
    return x.permute(1, 2, 0).numpy()  # (224, 224, 3) uint8


def upsample_nearest(plane: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(plane).resize((size, size), Image.NEAREST)
    )


def save_seg_overlay(base_rgb: np.ndarray, seg_label: np.ndarray, out_path: Path) -> None:
    seg_up = upsample_nearest(seg_label, MODEL_INPUT_SIZE)  # (224, 224) uint8 class ids
    color = SEG_COLORS[seg_up]  # (224, 224, 3)
    blended = (0.5 * base_rgb.astype(np.float32) + 0.5 * color.astype(np.float32)).astype(
        np.uint8
    )
    Image.fromarray(blended).save(out_path, quality=95)


def save_heatmap(plane_u8: np.ndarray, out_path: Path) -> None:
    """plane_u8: (16, 16) uint8 -> grayscale heatmap upsampled to 224x224."""
    up = upsample_nearest(plane_u8, MODEL_INPUT_SIZE)
    Image.fromarray(up, mode="L").save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-frames", type=Path, required=True)
    parser.add_argument("--drive-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument(
        "--skip", type=int, default=0, help="skip this many labelled frames before taking --num-frames"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(
        args.drive_labels.glob("*.bin"), key=lambda p: int(p.stem)
    )[args.skip : args.skip + args.num_frames]
    if not label_paths:
        msg = f"no .bin labels found under {args.drive_labels}"
        raise SystemExit(msg)

    for label_path in label_paths:
        frame_idx = int(label_path.stem)
        jpg_path = args.drive_frames / f"{frame_idx:09d}.jpg"
        if not jpg_path.exists():
            print(f"skip {frame_idx}: missing source jpg {jpg_path}", file=sys.stderr)
            continue

        labels = decode_lfg_label(label_path.read_bytes())  # (4, 16, 16) uint8
        seg_label, seg_purity, motion, confidence = labels

        base_rgb = load_model_input(jpg_path)

        prefix = args.out / f"{frame_idx:09d}"
        save_seg_overlay(base_rgb, seg_label, Path(f"{prefix}_00_seg_overlay.jpg"))
        save_heatmap(seg_purity, Path(f"{prefix}_01_seg_purity.jpg"))
        save_heatmap(motion, Path(f"{prefix}_02_motion.jpg"))
        save_heatmap(confidence, Path(f"{prefix}_03_confidence.jpg"))
        print(f"wrote frame {frame_idx}")


if __name__ == "__main__":
    main()
