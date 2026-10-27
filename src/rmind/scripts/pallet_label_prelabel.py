"""Re-pre-label Label Studio tasks with the silhouette fit (Grounding DINO + SAM2).

The keypoint net is still weak on real frames, so the box and pocket corners are
projected from the floor-constrained silhouette pose instead, which keeps them
geometrically consistent. Per task image (raw 1920x1080):

    box (Grounding DINO, best-scoring) -> mask (SAM2) -> `silhouette.fit`
    IoU >= --min-iou: project the 8 canonical box corners and the 8 pocket corners
    from the fitted pose (the far bottom corners are left out: on an approach the
    pallet itself hides them); otherwise leave the task without pre-labels

Reads the given task files, writes `--out` with the same tasks and new
predictions (model_version "silhouette").
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rmind.pallet.geometry import CAM_FORK, Pose2D
from rmind.pallet.palloc6d import BOX_CORNER_NAMES, BOX_CORNERS_3D
from rmind.pallet.silhouette import fit
from rmind.scripts.pallet_eval_keypoints import canonical
from rmind.scripts.pallet_label_export import (
    POCKET_CORNER_NAMES,
    POCKET_CORNERS_3D,
    _point,
)
from rmind.scripts.pallet_pipeline import DETECTOR_ID, Segmenter, pick_target
from rmind.scripts.pallet_poc_zeroshot import Proposer

HIDDEN_ON_APPROACH = {"far_bottom_left", "far_bottom_right"}


def main() -> None:
    from PIL import Image

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="labelling folder (images/ inside)"
    )
    parser.add_argument("--tasks", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    proposer = Proposer(args.device, DETECTOR_ID)
    segmenter = Segmenter(args.device)
    tasks = [t for f in args.tasks for t in json.loads(f.read_text())]
    done, fitted = 0, 0
    for task in tasks:
        name = task["data"]["image"].split("d=")[1]
        image = np.array(Image.open(args.root / name).convert("RGB"))
        result = []
        target = pick_target(proposer.detect_all(image[None])[0], None)
        if target is not None:
            mask = segmenter([image], [target[0]])[0]
            silhouette = fit(CAM_FORK, mask, tuple(target[0]))  # ty: ignore[invalid-argument-type]
            if silhouette is not None and silhouette.iou >= args.min_iou:
                fitted += 1
                pose = canonical(Pose2D(*silhouette.pose))
                for names, points in (
                    (BOX_CORNER_NAMES, BOX_CORNERS_3D),
                    (POCKET_CORNER_NAMES, POCKET_CORNERS_3D),
                ):
                    pixels, imaged = CAM_FORK.project(pose.apply(points))
                    result += [
                        _point(n, float(u), float(v))
                        for n, (u, v), ok in zip(names, pixels, imaged, strict=True)
                        if ok and n not in HIDDEN_ON_APPROACH
                    ]
        task["predictions"] = [{"model_version": "silhouette", "result": result}]
        done += 1  # ruff: ignore[enumerate-for-loop]
        if done % 25 == 0:
            print(done, "/", len(tasks), "fitted", fitted, flush=True)  # ruff: ignore[print]
    args.out.write_text(json.dumps(tasks, indent=1))
    print("tasks", len(tasks), "pre-labelled", fitted)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
