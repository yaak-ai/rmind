"""Veigel synthetic renders (Isaac Sim, 1280x720, five scenes) -> our detector and keypoint labels.

The renders are upscaled to the 1920x1080 cam_fork frame (x1.5) and written as JPEG, so every
pipeline reads them like real frames. Per frame, every pallet with a native tight 2D box (the
visible pixels of the pallet mesh, not its load) becomes a detector box; tiny (< `--min-box`
px wide) or mostly hidden (occlusion > `--max-occlusion`) ones are marked `unsure`, which the
detector ignores rather than counts as missed. Keypoints are the visible-only 12-point maps
(`nikita_entry_instances.json`); a pallet enters the keypoint set with at least `--min-points`
of them on an entry face at least `--min-width` px wide, cropped by its tight box.

Splits by scene: `--test-scenes` (default batch-04) is the synthetic test set, the rest train.
Writes into `--out`:
    images/rgb_NNNN.jpg
    multi_{train,test}.json          {abs image: [{box, load, floor, unsure, type, id}]}
    entry_{train,test}.json          {abs image#sI: {keypoint: [u, v]}} + .boxes.json + .load.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from structlog import get_logger

logger = get_logger(__name__)
SCALE = 1.5  # 1280x720 -> 1920x1080


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nasa/data/synthetic/pallets/20261029-veigel-entry-500"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/nasa/team-space/nikita/pallet-detector/datasets/synthetic/veigel-entry-500"
        ),
    )
    parser.add_argument("--test-scenes", nargs="+", default=["batch-04"])
    parser.add_argument("--min-box", type=float, default=24.0)
    parser.add_argument("--max-occlusion", type=float, default=0.8)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-width", type=float, default=60.0)
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    root = args.root
    views = {
        Path(v["image"]).name: v
        for v in json.loads((root / "views.json").read_text())["frames"]
    }
    entry = json.loads((root / "nikita_entry_instances.json").read_text())
    boxes_dir = root / "images/PalletView/bounding_box_2d_tight"
    multi: dict[str, dict] = {"train": {}, "test": {}}
    kps: dict[str, dict] = {"train": {}, "test": {}}
    crops: dict[str, dict] = {"train": {}, "test": {}}
    load: dict[str, dict] = {"train": {}, "test": {}}
    for name, view in sorted(views.items()):
        split = "test" if view["source_run"] in args.test_scenes else "train"
        idx = name[4:8]  # rgb_NNNN.png
        out_image = args.out / "images" / f"rgb_{idx}.jpg"
        if not out_image.exists():
            Image.open(root / view["image"]).convert("RGB").resize(
                (1920, 1080), Image.Resampling.BICUBIC
            ).save(out_image, quality=92)
        full = json.loads((root / f"labels/frame_{int(idx):06d}.json").read_text())
        by_prim = {p["prim_path"]: p for p in full["pallets"]}
        meta = {p["instance_id"]: p for p in entry.get(name, [])}
        tight = np.load(boxes_dir / f"bounding_box_2d_tight_{idx}.npy")
        prims = json.loads(
            (boxes_dir / f"bounding_box_2d_tight_prim_paths_{idx}.json").read_text()
        )
        pallets = []
        for row, prim in zip(tight, prims, strict=True):
            p = by_prim.get(prim)
            if p is None:  # not a pallet (loads, racks, crates, ...)
                continue
            box = [float(row[c]) * SCALE for c in ("x_min", "y_min", "x_max", "y_max")]
            box[2] += SCALE  # inclusive pixel bounds
            box[3] += SCALE
            up = p["relative_pose"]["camera"]["pallet_center"]["up_m"]
            pallets.append({
                "id": p["instance_id"],
                "box": box,
                # `loaded` only counts the generator's own loads: rack pallets carrying the
                # warehouse's native cartons are tagged empty. So only "loaded" is trusted.
                "load": "loaded" if p["loaded"] else None,
                "type": p["pallet_type"],
                "floor": bool(
                    up < -0.25
                ),  # camera at 0.44 m: a floor pallet centre is ~0.37 m below
                "unsure": bool(
                    box[2] - box[0] < args.min_box
                    or float(row["occlusionRatio"]) > args.max_occlusion
                ),
            })
            kp = {
                k: [u * SCALE, v * SCALE]
                for k, (u, v) in meta
                .get(p["instance_id"], {})
                .get("keypoints", {})
                .items()
            }
            us = [u for u, _ in kp.values()]
            if len(kp) >= args.min_points and max(us) - min(us) >= args.min_width:
                key = f"{out_image}#s{len(pallets) - 1}"
                kps[split][key], crops[split][key] = kp, box
                if pallets[-1]["load"]:
                    load[split][key] = pallets[-1]["load"]
        multi[split][str(out_image)] = pallets
    for split in ("train", "test"):
        (args.out / f"multi_{split}.json").write_text(json.dumps(multi[split]))
        base = args.out / f"entry_{split}.json"
        base.write_text(json.dumps(kps[split]))
        base.with_suffix(".boxes.json").write_text(json.dumps(crops[split]))
        base.with_suffix(".load.json").write_text(json.dumps(load[split]))
        boxes = [p for v in multi[split].values() for p in v]
        logger.info(
            "split",
            split=split,
            frames=len(multi[split]),
            boxes=len(boxes),
            positives=sum(not p["unsure"] for p in boxes),
            loaded=sum(p["load"] == "loaded" and not p["unsure"] for p in boxes),
            untagged=sum(p["load"] is None and not p["unsure"] for p in boxes),
            keypoint_pallets=len(kps[split]),
        )


if __name__ == "__main__":
    main()
