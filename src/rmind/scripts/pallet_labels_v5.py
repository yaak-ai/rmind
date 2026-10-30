"""Convert our label sets to the v5 format (`rmind.pallet.schema`), one JSONL per source.

    real_v5.jsonl       multi-pallet boxes (multi_labels_v3, with loaded tags) merged with the
                        12-point keypoint labels (entry_labels_v4; "frame.jpg#i" keys belong to
                        box i of that frame, plain keys are single-pallet yard/hall frames)
    synthetic_v5.jsonl  Isaac renders (niantic warehouse v2): every pallet, keypoints with
                        depth-checked visibility, metric pose relative to the camera

Splits: real "test" = the 10-12 recordings (detector test day), "val" = the keypoint
validation recordings (09-10 15-17-38, 10-02 15-48-24, 10-22 drive from frame 30000),
"train" = the rest; synthetic: every 10th frame is "val".
"""

import argparse
import json
from pathlib import Path

from rmind.pallet.schema import KEYPOINTS, box_of

W = Path("/nasa/team-space/nikita/pallet-detector")
VAL = ("2026-09-10--15-17-38", "2026-10-02--15-48-24")


def real_split(job: str, frame: int) -> str:
    if job.startswith("2026-10-12"):
        return "test"
    if job.startswith(VAL) or (job.startswith("2026-10-22") and frame >= 30000):
        return "val"
    return "train"


def convert_real(out: Path) -> int:
    multi = json.loads((W / "labels/multi_labels_v3.json").read_text())
    entry = json.loads((W / "labels/entry_labels_v4.json").read_text())
    entry_load = json.loads((W / "labels/entry_labels_v4.load.json").read_text())
    names = sorted({*multi, *(k.split("#")[0] for k in entry)})
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for name in names:
            job, frame = name[:-4].rsplit("_", 1)
            image = W / ("images/multi" if name in multi else "images/entry") / name
            pallets = []
            for i, p in enumerate(multi.get(name, [])):
                kp = entry.get(f"{name}#{i}", {})
                pallets.append({
                    "id": f"{name}#{i}",
                    "box": p["box"],
                    "loaded": {"loaded": True, "empty": False}.get(p.get("load")),
                    "type": None,
                    "on_floor": p.get("floor"),
                    "truncated": bool(p.get("trunc")),
                    "occluded": bool(p.get("occl")),
                    "unsure": bool(p.get("unsure")),
                    "carried": bool(p.get("carried")),
                    "keypoints": {k: [*kp[k], "visible"] for k in KEYPOINTS if k in kp},
                    "pose": None,
                })
            if (
                name in entry
            ):  # single labelled pallet of a yard/hall frame, no box labels
                kp = entry[name]
                pallets.append({
                    "id": name,
                    "box": box_of(list(kp.values()), 1920, 1080),
                    "loaded": {"loaded": True, "empty": False}.get(
                        entry_load.get(name)
                    ),
                    "type": None,
                    "on_floor": True,
                    "truncated": False,
                    "occluded": False,
                    "unsure": False,
                    "carried": False,
                    "keypoints": {k: [*kp[k], "visible"] for k in KEYPOINTS if k in kp},
                    "pose": None,
                })
            f.write(
                json.dumps({
                    "image": str(image),
                    "source": "real",
                    "dataset": "multi_v3+entry_v4",
                    "job": job,
                    "frame": int(frame),
                    "camera": "cam_fork",
                    "width": 1920,
                    "height": 1080,
                    "split": real_split(job, int(frame)),
                    "pallets": pallets,
                })
                + "\n"
            )
            n += 1
    return n


def convert_synthetic(root: Path, out: Path) -> int:
    views = {
        v["frame"]: v for v in json.loads((root / "views.json").read_text())["frames"]
    }
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for path in sorted((root / "labels").glob("frame_*.json")):
            d = json.loads(path.read_text())
            if d.get("status") != "ok":
                continue
            view = views.get(d["frame"], {})
            mode = view.get("mode")
            w, h = d["calibration"]["width"], d["calibration"]["height"]
            pallets = []
            for i, p in enumerate(d["pallets"]):
                e = p["labels"]["entry_12"]
                kp = {
                    k: [*e["keypoints"][k], e["visibility"][k]]
                    for k in KEYPOINTS
                    if k in e["keypoints"]
                    and e["in_frame"].get(k)
                    and e["in_front_of_camera"].get(k)
                }
                pose = p.get("relative_pose", {}).get("camera", {})
                centre = pose.get("entry_face_center")
                # bottom-of-stack first; a pallet with another pallet or cargo on it counts as loaded
                loaded = mode == "loaded" if i == 0 else None
                if mode == "stacked" and i == 0:
                    loaded = True
                pallets.append({
                    "id": f"{d.get('source_run')}/{p['instance_id']}",
                    "box": box_of(list(kp.values()), w, h),
                    "loaded": loaded,
                    "type": "eur",
                    "on_floor": i == 0 if mode == "stacked" else True,
                    "truncated": len(kp) < len(KEYPOINTS),
                    "occluded": any(v[2] == "hidden" for v in kp.values()),
                    "unsure": False,
                    "carried": False,
                    "keypoints": kp,
                    "pose": None
                    if centre is None
                    else {
                        "entry_centre_m": [
                            centre["forward_m"],
                            centre["left_m"],
                            centre["up_m"],
                        ],
                        "yaw_deg": pose.get("entry_face_yaw_deg"),
                        "source": "gt",
                    },
                })
            f.write(
                json.dumps({
                    "image": str(root / d["image"]),
                    "source": "synthetic",
                    "dataset": root.name,
                    "job": None,
                    "frame": d["frame"],
                    "camera": "cam_fork",
                    "width": w,
                    "height": h,
                    "split": "val" if d["frame"] % 10 == 0 else "train",
                    "pallets": pallets,
                    "camera_model": d.get("projection"),
                })
                + "\n"
            )
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=W / "labels/v5")
    parser.add_argument(
        "--synthetic",
        type=Path,
        nargs="*",
        default=[
            Path("/nasa/data/d12/yaak/pallet-niantic-20261028-warehouse-labels-v2")
        ],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("real", convert_real(args.out / "real_v5.jsonl"))  # ruff: ignore[print]
    for root in args.synthetic:
        print(  # ruff: ignore[print]
            root.name,
            convert_synthetic(root, args.out / f"synthetic_{root.name}.jsonl"),
        )


if __name__ == "__main__":
    main()
