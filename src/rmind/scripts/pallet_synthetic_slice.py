"""Prepare one Veigel synthetic slice for the synthetic-data report (scaling + slices).

A slice (e.g. `/nasa/data/synthetic/pallets/20261101-veigel-20k/slice-01`) ships our label
format already (`multi_*.json` boxes, `entry_*.json` 12 points, `labels_v5.jsonl` with
per-pallet metadata). This writes, into `--out`:

    multi_train_<N>.json, entry_train_<N>.json (+ .boxes.json, .load.json)
        nested random subsets of N training FRAMES (`--sizes`), so a bigger subset
        contains the smaller ones; keypoint crops use the pallet's box, as a detector gives
    multi_test.json, entry_test.json (+ .boxes.json, .load.json, .meta.json)
        the slice's test layouts; `.meta.json` per pallet: distance, lateral offset,
        stack size / position, rack level, loaded, type, face width, box centre x
        (fisheye position), for error slices

Keys stay `<abs image>#sI` with I the pallet's index in that frame's `labels_v5` list.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument(
        "--min-width", type=float, default=60.0, help="entry-face width in px"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for line in (args.root / "labels_v5.jsonl").read_text().splitlines():
        d = json.loads(line)
        frames[d["image"]] = d

    def usable(pts: dict) -> bool:
        us = [u for u, _ in pts.values()]
        return len(pts) >= args.min_points and max(us) - min(us) >= args.min_width

    # keypoint pallets: enough points on a face wide enough to place them
    entry = {
        s: {
            k: v
            for k, v in json.loads((args.root / f"entry_{s}.json").read_text()).items()
            if usable(v)
        }
        for s in ("train", "test")
    }
    multi = {
        s: json.loads((args.root / f"multi_{s}.json").read_text())
        for s in ("train", "test")
    }

    def pallet(key: str) -> dict:
        image, idx = key.split("#s")
        return frames[image]["pallets"][int(idx)]

    # the key's index must point at the same pallet: compare one keypoint
    bad = 0
    for key, pts in entry["test"].items():
        name, (u, v) = next(iter(pts.items()))
        q = pallet(key)["keypoints"].get(name)
        bad += q is None or abs(q[0] - u) > 0.5 or abs(q[1] - v) > 0.5
    if bad:
        msg = f"{bad} test keys do not match labels_v5 pallets by index"
        raise ValueError(msg)

    def write(name: str, keys: list[str], split: str, *, meta: bool) -> None:
        labels = {k: entry[split][k] for k in keys}
        base = args.out / f"{name}.json"
        base.write_text(json.dumps(labels))
        base.with_suffix(".boxes.json").write_text(
            json.dumps({k: pallet(k)["box"] for k in keys})
        )
        base.with_suffix(".load.json").write_text(
            json.dumps({k: "loaded" if pallet(k)["loaded"] else "empty" for k in keys})
        )
        if meta:
            out = {}
            for k in keys:
                p = pallet(k)
                xy = np.array(list(labels[k].values()))
                fwd, left, up = (
                    p["pose"]["entry_centre_m"] if p.get("pose") else (None,) * 3
                )
                out[k] = {
                    "distance_m": fwd,
                    "left_m": left,
                    "up_m": up,
                    "stack_size": p.get("stack_size") or 1,
                    "stack_position": p.get("stack_position") or 0,
                    "rack_level": p.get("rack_level"),
                    "on_floor": p.get("on_floor"),
                    "loaded": p.get("loaded"),
                    "type": p.get("type"),
                    "face_width_px": float(xy[:, 0].max() - xy[:, 0].min()),
                    "box_centre_x": (p["box"][0] + p["box"][2]) / 2,
                }
            base.with_suffix(".meta.json").write_text(json.dumps(out))

    rng = np.random.default_rng(args.seed)
    train_frames = list(rng.permutation(sorted(multi["train"])))
    by_frame: dict[str, list[str]] = {}
    for k in entry["train"]:
        by_frame.setdefault(k.split("#")[0], []).append(k)
    for n in [*args.sizes, len(train_frames)]:
        subset = train_frames[:n]
        (args.out / f"multi_train_{n}.json").write_text(
            json.dumps({f: multi["train"][f] for f in subset})
        )
        keys = [k for f in subset for k in by_frame.get(f, [])]
        write(f"entry_train_{n}", keys, "train", meta=False)
        logger.info("train subset", frames=n, keypoint_pallets=len(keys))
    test_multi = {
        f: [
            p
            | {
                k: frames[f]["pallets"][i].get(k)
                for k in ("stack_size", "stack_position", "rack_level")
            }
            for i, p in enumerate(ps)
        ]
        for f, ps in multi["test"].items()
    }
    (args.out / "multi_test.json").write_text(json.dumps(test_multi))
    write("entry_test", sorted(entry["test"]), "test", meta=True)
    logger.info("test", frames=len(multi["test"]), keypoint_pallets=len(entry["test"]))


if __name__ == "__main__":
    main()
