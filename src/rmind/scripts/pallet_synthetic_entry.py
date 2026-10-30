"""Synthetic renders (labels v5 JSONL) -> extra keypoint-training labels in the entry format.

Writes `--out` as `{image#i: {keypoint: [u, v]}}` with `.boxes.json` (null: crops come from
the labelled points' bounds) and `.load.json`, so `pallet_train_entry --pseudo-labels` takes
it like the pseudo-labels. Keeps points that are in frame and not occluded (depth check
"visible" or "unknown"), and pallets with at least `--min-points` of them whose entry face is
at least `--min-width` pixels wide. Every frame is used: validation stays on real frames.
"""

import argparse
import json
from pathlib import Path

from structlog import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--min-width", type=float, default=60.0)
    args = parser.parse_args()
    labels, boxes, load = {}, {}, {}
    for path in args.labels:
        for line in path.read_text().splitlines():
            frame = json.loads(line)
            for i, p in enumerate(frame["pallets"]):
                kp = {
                    k: [u, v]
                    for k, (u, v, vis) in p["keypoints"].items()
                    if vis in {"visible", "unknown"}
                }
                us = [u for u, _ in kp.values()]
                if len(kp) < args.min_points or max(us) - min(us) < args.min_width:
                    continue
                key = f"{frame['image']}#s{i}"
                labels[key], boxes[key] = kp, None
                if p["loaded"] is not None:
                    load[key] = "loaded" if p["loaded"] else "empty"
    args.out.write_text(json.dumps(labels))
    args.out.with_suffix(".boxes.json").write_text(json.dumps(boxes))
    args.out.with_suffix(".load.json").write_text(json.dumps(load))
    logger.info(
        "done",
        pallets=len(labels),
        loaded=sum(v == "loaded" for v in load.values()),
        frames=len({k.split("#")[0] for k in labels}),
    )


if __name__ == "__main__":
    main()
