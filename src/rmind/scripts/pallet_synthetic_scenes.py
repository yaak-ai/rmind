"""Scene-type metadata for a Veigel slice's test frames and pallets, for crossing errors with scenes.

Per frame (`<out>/scene_meta.json`, keyed by image path): floor/wall material of the layout
(`render-plan.json` background_variant), view kind (curved / straight approach or free view),
cone, light intensity tercile, camera height and pitch relative to nominal. The same fields
are merged into `entry_test.meta.json` (per pallet) so `pallet_eval_entry --meta` slices by them.
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
    parser.add_argument(
        "--out", type=Path, required=True, help="pallet_synthetic_slice output"
    )
    args = parser.parse_args()
    frames = {}
    for line in (args.root / "labels_v5.jsonl").read_text().splitlines():
        d = json.loads(line)
        frames[d["image"]] = d
    plans = {}
    for scene in json.loads((args.root / "scenes.json").read_text()):
        plans[scene["scene_id"]] = json.loads(
            (args.root / scene["path"] / "render-plan.json").read_text()
        ) | {"path": scene["path"]}
    test = json.loads((args.out / "multi_test.json").read_text())
    rows = {}
    for image in test:
        f = frames[image]
        plan = plans[f["job"]]
        view = plan["views"][f["frame"]]
        rows[image] = {
            "background": plan["background_variant"],
            "view_kind": f.get("path_kind") or "free",
            "cone": bool(view["cone"]),
            "light": view["light_intensity"],
            "camera_height": "below" if view["camera_height_delta_m"] < 0 else "above",
            "camera_pitch": "down" if view["camera_pitch_deg"] < 0 else "up",
            "layout": f["job"].rsplit("-", 1)[-1],
        }
    edges = np.quantile([r["light"] for r in rows.values()], [1 / 3, 2 / 3])
    for r in rows.values():
        r["light"] = (
            "dim"
            if r["light"] < edges[0]
            else "medium"
            if r["light"] < edges[1]
            else "bright"
        )
    (args.out / "scene_meta.json").write_text(json.dumps(rows))
    meta_path = args.out / "entry_test.meta.json"
    meta = json.loads(meta_path.read_text())
    for key, m in meta.items():
        m.update(rows[key.split("#")[0]])
    meta_path.write_text(json.dumps(meta))
    counts = {}
    for r in rows.values():
        for k in ("background", "view_kind", "cone", "light"):
            counts.setdefault(k, {}).setdefault(str(r[k]), 0)
            counts[k][str(r[k])] += 1
    logger.info(
        "scene meta", frames=len(rows), **{k: json.dumps(v) for k, v in counts.items()}
    )


if __name__ == "__main__":
    main()
