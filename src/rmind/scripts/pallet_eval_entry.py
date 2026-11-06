"""Score trained keypoint nets (`pallet_train_entry` checkpoints) on a labelled pallet set.

    uv run python -m rmind.scripts.pallet_eval_entry --labels .../entry_test.json \
        --boxes .../entry_test.boxes.json --checkpoints .../best.pt ...

Crops come from `--boxes` (the label set's `.boxes.json` by default; a null box falls back to
the labelled points' bounds), shaped as each checkpoint was trained (square or `crop_shape`).
Same metrics as training: raw-pixel error on labelled points (median, p90, PCK at 5 / 10 px).
With `--meta` (`pallet_synthetic_slice` `.meta.json`), also per slice: distance, stack
(single / bottom / upper), rack, loaded, type, horizontal image position (fisheye edges).
Writes `--out` JSON when given.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from structlog import get_logger
from torch.utils.data import DataLoader

from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import EntryKeypoints, load_labels
from rmind.pallet.keypoints import PalletKeypointNet, decode

logger = get_logger(__name__)


def summary(errors: np.ndarray) -> dict[str, float]:
    if not len(errors):
        return {"n": 0}
    return {
        "median_px": float(np.median(errors)),
        "p90_px": float(np.quantile(errors, 0.9)),
        "pck5": float((errors < 5).mean()),
        "pck10": float((errors < 10).mean()),
        "n": len(errors),
    }


def slice_of(m: dict) -> dict[str, str]:
    d, x = m["distance_m"], m["box_centre_x"]
    stack = (
        "single"
        if m["stack_size"] <= 1
        else "stack_bottom"
        if m["stack_position"] == 0
        else "stack_upper"
    )
    out = {
        "distance": "<2m"
        if d is not None and d < 2
        else "2-4m"
        if d is not None and d < 4
        else ">4m",
        "stack": stack,
        "rack": "rack" if m["rack_level"] else "floor_or_stack",
        "loaded": "loaded" if m["loaded"] else "empty",
        "type": str(m["type"]),
        "image_x": "edge"
        if abs(x - 960) > 600
        else "middle"
        if abs(x - 960) > 250
        else "centre",
    }
    if "level" in m:  # real test set v2: floor / rack1 / rack2+ / stack
        out["level"] = str(m["level"])
    # scene types (`pallet_synthetic_scenes`), alone and crossed with stacking
    for axis in (
        "background",
        "cone",
        "light",
        "camera_height",
        "camera_pitch",
        "layout",
    ):
        if axis in m:
            out[axis] = str(m[axis])
    for axis in ("background", "light", "cone"):
        if axis in m:
            out[f"{axis}x{'stack'}"] = f"{m[axis]}/{stack}"
    # extra per-pallet tags (e.g. occluded / light / view angle on real test sets)
    out.update({k[4:]: str(v) for k, v in m.items() if k.startswith("tag_")})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, default=Path("/"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--min-face-width",
        type=float,
        default=0.0,
        help="score only pallets whose labelled points span at least this many px (near)",
    )
    parser.add_argument(
        "--keys-prefix",
        nargs="*",
        default=[],
        help="score only keys with these prefixes",
    )
    args = parser.parse_args()
    labels = {
        n: (xy, ok)
        for n, (xy, ok) in load_labels(args.labels).items()
        if (not args.keys_prefix or n.startswith(tuple(args.keys_prefix)))
        and ok.sum() >= 2
        and np.ptp(xy[ok][:, 0]) >= args.min_face_width
    }
    logger.info("pallets", n=len(labels))
    boxes = json.loads(
        (args.boxes or args.labels.with_suffix(".boxes.json")).read_text()
    )
    meta = json.loads(args.meta.read_text()) if args.meta else {}
    results = {}
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PalletKeypointNet(
            ck["backbone"], num_keypoints=len(NAMES), pretrained=False
        )
        net.load_state_dict(ck["state_dict"])
        net = net.to(args.device).eval()
        shape = ck.get("crop_shape")
        ds = EntryKeypoints(
            args.images,
            labels,
            boxes,
            train=False,
            crop_shape=tuple(shape) if shape else None,
            crop_margin=ck.get("crop_margin", 0.3),
        )
        names = [n for n, *_ in ds.items]
        per_pallet: list[np.ndarray] = []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=32, num_workers=6):
                pixels, _ = decode(net(b["image"].to(args.device)))
                err = (
                    ((pixels.cpu() - b["keypoints"]) * b["scale_xy"][:, None, :])
                    .norm(dim=-1)
                    .numpy()
                )
                per_pallet.extend(
                    e[v] for e, v in zip(err, b["visible"].numpy(), strict=True)
                )
        key = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
        res = {"all": summary(np.concatenate(per_pallet))}
        if meta:
            groups: dict[str, list[np.ndarray]] = {}
            for name, e in zip(names, per_pallet, strict=True):
                if name in meta:
                    for axis, value in slice_of(meta[name]).items():
                        groups.setdefault(f"{axis}={value}", []).append(e)
            res |= {g: summary(np.concatenate(v)) for g, v in sorted(groups.items())}
        results[key] = res
        logger.info(
            "result",
            model=key,
            **{
                a: round(b, 3) if isinstance(b, float) else b
                for a, b in res["all"].items()
            },
        )
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
