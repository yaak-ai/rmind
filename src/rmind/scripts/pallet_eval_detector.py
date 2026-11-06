"""Score trained detector heads (`pallet_proposer_multi` checkpoints) on a labelled frame set.

    uv run python -m rmind.scripts.pallet_eval_detector --labels .../multi_test.json \
        --heads .../proposer_v3_512_load/fold0.pt ... --images .../images/multi

Same metrics as the k-fold training (centre recall, precision, loaded/empty accuracy), per
threshold, on positive (not `unsure`) pallets. Writes `--out` JSON when given.
"""

import argparse
import json
from pathlib import Path

import torch
from structlog import get_logger

from rmind.components.timm_backbone import TimmBackbone
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER
from rmind.scripts.pallet_proposer_multi import (
    Frames,
    decode,
    is_positive,
    make_head,
    score,
)

logger = get_logger(__name__)
SIZES = ((0, 100), (100, 250), (250, 2000))  # box width in px at 1920: far, mid, near


def recall_by(
    pred: dict[str, list], labels: dict[str, list], groups
) -> dict[str, float]:
    """Centre recall of positive pallets per group; `groups(frame, pallet)` -> group names."""
    n: dict[str, int] = {}
    hit: dict[str, int] = {}
    for name, pallets in labels.items():
        for q in pallets:
            if not is_positive(q):
                continue
            cx, cy = (q["box"][0] + q["box"][2]) / 2, (q["box"][1] + q["box"][3]) / 2
            found = any(
                b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in pred.get(name, [])
            )
            for g in groups(name, q):
                n[g] = n.get(g, 0) + 1
                hit[g] = hit.get(g, 0) + found
    return {f"centre_{g}": hit[g] / n[g] for g in sorted(n)} | {
        f"n_{g}": n[g] for g in sorted(n)
    }


def pallet_groups(scene: dict[str, dict]):
    """Stack / rack / loaded of the pallet and the frame's scene type, if known."""

    def groups(name: str, q: dict) -> list[str]:
        out = []
        if q.get("stack_size") is not None:
            stack = (
                "single"
                if (q["stack_size"] or 1) <= 1
                else "stack_bottom"
                if not q.get("stack_position")
                else "stack_upper"
            )
            out += [f"stack={stack}", "rack=rack" if q.get("rack_level") else "rack=no"]
        if q.get("level"):  # real test set v2: floor / rack1 / rack2+ / stack
            out.append(f"level={q['level']}")
        if q.get("load") in {"loaded", "empty"}:
            out.append(f"loaded={q['load']}")
        out += [f"{k[4:]}={v}" for k, v in q.items() if k.startswith("tag_")]
        for axis, value in scene.get(name, {}).items():
            if axis in {"background", "cone", "light", "layout"}:
                out.append(f"{axis}={value}")
        return out

    return groups


def recall_by_size(pred: dict[str, list], labels: dict[str, list]) -> dict[str, float]:
    """Centre recall of positive pallets per box-width bucket (`SIZES`)."""
    out = {}
    for lo, hi in SIZES:
        n = hit = 0
        for name, pallets in labels.items():
            for q in pallets:
                w = q["box"][2] - q["box"][0]
                if not is_positive(q) or not lo <= w < hi:
                    continue
                cx, cy = (
                    (q["box"][0] + q["box"][2]) / 2,
                    (q["box"][1] + q["box"][3]) / 2,
                )
                n += 1
                hit += any(
                    b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
                    for b in pred.get(name, [])
                )
        out[f"centre_w{lo}-{hi}"] = hit / max(n, 1)
        out[f"n_w{lo}-{hi}"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--heads", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, default=Path("/"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=20,
        help="boxes kept per frame; dense synthetic frames have 50+ pallets",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--scene-meta",
        type=Path,
        default=None,
        help="pallet_synthetic_scenes scene_meta.json: recall per scene type",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    groups = pallet_groups(
        json.loads(args.scene_meta.read_text()) if args.scene_meta else {}
    )
    names = sorted(labels)
    results = {}
    frames, enc = None, None
    for path in args.heads:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        res = tuple(ck["res"])
        if frames is None or frames.res != res:
            frames = Frames(args.images, names, res)
            enc = (
                TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
                .to(args.device)
                .eval()
            )
        head = make_head(ck["state_dict"]).to(args.device).eval()
        preds: dict[float, dict[str, list]] = {t: {} for t in args.thresholds}
        for i in range(0, len(names), 16):
            chunk = names[i : i + 16]
            x, _ = frames.batch(chunk, labels, None)
            with torch.no_grad():
                out = head(enc(x.to(args.device)))
            for t in args.thresholds:
                preds[t] |= dict(
                    zip(chunk, decode(out, t, k=args.max_boxes), strict=True)
                )
        for t in args.thresholds:
            key = f"{path.parent.name}/{path.stem}@{t}"
            results[key] = (
                score(preds[t], labels)
                | recall_by_size(preds[t], labels)
                | recall_by(preds[t], labels, groups)
            )
            logger.info(
                "result", model=key, **{a: round(b, 3) for a, b in results[key].items()}
            )
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
