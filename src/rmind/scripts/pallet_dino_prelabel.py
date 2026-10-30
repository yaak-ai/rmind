"""DINOv3-correspondence pre-labels from a few clicked support frames (Label Studio).

    1. reads the support annotations (Label Studio API, project `--support-project`)
    2. leave-one-out: each support frame predicted from the others -> pixel error
       per keypoint, raw match vs after snapping (an honest estimate of pre-label
       quality before you look at it)
    3. optionally pre-labels `--tasks` (Label Studio task JSON) with all supports
       and writes `--out`, plus an overlay sheet for a quick look

The box per image comes from Grounding DINO (best-scoring pallet box).

    LABEL_STUDIO_TOKEN=... uv run --extra pallet python -m rmind.scripts.pallet_dino_prelabel \
        --root /nasa/team-space/nikita/pallet-detector/legacy/pallet-labels --support-project 4 --tasks .../tasks.json --out .../tasks_dino.json
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from rmind.pallet.correspondence import NAMES, DinoMatcher, snap
from rmind.scripts.pallet_label_export import _point
from rmind.scripts.pallet_pipeline import DETECTOR_ID, pick_target
from rmind.scripts.pallet_poc_zeroshot import HEIGHT, WIDTH, Proposer


def load_support(
    url: str, token: str, project: int
) -> list[tuple[str, dict[str, tuple[float, float]]]]:
    request = urllib.request.Request(  # ruff: ignore[suspicious-url-open-usage]
        f"{url}/api/projects/{project}/export?exportType=JSON",
        headers={"Authorization": f"Token {token}"},
    )
    with urllib.request.urlopen(request) as response:  # ruff: ignore[suspicious-url-open-usage]
        tasks = json.load(response)
    out = []
    for task in tasks:
        annotations = [
            a for a in task.get("annotations", []) if not a.get("was_cancelled")
        ]
        if not annotations:
            continue
        points = {}
        for r in annotations[-1]["result"]:
            if r.get("type") == "keypointlabels":
                v = r["value"]
                points[v["keypointlabels"][0]] = (
                    v["x"] / 100 * WIDTH,
                    v["y"] / 100 * HEIGHT,
                )
        if len(points) >= 4:
            out.append((task["data"]["image"].split("d=")[1], points))
    return out


def main() -> None:
    import cv2
    from PIL import Image

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--support-project", type=int, required=True)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    token = os.environ["LABEL_STUDIO_TOKEN"]

    support = load_support(args.url, token, args.support_project)
    print("support frames", len(support), [len(p) for _, p in support])  # ruff: ignore[print]
    proposer = Proposer(args.device, DETECTOR_ID)

    def image_and_box(name: str) -> tuple[np.ndarray, tuple[float, ...] | None]:
        image = np.array(Image.open(args.root / name).convert("RGB"))
        target = pick_target(proposer.detect_all(image[None])[0], None)
        return image, (tuple(target[0]) if target is not None else None)

    frames = [(*image_and_box(name), points) for name, points in support]
    matcher = DinoMatcher(args.device)

    # leave-one-out
    raw_err, snap_err = [], []
    for i, (image, box, points) in enumerate(frames):
        if box is None:
            continue
        matcher.support = []
        for j, (im_j, box_j, pts_j) in enumerate(frames):
            if j != i and box_j is not None:
                matcher.add_support(im_j, box_j, pts_j)
        m = matcher.match(image, box)
        snapped = snap(image, m.pixels)
        for k, name in enumerate(NAMES):
            if name in points and np.isfinite(m.pixels[k]).all():
                raw_err.append(np.linalg.norm(m.pixels[k] - points[name]))
                snap_err.append(np.linalg.norm(snapped[k] - points[name]))
    if raw_err:
        _r, s = np.array(raw_err), np.array(snap_err)

    if args.tasks is None:
        return
    matcher.support = []
    for image, box, points in frames:
        if box is not None:
            matcher.add_support(image, box, points)
    tasks = json.loads(args.tasks.read_text())
    sheet = []
    for n, task in enumerate(tasks):
        image, box = image_and_box(task["data"]["image"].split("d=")[1])
        result = []
        if box is not None:
            m = matcher.match(image, box)
            pixels = snap(image, m.pixels)
            result = [
                _point(name, float(u), float(v))
                for name, (u, v) in zip(NAMES, pixels, strict=True)
                if np.isfinite(u) and 0 <= u < WIDTH and 0 <= v < HEIGHT
            ]
            if len(sheet) < 16 and result:
                view = image[:, :, ::-1].copy()
                for (u, v), s in zip(pixels, m.score, strict=True):
                    if np.isfinite(u):
                        cv2.circle(
                            view,
                            (int(u), int(v)),
                            8,
                            (0, 255, 0) if s > 0 else (0, 165, 255),
                            -1,
                        )
                x0, y0, x1, y1 = (int(c) for c in box)
                pad = 150
                sheet.append(
                    cv2.resize(
                        view[max(y0 - pad, 0) : y1 + pad, max(x0 - pad, 0) : x1 + pad],
                        (480, 270),
                    )
                )
        task["predictions"] = [
            {"model_version": "dino-correspondence", "result": result}
        ]
        if n % 50 == 0:
            print(n, "/", len(tasks), flush=True)  # ruff: ignore[print]
    args.out.write_text(json.dumps(tasks, indent=1))
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        cv2.imwrite(
            str(args.out.with_suffix(".jpg")),
            np.vstack([np.hstack(sheet[k : k + 4]) for k in range(0, len(sheet), 4)]),
        )


if __name__ == "__main__":
    main()
