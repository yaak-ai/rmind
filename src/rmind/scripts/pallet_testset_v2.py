"""Real `cam_fork` test set v2 for the pallet box detector and the 12-point keypoint net.

Frozen, hand-checked labels from held-out recordings of one warehouse site (frames from
these recordings must never enter training data). Steps, each a subcommand:

    thumbs     every `--step`-th frame of each test recording at thumbnail size, near-
               duplicates of the previous kept frame dropped (parked truck), on screening
               sheets with the cached detections (`detections_v3`) drawn; for picking
               frames by eye (blocked by a load on the forks / no pallet -> skip)
    export     the picked frames (`--picks` {job: [frame, ...]}) as full-res JPEG
               `<job>_<frame:06d>.jpg`
    prelabel   draft labels to correct by hand: detector boxes per image (`--det-head`)
               or, with `--keypoints`, 12 points per labelled pallet (`--kp-ckpt`, cropped
               from the pallet's labelled box as at eval time)
    view       renders for labelling: `frame` (full frame, boxes + indices + tags),
               `boxes` (one zoomed, 10 px-ticked tile per box), `kp` (one pallet, zoomed,
               all points drawn), `nudge` (`pallet_label_tools.nudge` on a labels file:
               apply relative moves, one ticked tile per point)
    finalize   merge the labelling batches into `boxes.json`, `keypoints.json` (+ `.boxes`,
               `.load`, `.meta`), grid-cell counts, pre-label correction stats, contact sheets
    noise      per-point disagreement of the double-labelled pallets (label-noise floor)

Box labels: `{image: [{"box", "level", "stack_position", "loaded", "trunc", "occl",
"unsure", "face", ...}]}`; `level` is "floor", "rack1", "rack2", ... or "stack"
(`stack_position` 0 = bottom). Keypoints: `{"<image>#<box index>": {name: [u, v]}}`.
"""

import argparse
import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final

import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

D12: Final = Path("/nasa/data/d12/yaak")
WORKSPACE: Final = Path("/nasa/team-space/nikita/pallet-detector")
IMAGES: Final = WORKSPACE / "images" / "test_v2"
LABELS: Final = WORKSPACE / "labels" / "test_v2"
DETECTIONS: Final = Path("/nasa/team-space/nikita/pallet-detector-renate/detections_v3")
DET_HEAD: Final = Path(
    "/nasa/team-space/nikita/pallet-detector-tresor/report_slice01/det_real_syn1000/fold0.pt"
)
KP_CKPT: Final = Path(
    "/nasa/team-space/nikita/pallet-detector-kitkat/report_slice01/kp_real_syn1700_rect/best.pt"
)
# held-out recordings and the first frame of each that belongs to the test set
TEST_JOBS: Final = {
    "2026-10-22--09-50-09": 30000,  # frames before 30000 are training data
    "2026-10-27--14-40-57": 0,
    "2026-10-28--12-31-37": 0,
    "2026-10-28--08-00-19": 0,
    # the yellow-floor hall; replaces 2026-10-22--13-01-42, whose cam_fork view is
    # covered by a load for the whole recording
    "2026-10-22--12-21-39": 0,
}
WIDTH, HEIGHT = 1920, 1080
TW, TH = 320, 180  # screening thumbnails
SHORT: Final = {
    "pallet_bottom_left": "BL",
    "pallet_bottom_right": "BR",
    "pallet_top_left": "TL",
    "pallet_top_right": "TR",
    "pocket_l_bottom_left": "lBL",
    "pocket_l_bottom_right": "lBR",
    "pocket_l_top_left": "lTL",
    "pocket_l_top_right": "lTR",
    "pocket_r_bottom_left": "rBL",
    "pocket_r_bottom_right": "rBR",
    "pocket_r_top_left": "rTL",
    "pocket_r_top_right": "rTR",
}


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def video_of(job: str) -> Path:
    return D12 / job / "cam_fork" / "video.mp4"


def decode_every(video: Path, step: int, size: tuple[int, int]) -> np.ndarray:
    """Every `step`-th frame (0, step, 2 step, ...) resized to `size` (W, H), RGB."""
    raw = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{step})),scale={size[0]}:{size[1]}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        env=_env(),
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size[1], size[0], 3)


def cached_detections(job: str, threshold: float) -> dict[int, list[list[float]]]:
    import polars as pl

    path = DETECTIONS / f"{job}.parquet"
    if not path.exists():
        return {}
    d = pl.read_parquet(path)
    return {
        f: [b for b in bs if b[4] >= threshold]
        for f, bs in zip(d["frame"].to_list(), d["boxes"].to_list(), strict=True)
    }


def thumbs(args: argparse.Namespace) -> None:
    """Screening sheets of de-duplicated thumbnails per test recording."""
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    kept_all: dict[str, list[int]] = {}
    for job, first in TEST_JOBS.items():
        frames = decode_every(video_of(job), args.step, (TW, TH))
        dets = cached_detections(job, args.threshold)
        kept, last = [], None
        for i, im in enumerate(frames):
            n = i * args.step
            if n < first:
                continue
            small = cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            if last is not None and np.abs(small - last).mean() < args.min_change:
                continue
            last = small
            kept.append(n)
        kept_all[job] = kept
        tiles = []
        for n in kept:
            t = cv2.cvtColor(frames[n // args.step], cv2.COLOR_RGB2BGR)
            for b in dets.get(n, []):
                s = TW / WIDTH
                cv2.rectangle(
                    t,
                    (int(b[0] * s), int(b[1] * s)),
                    (int(b[2] * s), int(b[3] * s)),
                    (0, 255, 0),
                    1,
                )
            cv2.putText(t, str(n), (3, 14), 0, 0.45, (0, 0, 0), 3)
            cv2.putText(t, str(n), (3, 14), 0, 0.45, (0, 255, 255), 1)
            tiles.append(t)
        per = args.cols * args.rows
        for k in range(0, len(tiles), per):
            chunk = tiles[k : k + per]
            chunk += [np.zeros_like(tiles[0])] * (per - len(chunk))
            sheet = np.vstack([
                np.hstack(chunk[r * args.cols : (r + 1) * args.cols])
                for r in range(args.rows)
            ])
            cv2.imwrite(str(args.out / f"{job}_{k // per:03d}.jpg"), sheet)
        logger.info("thumbs", job=job, frames=len(frames), kept=len(kept))
    (args.out / "candidates.json").write_text(json.dumps(kept_all))


def select(args: argparse.Namespace) -> None:
    """Pick frames from the screening tags (`<screen>/<job>.json`, one entry per usable
    thumbnail: frame, n, levels, near, far, edge, loaded, empty, face, q).

    Greedy by interest: rare levels (rack / stack) first, then far / near / edge content,
    so the grid is filled rather than sampled uniformly. A frame is taken only if it is
    >= `--min-gap` frames from every picked frame of its recording and its thumbnail
    differs from theirs (mean abs difference >= `--min-diff`: no parked-truck repeats);
    at most `--max-share` of the picks from one recording.
    """
    entries = []
    small: dict[tuple[str, int], np.ndarray] = {}
    for job in TEST_JOBS:
        path = args.screen / f"{job}.json"
        if not path.exists():
            continue
        tags = json.loads(path.read_text())
        frames = decode_every(video_of(job), 30, (64, 36))
        for t in tags:
            if t["frame"] // 30 < len(frames):
                small[job, t["frame"]] = frames[t["frame"] // 30].astype(np.float32)
                levels = set(t.get("levels", []))
                score = (
                    t.get("q", 1)
                    + 2.5 * bool(levels & {"rack1", "rack2plus", "stack"})
                    + 1.0 * ("rack2plus" in levels)
                    + 0.7 * bool(t.get("far"))
                    + 0.5 * bool(t.get("edge"))
                    + 0.5 * bool(t.get("near"))
                    + 0.5 * bool(t.get("face"))
                    + 0.1 * min(t.get("n", 1), 5)
                )
                entries.append((score, job, t["frame"]))
    entries.sort(key=lambda e: (-e[0], e[1], e[2]))
    picked: dict[str, list[int]] = {}
    cap = int(args.max_share * args.n)
    for _score, job, f in entries:
        mine = picked.setdefault(job, [])
        if len(mine) >= cap or any(abs(f - g) < args.min_gap for g in mine):
            continue
        if any(
            np.abs(small[job, f] - small[job, g]).mean() < args.min_diff for g in mine
        ):
            continue
        mine.append(f)
        if sum(map(len, picked.values())) >= args.n:
            break
    picked = {j: sorted(v) for j, v in picked.items()}
    args.out.write_text(json.dumps(picked, indent=1))
    logger.info("select", **{j: len(v) for j, v in picked.items()})


def export(args: argparse.Namespace) -> None:
    from PIL import Image

    from rmind.scripts.pallet_label_export import read_frames

    picks: dict[str, list[int]] = json.loads(args.picks.read_text())
    args.images.mkdir(parents=True, exist_ok=True)
    for job, wanted in picks.items():
        first = TEST_JOBS[job]
        frames = sorted({f for f in wanted if f >= first})
        todo = [f for f in frames if not (args.images / f"{job}_{f:06d}.jpg").exists()]
        if not todo:
            continue
        for k in range(0, len(todo), 20):  # ffmpeg fails on very long select chains
            chunk = todo[k : k + 20]
            for f, im in zip(chunk, read_frames(video_of(job), chunk), strict=True):
                Image.fromarray(im).save(args.images / f"{job}_{f:06d}.jpg", quality=95)
        logger.info("export", job=job, frames=len(todo))


def _detector(device: str) -> tuple:
    import torch

    from rmind.components.timm_backbone import TimmBackbone
    from rmind.scripts.pallet_proposer_head import ENCODER, LAYER
    from rmind.scripts.pallet_proposer_multi import make_head

    ck = torch.load(DET_HEAD, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(device)
        .eval()
    )
    return res, head, enc


def prelabel_boxes(args: argparse.Namespace) -> None:
    """Detector boxes per image as a draft box-label file (tags unset)."""
    import torch

    from rmind.scripts.pallet_proposer_multi import Frames, decode

    names = sorted(p.name for p in args.images.glob("*.jpg"))
    res, head, enc = _detector(args.device)
    frames = Frames(args.images, names, res)
    out = {}
    for i in range(0, len(names), 16):
        chunk = names[i : i + 16]
        x, _ = frames.batch(chunk, {}, None)
        with torch.no_grad():
            dets = decode(head(enc(x.to(args.device))), args.threshold, k=100)
        for n, bs in zip(chunk, dets, strict=True):
            out[n] = [
                {
                    "box": [round(v, 1) for v in b[:4]],
                    "score": round(b[4], 3),
                    "p_loaded": round(b[5], 3),
                }
                for b in bs
            ]
    args.out.write_text(json.dumps(out, indent=1))
    logger.info("prelabel boxes", images=len(out), boxes=sum(map(len, out.values())))


def prelabel_keypoints(args: argparse.Namespace) -> None:
    """12 points per pallet key ("<image>#<i>") from the keypoint net on its box."""
    import torch

    from rmind.pallet.correspondence import NAMES
    from rmind.pallet.entry import EntryKeypoints
    from rmind.pallet.keypoints import PalletKeypointNet, decode

    boxes: dict[str, list[float]] = json.loads(args.keypoints.read_text())
    ck = torch.load(KP_CKPT, map_location="cpu", weights_only=False)
    net = PalletKeypointNet(ck["backbone"], num_keypoints=len(NAMES), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    net = net.to(args.device).eval()
    # EntryKeypoints needs points; a dummy point at the box centre stands in
    labels = {
        k: (
            np.full((len(NAMES), 2), [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]),
            np.ones(len(NAMES), dtype=bool),
        )
        for k, b in boxes.items()
    }
    shape = ck.get("crop_shape")
    ds = EntryKeypoints(
        args.images,
        labels,
        boxes,
        train=False,
        crop_shape=tuple(shape) if shape else None,
        crop_margin=ck.get("crop_margin", 0.3),
    )
    keys = [n for n, *_ in ds.items]
    out, j = {}, 0
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, batch_size=32, num_workers=4):
            pixels, conf = decode(net(b["image"].to(args.device)))
            frame = pixels.cpu() * b["scale_xy"][:, None, :] + b["crop"][:, None, :2]
            for p, c in zip(frame.numpy(), conf.cpu().numpy(), strict=True):
                out[keys[j]] = {
                    n: [round(float(u), 1), round(float(v), 1)]
                    for n, (u, v), s in zip(NAMES, p, c, strict=True)
                    if s >= args.kp_conf
                }
                j += 1
    args.out.write_text(json.dumps(out, indent=1))
    logger.info("prelabel keypoints", pallets=len(out))


def _tag(p: dict) -> str:
    lvl = p.get("level", "?")
    if lvl == "stack":
        lvl = f"stack{p.get('stack_position', '?')}"
    flags = "".join(
        c
        for c, k in (("L", "loaded"), ("T", "trunc"), ("O", "occl"), ("U", "unsure"))
        if p.get(k)
    )
    face = "" if p.get("face", True) else " noface"
    return f"{lvl}{' ' + flags if flags else ''}{face}"


def view_frame(image: Path, pallets: list[dict], out: Path, region=None) -> Path:
    """The frame (or a region of it, zoomed to 1600 px wide) with numbered boxes."""
    import cv2

    im = cv2.imread(str(image))
    x0, y0, x1, y1 = region or (0, 0, WIDTH, HEIGHT)
    z = 1600 / (x1 - x0)
    c = cv2.resize(
        im[y0:y1, x0:x1],
        None,
        fx=z,
        fy=z,
        interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA,
    )
    step = 100 if region is None else 50 if x1 - x0 > 600 else 10
    for x in range((x0 // step + 1) * step, x1, step):
        p = int((x - x0) * z)
        cv2.line(c, (p, 0), (p, 10), (0, 255, 255), 1)
        if x % (step * 2 if step == 10 else step) == 0:
            cv2.putText(c, str(x), (p + 2, 22), 0, 0.4, (0, 255, 255), 1)
    for y in range((y0 // step + 1) * step, y1, step):
        p = int((y - y0) * z)
        cv2.line(c, (0, p), (10, p), (0, 255, 255), 1)
        if y % (step * 2 if step == 10 else step) == 0:
            cv2.putText(c, str(y), (12, p + 4), 0, 0.4, (0, 255, 255), 1)
    for i, p in enumerate(pallets):
        b = p["box"]
        col = (0, 165, 255) if p.get("unsure") else (0, 255, 0)
        q0 = (int((b[0] - x0) * z), int((b[1] - y0) * z))
        q1 = (int((b[2] - x0) * z), int((b[3] - y0) * z))
        cv2.rectangle(c, q0, q1, col, 1)
        txt = f"{i} {_tag(p)}" if "level" in p else f"{i} {p.get('score', '')}"
        q = (q0[0] + 2, max(q0[1] - 4, 12))
        cv2.putText(c, txt, q, 0, 0.5, (0, 0, 0), 3)
        cv2.putText(c, txt, q, 0, 0.5, col, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def view_boxes(image: Path, pallets: list[dict], out: Path, only=None) -> Path:
    """One tile per box: box + 25 px / 40 % margin, zoomed, 10 px ticks along the edges.

    Raises:
        ValueError: no boxes to show.
    """
    import cv2

    im = cv2.imread(str(image))
    tiles = []
    th = 300
    for i, p in enumerate(pallets):
        if only is not None and i not in only:
            continue
        b = p["box"]
        w, h = b[2] - b[0], b[3] - b[1]
        mx, my = max(25, 0.4 * w), max(25, 0.6 * h)
        x0, y0 = int(max(0, b[0] - mx)), int(max(0, b[1] - my))
        x1, y1 = int(min(WIDTH, b[2] + mx)), int(min(HEIGHT, b[3] + my))
        z = min(900 / (x1 - x0), th * 2 / (y1 - y0), 8.0)
        c = cv2.resize(
            im[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC
        )
        col = (0, 165, 255) if p.get("unsure") else (0, 255, 0)
        q0 = (round((b[0] - x0) * z), round((b[1] - y0) * z))
        q1 = (round((b[2] - x0) * z), round((b[3] - y0) * z))
        cv2.rectangle(c, q0, q1, col, 1)
        for x in range((x0 // 10 + 1) * 10, x1, 10):
            q, major = int((x - x0) * z), x % 50 == 0
            cv2.line(c, (q, 0), (q, 10 if major else 5), (0, 255, 255), 1)
            if major and z * 50 > 40:
                cv2.putText(c, str(x), (q + 2, 20), 0, 0.35, (0, 255, 255), 1)
        for y in range((y0 // 10 + 1) * 10, y1, 10):
            q, major = int((y - y0) * z), y % 50 == 0
            cv2.line(c, (0, q), (10 if major else 5, q), (0, 255, 255), 1)
            if major and z * 50 > 25:
                cv2.putText(c, str(y), (12, q + 4), 0, 0.35, (0, 255, 255), 1)
        txt = f"{i} {_tag(p)}" if "level" in p else f"{i}"
        txt += f"  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
        cv2.putText(c, txt, (4, c.shape[0] - 6), 0, 0.5, (0, 0, 0), 3)
        cv2.putText(c, txt, (4, c.shape[0] - 6), 0, 0.5, (255, 255, 255), 1)
        tiles.append(c)
    if not tiles:
        msg = "no boxes"
        raise ValueError(msg)
    # rows of up to 1800 px
    rows, row, wsum = [], [], 0
    for t in tiles:
        if row and wsum + t.shape[1] > 1800:
            rows.append(row)
            row, wsum = [], 0
        row.append(t)
        wsum += t.shape[1]
    rows.append(row)
    width = max(sum(t.shape[1] + 4 for t in r) for r in rows)
    out_rows = []
    for r in rows:
        canvas = np.full((max(t.shape[0] for t in r), width, 3), 80, np.uint8)
        x = 0
        for t in r:
            canvas[: t.shape[0], x : x + t.shape[1]] = t
            x += t.shape[1] + 4
        out_rows.append(canvas)
    sheet = np.vstack([
        np.vstack([r, np.full((4, width, 3), 80, np.uint8)]) for r in out_rows
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def view_kp(
    image: Path, points: dict[str, list[float]], box: list[float] | None, out: Path
) -> Path:
    """One pallet zoomed (crop around its box / points), every point drawn and named."""
    import cv2

    im = cv2.imread(str(image))
    xs = [p[0] for p in points.values()] + ([box[0], box[2]] if box else [])
    ys = [p[1] for p in points.values()] + ([box[1], box[3]] if box else [])
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    mx, my = max(30, 0.15 * w), max(30, 0.4 * h)
    x0, y0 = int(max(0, min(xs) - mx)), int(max(0, min(ys) - my))
    x1, y1 = int(min(WIDTH, max(xs) + mx)), int(min(HEIGHT, max(ys) + my))
    z = min(1500 / (x1 - x0), 800 / (y1 - y0), 10.0)
    c = cv2.resize(im[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    for x in range((x0 // 10 + 1) * 10, x1, 10):
        q, major = int((x - x0) * z), x % 50 == 0
        cv2.line(c, (q, 0), (q, 12 if major else 6), (0, 255, 255), 1)
        if major:
            cv2.putText(c, str(x), (q + 2, 24), 0, 0.4, (0, 255, 255), 1)
    for y in range((y0 // 10 + 1) * 10, y1, 10):
        q, major = int((y - y0) * z), y % 50 == 0
        cv2.line(c, (0, q), (12 if major else 6, q), (0, 255, 255), 1)
        if major:
            cv2.putText(c, str(y), (14, q + 4), 0, 0.4, (0, 255, 255), 1)
    if box:
        cv2.rectangle(
            c,
            (int((box[0] - x0) * z), int((box[1] - y0) * z)),
            (int((box[2] - x0) * z), int((box[3] - y0) * z)),
            (0, 140, 0),
            1,
        )
    for k, (u, v) in points.items():
        q = (round((u - x0) * z), round((v - y0) * z))
        col = (
            (0, 0, 255)
            if k.startswith("pallet")
            else (255, 120, 0)
            if k.startswith("pocket_l")
            else (255, 0, 255)
        )
        cv2.line(c, (q[0] - 8, q[1]), (q[0] + 8, q[1]), col, 1)
        cv2.line(c, (q[0], q[1] - 8), (q[0], q[1] + 8), col, 1)
        dy = -6 if "top" in k else 16
        cv2.putText(c, SHORT[k], (q[0] + 3, q[1] + dy), 0, 0.45, (0, 0, 0), 3)
        cv2.putText(c, SHORT[k], (q[0] + 3, q[1] + dy), 0, 0.45, col, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def edit(args: argparse.Namespace) -> None:
    """Change one image's box labels in a labels file.

    JSON: a list (replaces all), or any of {"edit": {i: [dx0, dy0, dx1, dy1]},
    "box": {i: [x0, y0, x1, y1]}, "set": {i: {field: value}}, "add": [pallet, ...],
    "drop": [i, ...]}, applied in that order (indices before adding / dropping).
    """
    labels = json.loads(args.labels.read_text()) if args.labels.exists() else {}
    cur = [dict(p) for p in labels.get(args.key, [])]
    arg = json.loads(args.json)
    if isinstance(arg, list):
        cur = arg
    else:
        for i, d in arg.get("edit", {}).items():
            cur[int(i)]["box"] = [
                round(a + b, 1) for a, b in zip(cur[int(i)]["box"], d, strict=True)
            ]
        for i, b in arg.get("box", {}).items():
            cur[int(i)]["box"] = [float(v) for v in b]
        for i, fields in arg.get("set", {}).items():
            cur[int(i)] |= fields
        cur += arg.get("add", [])
        drop = set(arg.get("drop", []))
        cur = [p for i, p in enumerate(cur) if i not in drop]
    labels[args.key] = cur
    args.labels.write_text(json.dumps(labels, indent=1))
    for i, p in enumerate(cur):
        print(i, [round(v) for v in p["box"]], _tag(p) if "level" in p else "untagged")  # ruff: ignore[print]


# --- finalize -------------------------------------------------------------------

FACE_W: Final = 1.0  # m; an entry face is 0.8 (EUR short side) or 1.2 m wide
LEVELS: Final = ("floor", "rack1", "rack2+", "stack")


def distance_m(left: tuple[float, float], right: tuple[float, float]) -> float | None:
    """Camera distance from the angle a ~1 m wide face subtends (camera-height free,
    the forks lift the camera; +-20 % from the unknown 0.8 / 1.2 m face width)."""
    from rmind.pallet.geometry import CAM_FORK

    rays = CAM_FORK.rays(np.array([left, right], dtype=np.float64))
    if not np.isfinite(rays).all():
        return None
    angle = float(np.arccos(np.clip(rays[0] @ rays[1], -1.0, 1.0)))
    return FACE_W / (2 * np.tan(angle / 2)) if angle > 0 else None


def phase(d: float | None) -> str:
    return (
        "unknown"
        if d is None
        else "last_metre"
        if d < 2
        else "approaching"
        if d < 4
        else "far"
    )


def level_bucket(p: dict) -> str:
    lvl = p["level"]
    if lvl.startswith("rack"):
        return "rack1" if lvl == "rack1" else "rack2+"
    return lvl


def width_bucket(w: float) -> str:
    return "<100" if w < 100 else "100-250" if w < 250 else ">250"


def normalise(p: dict) -> dict:
    """A labelled pallet in the output schema (also what the detector scoring reads)."""
    lvl = p["level"]
    rack = int(lvl[4:]) if lvl.startswith("rack") else int(p.get("rack_level") or 0)
    stack = lvl == "stack"
    pos = int(p.get("stack_position", 0)) if stack else 0
    size = int(p.get("stack_size") or (pos + 1 if stack else 1))
    return {
        "box": [round(float(v), 1) for v in p["box"]],
        "level": lvl,
        "stack_position": pos if stack else None,
        "stack_size": max(size, 2) if stack else 1,
        "rack_level": rack or None,
        "loaded": bool(p.get("loaded")),
        "face": bool(p.get("face", True)),
        "trunc": bool(p.get("trunc")),
        "occl": bool(p.get("occl")),
        "unsure": bool(p.get("unsure")),
        "load": "loaded" if p.get("loaded") else "empty",
        # the first labelling convention's flags (`pallet_proposer_multi --positives`)
        "floor": lvl == "floor" or (stack and pos == 0 and not rack),
        **({"note": p["note"]} if p.get("note") else {}),
    }


def _merge(pattern: str) -> dict:
    out: dict = {}
    for f in sorted(LABELS.glob(f"work/{pattern}")):
        out |= json.loads(f.read_text())
    return out


def confirmed(pattern: str) -> tuple[dict, dict, dict]:
    """Keypoint batches -> (confirmed points, unsure reasons, unconfirmed keys).

    A pallet counts only if its points are exactly the ones last recorded as checked
    (`view nudge --confirm`, which renders a tile per point): no edits after the check.
    """
    points, unsure, open_ = {}, {}, {}
    for f in sorted(LABELS.glob(f"work/{pattern}")):
        labels = json.loads(f.read_text())
        cf = f.with_suffix(".checks.json")
        checks = json.loads(cf.read_text()) if cf.exists() else {}
        for key, pts in labels.items():
            c = checks.get(key, {})
            if "unsure" in c and not pts:
                unsure[key] = c["unsure"]
            elif pts and c.get("checked") == pts:
                points[key] = pts
            else:
                open_[key] = f.name
    return points, unsure, open_


def _match_moves(pre: list[list[float]], final: list[list[float]]) -> list[float]:
    """Greedy IoU matching of pre-label boxes to final boxes; max edge move per match."""
    from rmind.scripts.pallet_proposer_head import iou

    pairs = sorted(
        ((iou(a, b), i, j) for i, a in enumerate(pre) for j, b in enumerate(final)),
        reverse=True,
    )
    used_i, used_j, moves = set(), set(), []
    for v, i, j in pairs:
        if v < 0.3 or i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        moves.append(max(abs(a - b) for a, b in zip(pre[i], final[j], strict=True)))
    return moves


def _crosstab(rows: list[dict], a: str, b: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out.setdefault(r[a], {}).setdefault(r[b], 0)
        out[r[a]][r[b]] += 1
    return out


def finalize(args: argparse.Namespace) -> None:
    """Merge the labelling batches into the frozen files and write the statistics."""
    boxes = {
        k: [normalise(p) for p in v] for k, v in sorted(_merge("boxes_b*.json").items())
    }
    (LABELS / "boxes.json").write_text(json.dumps(boxes, indent=1))
    points, kp_unsure, kp_open = confirmed("kp_c??.json")
    # a pallet whose box ended up unsure (cross-batch harmonisation) is not scored
    for key in [
        k for k in points if boxes[k.split("#")[0]][int(k.split("#")[1])]["unsure"]
    ]:
        kp_unsure[key] = "box unsure"
        del points[key]
    points = dict(sorted(points.items()))
    if kp_open:
        logger.warning("unconfirmed keypoint pallets left out", n=len(kp_open))
    meta, kboxes, load = {}, {}, {}
    for key, pts in points.items():
        name, _, i = key.partition("#")
        p = boxes[name][int(i)]
        kboxes[key], load[key] = p["box"], p["load"]
        pairs = [
            (f"pallet_{e}_left", f"pallet_{e}_right")
            for e in ("bottom", "top")
            if f"pallet_{e}_left" in pts and f"pallet_{e}_right" in pts
        ]
        left, right = (
            (pts[pairs[0][0]], pts[pairs[0][1]])
            if pairs
            else ((p["box"][0], p["box"][3]), (p["box"][2], p["box"][3]))
        )
        d = distance_m(left, right)
        meta[key] = {
            "distance_m": None if d is None else round(d, 2),
            "phase": phase(d),
            "stack_size": p["stack_size"],
            "stack_position": p["stack_position"] or 0,
            "rack_level": p["rack_level"],
            "level": p["level"],
            "on_floor": p["floor"],
            "loaded": p["loaded"],
            "type": None,
            "face_width_px": round(float(np.hypot(*np.subtract(right, left))), 1),
            "face_from_corners": bool(pairs),
            "box_centre_x": round((p["box"][0] + p["box"][2]) / 2, 1),
            "points": len(pts),
        }
    stem = LABELS / "keypoints"
    (LABELS / "keypoints.json").write_text(json.dumps(points, indent=1))
    for suffix, data in (("boxes", kboxes), ("load", load), ("meta", meta)):
        stem.with_suffix(f".{suffix}.json").write_text(json.dumps(data, indent=1))

    # grid cells
    box_rows = []
    for name, ps in boxes.items():
        for p in ps:
            if p["unsure"]:
                continue
            b = p["box"]
            d = distance_m((b[0], b[3]), (b[2], b[3]))
            box_rows.append({
                "job": name.rsplit("_", 1)[0],
                "level": level_bucket(p),
                "width": width_bucket(b[2] - b[0]),
                "loaded": p["load"],
                "image_x": "edge" if abs((b[0] + b[2]) / 2 - 960) > 600 else "centre",
                "phase": phase(d),
                "face": "face" if p["face"] else "no_face",
            })
    kp_rows = [
        {
            "job": k.split("#")[0].rsplit("_", 1)[0],
            "level": level_bucket(boxes[k.split("#")[0]][int(k.split("#")[1])]),
            "width": width_bucket(m["face_width_px"]),
            "loaded": "loaded" if m["loaded"] else "empty",
            "image_x": "edge" if abs(m["box_centre_x"] - 960) > 600 else "centre",
            "phase": m["phase"],
        }
        for k, m in meta.items()
    ]

    def counts(rows: list[dict]) -> dict:
        axes = [a for a in rows[0] if a != "job"] + ["job"] if rows else []
        out: dict = {"n": len(rows)}
        for a in axes:
            out[a] = {}
            for r in rows:
                out[a][r[a]] = out[a].get(r[a], 0) + 1
        out["level_x_width"] = _crosstab(rows, "level", "width")
        out["level_x_loaded"] = _crosstab(rows, "level", "loaded")
        out["level_x_image_x"] = _crosstab(rows, "level", "image_x")
        return out

    stats: dict = {
        "frames": len(boxes),
        "pallets": sum(map(len, boxes.values())),
        "unsure": sum(p["unsure"] for ps in boxes.values() for p in ps),
        "boxes": counts(box_rows),
        "keypoint_pallets": counts(kp_rows),
        "keypoints": sum(len(v) for v in points.values()),
    }

    # how far the pre-labels were from the final labels
    pre_b = json.loads((LABELS / "work" / "prelabel_boxes.json").read_text())
    moves, n_pre, n_fin = [], 0, 0
    for name, ps in boxes.items():
        pre = [b["box"] for b in pre_b.get(name, [])]
        fin = [p["box"] for p in ps]
        moves += _match_moves(pre, fin)
        n_pre, n_fin = n_pre + len(pre), n_fin + len(fin)
    mv = np.array(moves)
    stats["box_prelabels"] = {
        "prelabel_boxes": n_pre,
        "final_boxes": n_fin,
        "matched": len(mv),
        "dropped_false_positive": n_pre - len(mv),
        "added_missed": n_fin - len(mv),
        "unchanged_le2px": int((mv <= 2).sum()),
        "corrected": int((mv > 2).sum()),
        "median_max_edge_move_px": float(np.median(mv)) if len(mv) else None,
        "median_max_edge_move_px_of_corrected": float(np.median(mv[mv > 2]))
        if (mv > 2).any()
        else None,
    }
    pre_k = _merge("prelabel_kp2_*.json")
    d_all, added, dropped, checks = [], 0, 0, {}
    for key, pts in points.items():
        pre = pre_k.get(key, {})
        checks[key] = {
            n: {
                "checked": True,
                "move_px": round(float(np.hypot(*np.subtract(pts[n], pre[n]))), 2)
                if n in pre
                else None,
            }
            for n in pts
        } | {n: {"dropped": True} for n in pre if n not in pts}
        d_all += [checks[key][n]["move_px"] for n in pts if n in pre]
        added += sum(n not in pre for n in pts)
        dropped += sum(n not in pts for n in pre)
    for key, why in kp_unsure.items():
        checks[key] = {"unsure": why}
    (LABELS / "work" / "kp_checks.json").write_text(json.dumps(checks, indent=1))
    d = np.array(d_all)
    stats["kp_prelabels"] = {
        "pallets_confirmed": len(points),
        "pallets_unsure_not_labelled": len(kp_unsure),
        "pallets_unconfirmed_left_out": len(kp_open),
        "points_final": int(sum(len(v) for v in points.values())),
        "points_from_prelabel": len(d),
        "median_move_px": float(np.median(d)) if len(d) else None,
        "p90_move_px": float(np.quantile(d, 0.9)) if len(d) else None,
        "share_moved_gt1px": float((d > 1).mean()) if len(d) else None,
        "share_moved_gt2px": float((d > 2).mean()) if len(d) else None,
        "share_moved_gt10px": float((d > 10).mean()) if len(d) else None,
        "added": added,
        "dropped_not_visible": dropped,
        "unsure_reasons": dict(sorted(kp_unsure.items())),
    }
    (LABELS / "stats.json").write_text(json.dumps(stats, indent=1))
    logger.info("finalize", frames=len(boxes), kp_pallets=len(points))
    contact_sheets(boxes, points, kboxes, args.seed)


def contact_sheets(boxes: dict, points: dict, kboxes: dict, seed: int) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    names = sorted(boxes)
    pick = sorted(rng.choice(len(names), min(40, len(names)), replace=False))
    tiles = []
    for i in pick:
        im = cv2.imread(str(IMAGES / names[i]))
        for j, p in enumerate(boxes[names[i]]):
            b = [int(v) for v in p["box"]]
            col = (0, 165, 255) if p["unsure"] else (0, 255, 0)
            cv2.rectangle(im, b[:2], b[2:], col, 3)
            txt = f"{j} {_tag(p)}"
            cv2.putText(im, txt, (b[0], max(b[1] - 8, 30)), 0, 1.1, (0, 0, 0), 6)
            cv2.putText(im, txt, (b[0], max(b[1] - 8, 30)), 0, 1.1, col, 2)
        t = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(t, names[i][:-4], (4, 264), 0, 0.45, (0, 0, 0), 3)
        cv2.putText(t, names[i][:-4], (4, 264), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    cv2.imwrite(
        str(LABELS / "contact_boxes.jpg"),
        _sheet(tiles, 5),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )
    keys = sorted(points)
    if not keys:
        return
    pick = sorted(rng.choice(len(keys), min(40, len(keys)), replace=False))
    tiles = []
    for i in pick:
        key = keys[i]
        im = cv2.imread(str(IMAGES / key.split("#")[0]))
        b = kboxes[key]
        w, h = b[2] - b[0], b[3] - b[1]
        x0, y0 = int(max(0, b[0] - 0.15 * w)), int(max(0, b[1] - 0.3 * h))
        x1, y1 = int(min(WIDTH, b[2] + 0.15 * w)), int(min(HEIGHT, b[3] + 0.3 * h))
        z = min(470 / (x1 - x0), 250 / (y1 - y0))
        c = cv2.resize(
            im[y0:y1, x0:x1],
            None,
            fx=z,
            fy=z,
            interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA,
        )
        for k, (u, v) in points[key].items():
            q = (round((u - x0) * z), round((v - y0) * z))
            col = (
                (0, 0, 255)
                if k.startswith("pallet")
                else (255, 120, 0)
                if k.startswith("pocket_l")
                else (255, 0, 255)
            )
            cv2.circle(c, q, 3, col, -1)
        t = np.zeros((270, 480, 3), np.uint8)
        t[: c.shape[0], : c.shape[1]] = c
        cv2.putText(
            t,
            key.removesuffix(".jpg")[5:].replace(".jpg", ""),
            (4, 264),
            0,
            0.42,
            (255, 255, 255),
            1,
        )
        tiles.append(t)
    cv2.imwrite(
        str(LABELS / "contact_keypoints.jpg"),
        _sheet(tiles, 5),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )


def _sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([
        np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)
    ])


def noise() -> None:
    """Disagreement between the first labels and an independent second labelling."""
    first, _, _ = confirmed("kp_c??.json")
    second, _, _ = confirmed("double/kp_e??.json")
    per_point: dict[str, list[float]] = {}
    only_one = 0
    for key, pts in second.items():
        a = first.get(key, {})
        for n in set(a) | set(pts):
            if n in a and n in pts:
                per_point.setdefault(n, []).append(
                    float(np.hypot(*np.subtract(a[n], pts[n])))
                )
            else:
                only_one += 1
    allv = (
        np.concatenate([np.array(v) for v in per_point.values()])
        if per_point
        else np.array([])
    )

    def s(v) -> dict:
        v = np.asarray(v)
        return {
            "n": len(v),
            "median_px": round(float(np.median(v)), 2),
            "p90_px": round(float(np.quantile(v, 0.9)), 2),
        }

    out = {
        "pallets": len(second),
        "all": s(allv) if len(allv) else {},
        "visibility_disagreements": only_one,
        "pairs_both": sum(1 for k in second if k in first),
        "per_point": {n: s(v) for n, v in sorted(per_point.items())},
    }
    (LABELS / "double_label.json").write_text(
        json.dumps({"second_labels": second, "stats": out}, indent=1)
    )
    logger.info(
        "noise", **out["all"], pallets=len(second), visibility_disagreements=only_one
    )


def fine_tiles(image: Path, pts: dict[str, list[float]], out: Path) -> Path:
    """Per point a +-16 px window at 10x, nearest-neighbour (pixel edges stay sharp),
    1-px ticks; the red crosshair is the point (pixel i covers [i, i+1))."""
    import cv2

    r, z = 16, 10
    im = cv2.imread(str(image))
    tiles = []
    for k, (u, v) in pts.items():
        m = np.array([[z, 0, -(u - r) * z], [0, z, -(v - r) * z]], dtype=np.float32)
        t = cv2.warpAffine(
            im,
            m,
            (2 * r * z, 2 * r * z),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        c = r * z
        for d in range(-r, r + 1):
            q = round(c + (np.floor(u) + d - u) * z)
            col = (0, 255, 255) if (np.floor(u) + d) % 5 == 0 else (0, 110, 110)
            cv2.line(t, (q, 0), (q, 10 if col[1] == 255 else 5), col, 1)
            q = round(c + (np.floor(v) + d - v) * z)
            col = (0, 255, 255) if (np.floor(v) + d) % 5 == 0 else (0, 110, 110)
            cv2.line(t, (0, q), (10 if col[1] == 255 else 5, q), col, 1)
        cv2.line(t, (c - 20, c), (c + 20, c), (0, 0, 255), 1)
        cv2.line(t, (c, c - 20), (c, c + 20), (0, 0, 255), 1)
        txt = f"{SHORT[k]} {u:.1f},{v:.1f}"
        cv2.putText(t, txt, (14, 2 * c - 10), 0, 0.6, (0, 0, 0), 3)
        cv2.putText(t, txt, (14, 2 * c - 10), 0, 0.6, (255, 255, 255), 1)
        tiles.append(t)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), _sheet(tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def view(args: argparse.Namespace) -> None:
    image = args.images / args.key.split("#")[0]
    out_dir = args.renders
    stem = args.key.replace(".jpg", "").replace("#", "_p")
    if args.what in {"frame", "boxes"}:
        pallets = json.loads(args.labels.read_text()).get(args.key, [])
        if args.what == "frame":
            region = tuple(args.region) if args.region else None
            suffix = "" if region is None else "_" + "-".join(map(str, region))
            out = out_dir / f"frame_{stem}{suffix}.jpg"
            print(view_frame(image, pallets, out, region))  # ruff: ignore[print]
        else:
            only = set(args.only) if args.only else None
            print(view_boxes(image, pallets, out_dir / f"boxes_{stem}.jpg", only))  # ruff: ignore[print]
    elif args.what == "kp":
        pts = json.loads(args.labels.read_text()).get(args.key, {})
        box = None
        if args.boxes is not None:
            name, _, i = args.key.partition("#")
            box = json.loads(args.boxes.read_text())[name][int(i)]["box"]
        print(view_kp(image, pts, box, out_dir / f"kp_{stem}.jpg"))  # ruff: ignore[print]
    else:  # nudge
        from rmind.scripts.pallet_label_tools import nudge

        checks_path = args.labels.with_suffix(".checks.json")
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        if args.unsure:
            # the face is not sharp / visible enough: no keypoints for this pallet
            labels = json.loads(args.labels.read_text())
            labels[args.key] = {}
            args.labels.write_text(json.dumps(labels, indent=1))
            checks[args.key] = {"unsure": args.unsure}
            checks_path.write_text(json.dumps(checks, indent=1))
            print(args.key, "-> unsure:", args.unsure)  # ruff: ignore[print]
            return
        ns = argparse.Namespace(
            labels=args.labels,
            images=args.images,
            image=args.key,
            offsets=args.offsets,
            workspace=out_dir,
        )
        print(nudge(ns))  # ruff: ignore[print]
        pts = json.loads(args.labels.read_text()).get(args.key, {})
        if pts:
            print(fine_tiles(image, pts, out_dir / "renders" / f"fine_{stem}.jpg"))  # ruff: ignore[print]
        if args.confirm:
            # every point as it is now was seen on the tiles just rendered
            pts = json.loads(args.labels.read_text())[args.key]
            checks[args.key] = {"checked": pts}
            checks_path.write_text(json.dumps(checks, indent=1))
            print("confirmed", len(pts), "points")  # ruff: ignore[print]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("thumbs")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--step", type=int, default=30)
    t.add_argument("--min-change", type=float, default=5.0)
    t.add_argument("--threshold", type=float, default=0.3)
    t.add_argument("--cols", type=int, default=6)
    t.add_argument("--rows", type=int, default=6)
    s = sub.add_parser("select")
    s.add_argument("--screen", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=260)
    s.add_argument("--min-gap", type=int, default=60)
    s.add_argument("--min-diff", type=float, default=6.0)
    s.add_argument("--max-share", type=float, default=0.4)
    e = sub.add_parser("export")
    e.add_argument("--picks", type=Path, required=True)
    e.add_argument("--images", type=Path, default=IMAGES)
    p = sub.add_parser("prelabel")
    p.add_argument("--images", type=Path, default=IMAGES)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--keypoints",
        type=Path,
        default=None,
        help="{pallet key: box} -> keypoint pre-labels instead of boxes",
    )
    p.add_argument("--kp-conf", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    v = sub.add_parser("view")
    v.add_argument("what", choices=["frame", "boxes", "kp", "nudge"])
    v.add_argument("key", help="image name, or '<image>#<box index>' for kp / nudge")
    v.add_argument("offsets", nargs="?", help="nudge: {kp: [dx, dy] | 'drop'}")
    v.add_argument("--labels", type=Path, required=True)
    v.add_argument("--boxes", type=Path, default=None, help="kp: draw the pallet box")
    v.add_argument("--region", type=int, nargs=4, default=None)
    v.add_argument("--only", type=int, nargs="+", default=None)
    v.add_argument("--images", type=Path, default=IMAGES)
    v.add_argument("--renders", type=Path, required=True)
    v.add_argument(
        "--confirm",
        action="store_true",
        help="nudge: record every point as checked on the tiles of this render",
    )
    v.add_argument(
        "--unsure",
        default=None,
        help="nudge: no keypoints for this pallet (reason), e.g. 'blurred'",
    )
    d = sub.add_parser("edit")
    d.add_argument("key", help="image name")
    d.add_argument("json")
    d.add_argument("--labels", type=Path, required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--seed", type=int, default=0)
    sub.add_parser("noise")
    args = parser.parse_args()
    if args.cmd == "select":
        select(args)
    elif args.cmd == "finalize":
        finalize(args)
    elif args.cmd == "noise":
        noise()
    elif args.cmd == "edit":
        edit(args)
    elif args.cmd == "thumbs":
        thumbs(args)
    elif args.cmd == "export":
        export(args)
    elif args.cmd == "prelabel":
        (prelabel_keypoints if args.keypoints else prelabel_boxes)(args)
    elif args.cmd == "view":
        view(args)


if __name__ == "__main__":
    main()
