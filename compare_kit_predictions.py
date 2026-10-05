#!/usr/bin/env python
"""Does the kit's TensorRT engine predict what this checkpoint predicts, on the same inputs?

Replays a palleter recording through the rmind checkpoint and diffs every output head
against the kit's own prediction for the same window.

The kit records, per inferred window (`policy/inputs`, `sent_to_model`), the four aux
tensors exactly as fed to the engine and a reference to each of the 6 cam_fork frames;
`policy/prediction` holds the engine's heads for that window (joined on `step_tov_ns`).
The aux tensors are used as-is. The frames are re-decoded from the session's mp4, so
they went through H.265 the kit never saw - exact equality is not possible, only close.

Two ways to turn a decoded frame into the 144x256 model image (`--image`):

    kit    the kit's own GPU chain, reimplemented: UYVY bt709 -> RGB at 1920x1080,
           bilinear to 576x324, bilinear to 256x144 (cu-drivr nvmm_preprocess.cu)
    train  how the training frames were made: `ffmpeg_extract_frames.sh` (scale_cuda to
           256x144, JPEG q16), decoded with simplejpeg as the dataset does

Frame references are joined to the mcap's `cam_fork/frame` records on `pts_ns`; record
k is encoded frame k + offset. `--offset auto` picks the offset whose predictions agree
best with the kit's on a subset of windows.

    . env; python compare_kit_predictions.py /path/to/2026-09-29--14-36-16 [--image kit]
"""
# ruff: noqa: T201, PLC0415, ANN001, ANN201, C901, PLR0912, PLR0914, PLR0915, S603, S607

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import make_reader

CAMERA = "cam_fork"
H, W = 144, 256
NATIVE_W, NATIVE_H = 1920, 1080
CAPTURE_W, CAPTURE_H = 576, 324  # cu-drivr nvmm.rs CAPTURE_WIDTH/HEIGHT
AUX = ("speed", "fork_above_300", "relative_ego_pos", "relative_dropoff_pos")
HEADS = ("traction", "steering", "fork1")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SCRIPTS = Path(__file__).parent / "src/rmind/scripts"
# ffmpeg is the system one: the venv's LD_LIBRARY_PATH (nix libstdc++) breaks it
FFMPEG_ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def numbered(paths) -> list[Path]:
    return sorted(paths, key=lambda p: int(re.findall(r"--(\d+)", p.name)[0]))


def read_kit(session: Path):
    """Windows the kit sent to the engine, its predictions, and each frame record's log time."""
    classes: dict[str, type] = {}

    def decode(schema, data):
        if schema.name not in classes:
            pool = descriptor_pool.DescriptorPool()
            for fd in descriptor_pb2.FileDescriptorSet.FromString(schema.data).file:
                pool.Add(fd)
            desc = pool.FindMessageTypeByName(schema.name)
            classes[schema.name] = message_factory.GetMessageClass(desc)
        return classes[schema.name].FromString(data)

    record_of_pts: dict[int, int] = {}
    record_log_time: list[int] = []
    windows: dict[int, dict] = {}
    heads: dict[int, dict[str, float]] = {}
    topics = [f"{CAMERA}/frame", "policy/inputs", "policy/prediction"]
    for path in numbered(session.glob("sensor--*.mcap")):
        with path.open("rb") as f:
            for schema, channel, msg in make_reader(f).iter_messages(topics=topics):
                m = decode(schema, msg.data)
                if channel.topic.endswith("/frame"):
                    record_of_pts[m.pts_ns] = len(record_log_time)
                    record_log_time.append(msg.log_time)
                elif channel.topic == "policy/inputs":
                    if m.sent_to_model:
                        frames = sorted(m.frames, key=lambda r: r.slot)
                        windows[m.step_tov_ns] = {
                            "pts": [r.pts_ns for r in frames],
                            "aux": {
                                t.name: np.asarray(t.values, np.float32) for t in m.aux
                            },
                        }
                else:
                    heads[m.step_tov_ns] = {v.name: v.value for v in m.heads}
    for w in windows.values():
        w["records"] = [record_of_pts[p] for p in w["pts"]]
    return windows, heads, np.asarray(record_log_time)


def kit_resize(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """cu-drivr's `sample_rgb_bilinear` at `resize_coordinate`, rounded to u8 like the kernel."""
    in_h, in_w = rgb.shape[:2]
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * in_w / out_w - 0.5
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * in_h / out_h - 0.5
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    dx, dy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    cx = lambda x: np.clip(x, 0, in_w - 1)  # ruff: ignore[lambda-assignment]
    cy = lambda y: np.clip(y, 0, in_h - 1)  # ruff: ignore[lambda-assignment]
    src = rgb.astype(np.float32)
    p00 = src[cy(y0)][:, cx(x0)]
    p10 = src[cy(y0)][:, cx(x0 + 1)]
    p01 = src[cy(y0 + 1)][:, cx(x0)]
    p11 = src[cy(y0 + 1)][:, cx(x0 + 1)]
    top = p00 + dx * (p10 - p00)
    bottom = p01 + dx * (p11 - p01)
    return np.clip(np.rint(top + dy * (bottom - top)), 0, 255).astype(np.uint8)


def kit_rgb(uyvy: np.ndarray) -> np.ndarray:
    """cu-drivr's `uyvy_rgb_at` (bt709 limited range) for every pixel of one frame."""
    pairs = uyvy.reshape(NATIVE_H, NATIVE_W // 2, 4).astype(np.float32)
    u = np.repeat(pairs[..., 0], 2, axis=1) - 128.0
    v = np.repeat(pairs[..., 2], 2, axis=1) - 128.0
    y = (
        np.stack([pairs[..., 1], pairs[..., 3]], axis=-1).reshape(NATIVE_H, NATIVE_W)
        - 16.0
    )
    c = 1.16438356 * y
    rgb = np.stack(
        [c + 1.79274107 * v, c - 0.21324861 * u - 0.53290933 * v, c + 2.11240179 * u],
        axis=-1,
    )
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def encoded_frames(video: Path) -> int:
    probe = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        str(video),
    ]
    return int(
        subprocess.run(
            probe, capture_output=True, text=True, check=True, env=FFMPEG_ENV
        ).stdout
    )


def frames_kit(videos: list[Path]) -> np.ndarray:
    out = []
    frame_bytes = NATIVE_W * NATIVE_H * 2
    for video in videos:
        before = len(out)
        # passthrough: one output frame per encoded frame (the default would pad to CFR)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "uyvy422",
            "-",
        ]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, env=FFMPEG_ENV) as proc:
            while chunk := proc.stdout.read(frame_bytes):
                raw = np.frombuffer(chunk, np.uint8)
                capture = kit_resize(kit_rgb(raw), CAPTURE_W, CAPTURE_H)
                out.append(kit_resize(capture, W, H))
        if proc.returncode or len(out) - before != encoded_frames(video):
            msg = f"{video}: decoded {len(out) - before} of {encoded_frames(video)} frames"
            raise RuntimeError(msg)
        print(
            f"  decoded {video.name} ({len(out)} frames so far)", end="\r", flush=True
        )
    print()
    return np.stack(out)


def frames_train(videos: list[Path]) -> np.ndarray:
    import simplejpeg

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for video in videos:
            d = Path(tmp) / video.stem
            res = subprocess.run(
                ["bash", str(SCRIPTS / "ffmpeg_extract_frames.sh"), str(video), str(d)],
                capture_output=True,
                text=True,
                check=True,
                env=FFMPEG_ENV,
            )
            if res.stderr.strip():
                print(f"  {video.name}: {res.stderr.strip()}")
            for jpg in sorted(d.glob("*.jpg")):
                out.append(
                    simplejpeg.decode_jpeg(
                        jpg.read_bytes(),
                        colorspace="rgb",
                        fastdct=True,
                        fastupsample=True,
                    )
                )
    return np.stack(out)


def load_frames(session: Path, mode: str) -> np.ndarray:
    cache = session / f".{CAMERA}_{mode}_{W}x{H}_v2.npy"
    if cache.exists():
        return np.load(cache)
    videos = numbered(session.glob(f"{CAMERA}--*.mp4"))
    print(f"decoding {len(videos)} videos ({mode})")
    frames = frames_kit(videos) if mode == "kit" else frames_train(videos)
    np.save(cache, frames)
    return frames


def batch(windows, steps, frames, offset, device):
    imgs = np.stack([frames[np.asarray(windows[s]["records"]) + offset] for s in steps])
    imgs = ((imgs.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 1, 4, 2, 3)
    data = {CAMERA: torch.from_numpy(np.ascontiguousarray(imgs))}
    for name in AUX:
        data[name] = torch.from_numpy(
            np.stack([windows[s]["aux"][name] for s in steps]).reshape(
                len(steps), 6, -1
            )
        )
    return {"data": {k: v.to(device) for k, v in data.items()}}


def predict(model, windows, steps, frames, offset, device, bs=32):
    out = {h: [] for h in HEADS}
    with torch.inference_mode():
        for i in range(0, len(steps), bs):
            o = model(batch(windows, steps[i : i + bs], frames, offset, device))
            for h in HEADS:
                out[h].append(
                    o["policy", "continuous", h].flatten().float().cpu().numpy()
                )
    return {h: np.concatenate(v) for h, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument(
        "--ckpt", default="lightning_logs/tfpakwx5/checkpoints/epoch=19-step=17580.ckpt"
    )
    ap.add_argument("--image", choices=["kit", "train"], default="kit")
    ap.add_argument("--offset", default="auto")
    ap.add_argument(
        "--search", type=int, default=30, help="auto: try offsets in [-N, N]"
    )
    args = ap.parse_args()

    from rmind.models.patch_policy_continuous import PatchPolicyContinuous

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchPolicyContinuous.load_for_export(checkpoint_path=args.ckpt).to(device)

    windows, kit_heads, _ = read_kit(args.session)
    steps = sorted(s for s in windows if s in kit_heads)
    frames = load_frames(args.session, args.image)
    kit = {
        h: np.array([kit_heads[s][f"policy.continuous.{h}"] for s in steps])
        for h in HEADS
    }
    records = np.concatenate([windows[s]["records"] for s in steps])
    print(
        f"{len(steps)} windows with a kit prediction · {len(frames)} encoded frames · "
        f"frame records {records.min()}..{records.max()}"
    )

    def error(pred, idx):
        return float(np.mean([np.abs(pred[h] - kit[h][idx]).mean() for h in HEADS]))

    if args.offset == "auto":
        idx = np.linspace(0, len(steps) - 1, min(40, len(steps))).astype(int)
        sub = [steps[i] for i in idx]
        scores = {}
        for off in range(-args.search, args.search + 1):
            lo, hi = records.min() + off, records.max() + off
            if lo < 0 or hi >= len(frames):
                continue
            scores[off] = error(predict(model, windows, sub, frames, off, device), idx)
        ranked = sorted(scores, key=scores.get)
        print(
            "offset search (mean |rmind - kit| over the heads):",
            ", ".join(f"{o:+d}: {scores[o]:.4f}" for o in ranked[:5]),
            f"... worst {ranked[-1]:+d}: {scores[ranked[-1]]:.4f}",
        )
        offset = ranked[0]
    else:
        offset = int(args.offset)

    pred = predict(model, windows, steps, frames, offset, device)
    print(
        f"\ncheckpoint {args.ckpt}\nimage path '{args.image}', frame offset {offset:+d}\n"
    )
    print(
        f"  {'head':<10}{'kit sd':>9}{'mean|err|':>11}{'p95|err|':>10}{'max|err|':>10}{'corr':>8}"
    )
    for h in HEADS:
        err = np.abs(pred[h] - kit[h])
        c = np.corrcoef(pred[h], kit[h])[0, 1] if kit[h].std() > 1e-8 else float("nan")
        print(
            f"  {h:<10}{kit[h].std():>9.4f}{err.mean():>11.5f}"
            f"{np.percentile(err, 95):>10.5f}{err.max():>10.5f}{c:>8.4f}"
        )


if __name__ == "__main__":
    main()
