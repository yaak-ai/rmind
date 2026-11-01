"""Convert a slice of NVIDIA PhysicalAI-Autonomous-Vehicles (PAI) into a per-clip
layout the patch_policy dataset (`config/_templates/dataset/yaak/pai_*.yaml`)
reads directly:

    <out>/clips/<clip_id>/frames/cam_front_left/576x324/{frame_index:09d}.jpg
    <out>/clips/<clip_id>/meta.parquet

Input is the raw per-chunk download (`physical_ai_av`'s `local_dir` layout:
`camera/camera_front_wide_120fov/*.chunk_NNNN.zip`, `labels/egomotion/...`,
`calibration/{camera_intrinsics,sensor_extrinsics}/...`):

    python -m rmind.scripts.pai_convert \
        --raw /nasa/drives/pai_sample_2k/raw \
        --clips /nasa/drives/pai_sample_2k/clips.parquet \
        --mask /nasa/drives/pai_sample_2k/our_valid_mask_576x324.png \
        --out /nasa/drives/pai_sample_2k

CAMERA. `camera_front_wide_120fov` (f-theta, 1920x1080) is re-rendered as a
LEVEL pinhole in the rig frame (optical axis = rig +x, no pitch/roll) with
our `cam_front_left` geometry: fx = fy = 389.4 at 576x324 (`yaw_aug_fx`), and
principal point (288, 180) -- the horizon row read off our frames, which have
no stored calibration. The black fisheye arcs of our defished frames are
copied on (`--mask`). Mount height (PAI ~1.6 m vs our windshield camera) and
the hood are NOT matched.

KINEMATICS. Same conventions as our drives, so `TrajectoryTarget` and the
waypoint goal encoder see identical semantics:
* speed: |v| from egomotion, km/h;
* `headings_denoised/heading`: compass bearing in degrees with the egomotion
  local frame's +x as east and +y as north (`rmind.components.dead_reckoning`);
* `waypoints/xy_normalized`: the pre-#261 `WaypointBuilder` route -- points
  0, 10, ..., 90 m ahead along the DRIVEN path, translated to the ego and
  rotated by the heading (`ST_Rotate(_, radians(heading))`: +y forward,
  +x right), divided by 100. Past the clip's last egomotion pose the path is
  extended straight along the final direction; `pai/waypoints_extrapolated_m`
  records how far, so the dataset can cap it.
* gas/brake/steering are 0.0 and turn_signal 0: PAI has no CAN. The finetune
  arm must zero `action_weight` and `turn_signal_weight`.
"""

import argparse
import io
import json
import multiprocessing
import subprocess  # noqa: S404
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import simplejpeg
from PIL import Image
from structlog import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

CAMERA = "camera_front_wide_120fov"
OUT_W, OUT_H = 576, 324
FX = 389.4
CX, CY = 288.0, 180.0
# ffmpeg pre-scale (area filter) before the bilinear remap, so the ~2.4x
# f-theta -> pinhole minification doesn't alias
DECODE_SCALE = 0.5
NUM_WAYPOINTS = 10
WAYPOINT_SPACING_M = 10.0
# arbitrary epoch: PAI timestamps are clip-relative microseconds
EPOCH_US = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)


def _quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def remap_grid(intrinsics: dict, extrinsics: dict) -> tuple[np.ndarray, np.ndarray]:
    """Source (x, y) pixel coordinates in the DECODE_SCALE'd f-theta frame for
    every output pixel of the level virtual pinhole.

    Raises:
        ValueError: if the camera's optical axis isn't roughly rig-forward.
    """
    cam_to_rig = _quat_to_matrix(
        extrinsics["qx"], extrinsics["qy"], extrinsics["qz"], extrinsics["qw"]
    )
    # OpenCV camera convention (x right, y down, z forward): the camera's
    # optical axis must point roughly along rig +x (forward)
    forward = cam_to_rig @ np.array([0.0, 0.0, 1.0])
    if forward[0] < 0.9:  # noqa: PLR2004
        msg = f"{CAMERA} optical axis in rig frame is {forward}, expected ~+x"
        raise ValueError(msg)

    u, v = np.meshgrid(np.arange(OUT_W) + 0.5, np.arange(OUT_H) + 0.5)
    x, y = (u - CX) / FX, (v - CY) / FX
    # virtual camera ray -> rig (forward = z, left = -x, up = -y) -> physical camera
    rig = np.stack([np.ones_like(x), -x, -y], axis=-1)
    cam = rig @ cam_to_rig  # == (cam_to_rig.T @ rig.T).T

    th = np.arccos(cam[..., 2] / np.linalg.norm(cam, axis=-1))
    th2r = np.polynomial.Polynomial([intrinsics[f"fw_poly_{i}"] for i in range(5)])
    r = th2r(th)
    rho = np.linalg.norm(cam[..., :2], axis=-1)
    px = intrinsics["cx"] + r * cam[..., 0] / rho
    py = intrinsics["cy"] + r * cam[..., 1] / rho
    # pixel-centre-aware rescale into the decoded frame
    return (px * DECODE_SCALE - 0.5), (py * DECODE_SCALE - 0.5)


def bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(np.int32), 0, h - 2)
    fx = np.clip(sx - x0, 0.0, 1.0)[..., None]
    fy = np.clip(sy - y0, 0.0, 1.0)[..., None]
    img = img.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return (top * (1 - fy) + bot * fy).round().clip(0, 255).astype(np.uint8)


def kinematics(  # noqa: PLR0914
    ego: pl.DataFrame, frame_ts: np.ndarray
) -> dict[str, np.ndarray]:
    t = ego["timestamp"].to_numpy().astype(np.float64)
    qx, qy, qz, qw = (ego[c].to_numpy() for c in ("qx", "qy", "qz", "qw"))
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    px, py = ego["x"].to_numpy(), ego["y"].to_numpy()
    speed = np.hypot(ego["vx"].to_numpy(), ego["vy"].to_numpy()) * 3.6

    fx_ = np.interp(frame_ts, t, px)
    fy_ = np.interp(frame_ts, t, py)
    fyaw = np.interp(frame_ts, t, yaw)
    heading_deg = np.mod(90.0 - np.degrees(fyaw), 360.0)

    # arc length along the driven path; strictly increasing for np.interp
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-3])  # noqa: PLR2004
    s_k, px_k, py_k = s[keep], px[keep], py[keep]
    s_end = s_k[-1]
    # final direction from the last >= 1 m of path, else the final yaw
    tail = np.searchsorted(s_k, s_end - 1.0)
    if s_end - s_k[tail] >= 0.5 and tail < len(s_k) - 1:  # noqa: PLR2004
        d = np.array([px_k[-1] - px_k[tail], py_k[-1] - py_k[tail]])
        d /= np.linalg.norm(d)
    else:
        d = np.array([np.cos(yaw[-1]), np.sin(yaw[-1])])

    s_frame = np.interp(frame_ts, t, s)
    targets = s_frame[:, None] + WAYPOINT_SPACING_M * np.arange(NUM_WAYPOINTS)
    over = np.maximum(targets - s_end, 0.0)
    clamped = np.minimum(targets, s_end)
    wx = np.interp(clamped, s_k, px_k) + over * d[0]
    wy = np.interp(clamped, s_k, py_k) + over * d[1]

    theta = np.radians(heading_deg)[:, None]
    rx, ry = wx - fx_[:, None], wy - fy_[:, None]
    nx = rx * np.cos(theta) - ry * np.sin(theta)
    ny = rx * np.sin(theta) + ry * np.cos(theta)

    return {
        "speed": np.interp(frame_ts, t, speed),
        "heading": heading_deg,
        "gnss_xy": np.stack([fx_, fy_], axis=-1),
        "waypoints_xy": np.stack([wx, wy], axis=-1),
        "waypoints_xy_normalized": np.stack([nx, ny], axis=-1) / 100.0,
        "extrapolated_m": over[:, -1],
        "valid": (frame_ts >= t[0]) & (frame_ts <= t[-1]),
    }


def _read_zip_member(path: Path, name: str) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read(name)


def decode_frames(mp4: bytes) -> np.ndarray:
    w, h = int(1920 * DECODE_SCALE), int(1080 * DECODE_SCALE)
    # a file, not stdin: the mp4 index (moov) may sit at the end, which needs seeking
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        proc = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "ffmpeg",
                "-loglevel",
                "error",
                "-threads",
                "2",
                "-i",
                f.name,
                "-vf",
                f"scale={w}:{h}:flags=area",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    if proc.returncode != 0 or not proc.stdout:
        msg = f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode()[-500:]}"
        raise RuntimeError(msg)
    return np.frombuffer(proc.stdout, np.uint8).reshape(-1, h, w, 3)


def convert_clip(  # noqa: PLR0914
    clip_id: str, chunk: int, raw: Path, out: Path, valid_mask: np.ndarray
) -> dict:
    clip_dir = out / "clips" / clip_id
    if (clip_dir / "meta.parquet").exists():
        return {"clip_id": clip_id, "status": "exists"}

    cam_zip = raw / f"camera/{CAMERA}/{CAMERA}.chunk_{chunk:04d}.zip"
    ego_zip = raw / f"labels/egomotion/egomotion.chunk_{chunk:04d}.zip"
    ego = pl.read_parquet(
        io.BytesIO(_read_zip_member(ego_zip, f"{clip_id}.egomotion.parquet"))
    )
    ts = pl.read_parquet(
        io.BytesIO(_read_zip_member(cam_zip, f"{clip_id}.{CAMERA}.timestamps.parquet"))
    )
    intr = (
        pl
        .read_parquet(
            raw
            / f"calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk:04d}.parquet"
        )
        .filter((pl.col("clip_id") == clip_id) & (pl.col("camera_name") == CAMERA))
        .row(0, named=True)
    )
    extr = (
        pl
        .read_parquet(
            raw
            / f"calibration/sensor_extrinsics/sensor_extrinsics.chunk_{chunk:04d}.parquet"
        )
        .filter((pl.col("clip_id") == clip_id) & (pl.col("sensor_name") == CAMERA))
        .row(0, named=True)
    )

    frames = decode_frames(_read_zip_member(cam_zip, f"{clip_id}.{CAMERA}.mp4"))
    frame_index = ts["frame_index"].to_numpy()
    frame_ts = ts["timestamp"].to_numpy().astype(np.float64)
    n = min(len(frames), len(frame_index))
    frames, frame_index, frame_ts = frames[:n], frame_index[:n], frame_ts[:n]

    sx, sy = remap_grid(intr, extr)
    img_dir = clip_dir / "frames/cam_front_left" / f"{OUT_W}x{OUT_H}"
    img_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in zip(frame_index, frames, strict=True):
        img = bilinear(frame, sx, sy)
        img[~valid_mask] = 0
        (img_dir / f"{idx:09d}.jpg").write_bytes(
            simplejpeg.encode_jpeg(img, quality=90)
        )

    k = kinematics(ego, frame_ts)
    v = k["valid"]
    zeros = np.zeros(int(v.sum()), np.float32)
    meta = pl.DataFrame({
        "meta/ImageMetadata.cam_front_left/time_stamp": pl.Series(
            (EPOCH_US + frame_ts[v]).astype(np.int64)
        ).cast(pl.Datetime("us")),
        "meta/ImageMetadata.cam_front_left/frame_idx": frame_index[v].astype(np.int32),
        "meta/VehicleMotion/speed": k["speed"][v].astype(np.float32),
        "meta/VehicleMotion/gas_pedal_normalized": zeros,
        "meta/VehicleMotion/brake_pedal_normalized": zeros,
        "meta/VehicleMotion/steering_angle_normalized": zeros,
        "meta/VehicleState/turn_signal": np.zeros(int(v.sum()), np.int64),
        "headings_denoised/heading": k["heading"][v].astype(np.float32),
        "meta/Gnss/xy": k["gnss_xy"][v].astype(np.float32),
        "waypoints/xy": k["waypoints_xy"][v].astype(np.float32),
        "waypoints/xy_normalized": k["waypoints_xy_normalized"][v].astype(np.float32),
        "pai/waypoints_extrapolated_m": k["extrapolated_m"][v].astype(np.float32),
    })
    meta.write_parquet(clip_dir / "meta.parquet")
    return {
        "clip_id": clip_id,
        "status": "ok",
        "frames": n,
        "rows": meta.height,
        "max_speed": float(k["speed"].max()),
        "frac_extrapolated_gt_30m": float((k["extrapolated_m"] > 30).mean()),  # noqa: PLR2004
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--clips", type=Path, required=True, help="parquet: clip_id index + chunk"
    )
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    clips = pl.read_parquet(args.clips).select("clip_id", "chunk")
    if args.limit:
        clips = clips.head(args.limit)
    valid_mask = np.asarray(Image.open(args.mask)) > 0
    results = []
    # forkserver: plain fork after polars' thread pool has started deadlocks
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("forkserver")
    ) as pool:
        futures = {
            pool.submit(convert_clip, c, int(k), args.raw, args.out, valid_mask): c
            for c, k in clips.iter_rows()
        }
        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                logger.warning("clip failed", clip_id=futures[f], error=repr(e))
                results.append({"clip_id": futures[f], "status": f"error: {e!r}"})
    report = args.out / "convert_report.json"
    report.write_text(json.dumps(results, indent=1))
    ok = [r for r in results if r["status"] in {"ok", "exists"}]
    logger.info("done", ok=len(ok), failed=len(results) - len(ok), report=str(report))


if __name__ == "__main__":
    main()
