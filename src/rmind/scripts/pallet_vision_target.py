"""The pick-up target from the camera instead of the UWB pallet tag.

For every pickup (`pallet_pickup_targets`: the pallet on the forks at the lift, tracked back
through the detections) the target box at each detection step is matched to the keypoint
pass (`pallet_predict_candidates`, same detector boxes, grown upwards) and its floor pose is
taken when the keypoints agree with it (>= `--min-inliers`, rms <= `--max-rms` px). That
pose, (x, y, yaw) of the entry-face centre in the robot frame (origin on the floor at the
fork base, x towards the forks' tips, y left), is the vision target for that frame.

Writes `<out>/<job>.parquet`: one row per detection step with a target pose
(pickup, pickup_frame, frame, x_m, y_m, yaw_rad, p_loaded, box). Only pickups with
`carried` >= `--min-carried` (a pallet really was on the forks afterwards).

With `--compare-uwb`, also joins the UWB target where the recording has one: the pallet tag
position relative to the truck tag, rotated into the truck frame exactly as the policy's
`relative_ego_pos` (before its /10 m scaling), and writes per-job agreement statistics to
`<out>/uwb_agreement.json` (joined rows in `uwb_joined.parquet`). The two use different
origins: the truck tag is a fixed offset from the fork base, and the pallet tag sits
somewhere on the pallet, so its offset from the entry face turns with the pallet's yaw.
Both offsets are fitted; the residual is the disagreement.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from structlog import get_logger

from rmind.scripts.pallet_pickup_targets import d12  # applies the missing-UWB patch
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT

logger = get_logger(__name__)


def match(box: list[float], rows: pl.DataFrame) -> dict | None:
    """The keypoint-pass row for a target box: same left, right and bottom edge (the pass
    grows boxes upwards only)."""
    for r in rows.iter_rows(named=True):
        b = r["box"]
        if abs(b[0] - box[0]) < 1 and abs(b[2] - box[2]) < 1 and abs(b[3] - box[3]) < 1:
            return r
    return None


def vision_targets(
    job: str, pickups: list[dict], kp: pl.DataFrame, args: argparse.Namespace
) -> pl.DataFrame:
    if kp.is_empty():
        return pl.DataFrame()
    by_frame = {f: g for (f,), g in kp.group_by("frame")}
    rows = []
    for i, p in enumerate(pickups):
        if p["carried"] < args.min_carried:
            continue
        for f, box in p["target"].items():
            r = match(box, by_frame.get(int(f), kp.clear()))
            if (
                r is None
                or r["pose"] is None
                or r["inliers"] < args.min_inliers
                or r["rms_px"] > args.max_rms
            ):
                continue
            x, y, yaw = r["pose"]
            rows.append({
                "job": job,
                "pickup": i,
                "pickup_frame": p["frame"],
                "frame": int(f),
                "x_m": x,
                "y_m": y,
                "yaw_rad": yaw,
                "p_loaded": r["p_loaded"],
                "box": box[:4],
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def uwb_relative(job: str, max_skew_s: float) -> pl.DataFrame | None:
    """cam_fork frame -> UWB pallet position in the truck frame (m), as `relative_ego_pos`."""
    job_dir = D12_ROOT / job
    d12.MCAP_FILE = (
        "data_for_training.mcap"
        if (job_dir / "data_for_training.mcap").is_file()
        else "data.mcap"
    )
    t = d12.row_table(
        job_dir,
        cameras=("cam_fork",),
        reference="cam_fork",
        traction_source="reported",
        max_skew_s=max_skew_s,
    )
    if "pallet_x" not in t.columns:
        return None
    th = np.radians(t["ego_heading"].to_numpy())
    dx = t["pallet_x"].to_numpy() - t["ego_x"].to_numpy()
    dy = t["pallet_y"].to_numpy() - t["ego_y"].to_numpy()
    return pl.DataFrame({
        "frame": t["cam_fork/frame_idx"].to_numpy().astype(np.int64),
        "uwb_fwd_m": np.cos(th) * dx + np.sin(th) * dy,
        "uwb_lat_m": -np.sin(th) * dx + np.cos(th) * dy,
    })


def agreement(v: pl.DataFrame) -> dict:
    """Vision vs UWB, explaining the two tag offsets: UWB = vision + R(yaw) a + b.

    `a` is the pallet tag's position in the pallet frame (it turns with the pallet's
    yaw relative to the truck), `b` the truck tag's offset from the fork base. Both are
    fitted (least squares, 4 numbers); the residual is the disagreement.
    """
    vx, vy, yaw = v["x_m"].to_numpy(), v["y_m"].to_numpy(), v["yaw_rad"].to_numpy()
    du = np.r_[v["uwb_fwd_m"].to_numpy() - vx, v["uwb_lat_m"].to_numpy() - vy]
    c, s_, one, zero = np.cos(yaw), np.sin(yaw), np.ones(len(v)), np.zeros(len(v))
    design = np.c_[np.r_[c, s_], np.r_[-s_, c], np.r_[one, zero], np.r_[zero, one]]
    coef, *_ = np.linalg.lstsq(design, du, rcond=None)
    res = (du - design @ coef).reshape(2, -1)
    err = np.hypot(res[0], res[1])
    return {
        "n": len(v),
        "pallet_tag_m": [round(float(coef[0]), 3), round(float(coef[1]), 3)],
        "truck_offset_m": [round(float(coef[2]), 3), round(float(coef[3]), 3)],
        "median_err_m": float(np.median(err)),
        "p90_err_m": float(np.quantile(err, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pickups", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-carried", type=float, default=0.5)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--max-rms", type=float, default=6.0)
    parser.add_argument("--compare-uwb", action="store_true")
    parser.add_argument("--max-skew-s", type=float, default=40.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats, all_rows = {}, []
    for path in sorted(args.pickups.glob("*.json")):
        if (
            path.stem == "summary"
            or not (args.keypoints / f"{path.stem}.parquet").exists()
        ):
            continue
        job = path.stem
        pickups = json.loads(path.read_text())["pickups"]
        v = vision_targets(
            job, pickups, pl.read_parquet(args.keypoints / f"{job}.parquet"), args
        )
        if v.is_empty():
            continue
        v.write_parquet(args.out / f"{job}.parquet")
        if args.compare_uwb:
            try:
                u = uwb_relative(job, args.max_skew_s)
            except Exception as e:  # ruff: ignore[blind-except] - recordings the loader cannot pair
                logger.warning("uwb skipped", job=job, error=str(e)[:120])
                u = None
            if u is not None:
                j = v.join(u, on="frame", how="inner").drop_nulls([
                    "uwb_fwd_m",
                    "uwb_lat_m",
                ])
                if len(j) >= 10:
                    stats[job] = agreement(j)
                    all_rows.append(
                        j.select(
                            "job",
                            "pickup",
                            "frame",
                            "x_m",
                            "y_m",
                            "yaw_rad",
                            "uwb_fwd_m",
                            "uwb_lat_m",
                        )
                    )
        logger.info("job", job=job, targets=len(v), uwb=job in stats)
    if all_rows:
        joined = pl.concat(all_rows)
        joined.write_parquet(args.out / "uwb_joined.parquet")
        stats["_all"] = agreement(joined)
        (args.out / "uwb_agreement.json").write_text(json.dumps(stats, indent=1))
        logger.info("uwb agreement", **stats["_all"])


if __name__ == "__main__":
    main()
