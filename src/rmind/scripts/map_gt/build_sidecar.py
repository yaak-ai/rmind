"""Build per-drive map-GT sidecar parquets (traffic rules / env awareness GT).

Output: ``<out-root>/<Vehicle>/<drive-id>.parquet`` with columns
(shared data contract):

  frame_idx                     Int32   cam_front_left frame index
  time_stamp_us                 Int64
  latitude, longitude           Float64 (NaN if no GNSS)
  max_speed_kmh                 Float32 asserted legal limit; NaN unknown;
                                        -1.0 explicitly unlimited (autobahn)
  road_class                    Utf8    raw OSM highway tag value
  env_class                     Utf8    city|rural|motorway|private|unknown
  osm_way_id                    Int64   nullable
  dist_to_next_traffic_light_m  Float32 NaN if none within 500 m ahead
  dist_to_next_stop_sign_m      Float32 same convention

Extra (non-contract, useful for audits):
  max_speed_derived_kmh Float32 (asserted, else env default: city 50,
  rural 100, motorway -1; NaN for private/unknown), max_speed_source Utf8
  (tag|directional|zone|mcap|none), road_class_source Utf8 (overpass|mcap|none),
  snap_dist_m Float32 (distance frame->matched overpass way).

Sources, in order of preference:
  1. per-drive osm.mcap (platform map matching: highway class + maxspeed per
     time interval) — used for road_class fallback, maxspeed fallback and to
     disambiguate snapping;
  2. Overpass (one cached polyline query per drive): way ids, raw tags
     (maxspeed=none -> -1, zones, ...), traffic_signals + stop nodes.

env_class heuristic (documented deliberately, pragmatic):
  motorway/motorway_link/trunk/trunk_link -> motorway; track/service/path or
  access=private/no -> private; residential/living_street/pedestrian -> city;
  otherwise: asserted limit <= 50 -> city, > 50 -> rural (proxy for place
  polygons, which we do not query); no info at all -> unknown.

Usage (plain venv with polars pyarrow numpy mcap protobuf requests shapely):
  python src/rmind/scripts/map_gt/build_sidecar.py \
      --drives Niro101-HQ/2022-12-25--09-58-33 [more ...]
  python src/rmind/scripts/map_gt/build_sidecar.py \
      --from-train-yaml config/dataset/yaak/train.yaml --stride 40 --limit 17
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import shapely
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from osm_mcap import read_way_intervals  # noqa: E402
from overpass import fetch_route_osm, parse_way_maxspeed  # noqa: E402
from yaak_metadata import parse_metadata_log  # noqa: E402

GNSS_MAX_GAP_US = 3_000_000  # frames farther than this from any fix get NaN pos
WAY_INTERVAL_SLACK_US = 2_000_000
SNAP_MAX_DIST_M = 30.0
CLASS_MATCH_BONUS_M = 15.0  # prefer overpass way whose class matches osm.mcap
NODE_ROUTE_MAX_DIST_M = 30.0
NEXT_NODE_LOOKAHEAD_M = 500.0
NEXT_NODE_BEHIND_TOL_M = 10.0

MOTORWAY_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}
PRIVATE_CLASSES = {"track", "service", "path", "footway", "cycleway", "bridleway"}
CITY_CLASSES = {"residential", "living_street", "pedestrian"}

ENV_DEFAULT_KMH = {"city": 50.0, "rural": 100.0, "motorway": -1.0}


def local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Equirectangular projection to metres around (lat0, lon0)."""
    kx = 111_320.0 * math.cos(math.radians(lat0))
    ky = 110_540.0
    return (lon - lon0) * kx, (lat - lat0) * ky


def classify_env(
    road_class: str | None, access: str | None, max_speed: float
) -> str:
    if road_class is None:
        return "unknown"
    if road_class in MOTORWAY_CLASSES:
        return "motorway"
    if road_class in PRIVATE_CLASSES or access in ("private", "no"):
        return "private"
    if road_class in CITY_CLASSES:
        return "city"
    if not math.isnan(max_speed):
        if max_speed == -1.0:
            return "rural"  # unlimited outside motorway classes: treat as rural
        return "city" if max_speed <= 50.0 else "rural"
    return "rural"  # matched to a public road but no asserted limit


def build_drive(drive: str, data_root: Path, out_root: Path) -> dict:
    vehicle, drive_id = drive.split("/", 1)
    drive_dir = data_root / vehicle / drive_id
    out_path = out_root / vehicle / f"{drive_id}.parquet"

    meta = parse_metadata_log(drive_dir / "metadata.log")
    frames = meta["frames"]
    gnss = meta["gnss"]
    if frames.is_empty():
        msg = f"{drive}: no cam_front_left ImageMetadata"
        raise RuntimeError(msg)

    frame_ts = frames["time_stamp_us"].to_numpy()
    n = len(frame_ts)

    # --- GNSS: linear interpolation onto frame timestamps -------------------
    gts = gnss["time_stamp_us"].to_numpy()
    lat = np.interp(frame_ts, gts, gnss["latitude"].to_numpy())
    lon = np.interp(frame_ts, gts, gnss["longitude"].to_numpy())
    # invalidate frames far from any fix (incl. beyond first/last fix)
    idx = np.searchsorted(gts, frame_ts)
    prev_gap = frame_ts - np.where(idx > 0, gts[np.maximum(idx - 1, 0)], -np.inf)
    next_gap = np.where(idx < len(gts), gts[np.minimum(idx, len(gts) - 1)], np.inf) - frame_ts
    pos_ok = np.minimum(prev_gap, next_gap) <= GNSS_MAX_GAP_US
    lat[~pos_ok] = np.nan
    lon[~pos_ok] = np.nan

    # --- osm.mcap way intervals as-of joined onto frames ---------------------
    mcap_path = drive_dir / "osm.mcap"
    if mcap_path.exists():
        ways_mcap = read_way_intervals(mcap_path)
    else:  # a handful of drives lack osm.mcap -> overpass-only
        print(f"WARNING {drive}: no osm.mcap, using overpass only")
        ways_mcap = pl.DataFrame()
    mcap_class = np.full(n, None, dtype=object)
    mcap_maxspeed = np.zeros(n, dtype=np.float64)
    if not ways_mcap.is_empty():
        starts = ways_mcap["start_us"].to_numpy()
        ends = ways_mcap["end_us"].to_numpy()
        classes = ways_mcap["highway"].to_numpy()
        speeds = ways_mcap["maxspeed_kmh"].to_numpy().astype(np.float64)
        pos = np.searchsorted(starts, frame_ts, side="right") - 1
        valid = pos >= 0
        pos_c = np.maximum(pos, 0)
        valid &= frame_ts <= ends[pos_c] + WAY_INTERVAL_SLACK_US
        mcap_class[valid] = classes[pos_c[valid]]
        mcap_maxspeed[valid] = speeds[pos_c[valid]]

    # --- route + overpass -----------------------------------------------------
    route_path = drive_dir / "map-matched.json"
    route_coords = None
    if route_path.exists():
        geo = json.loads(route_path.read_text())
        for feat in geo.get("features", []):
            if feat.get("geometry", {}).get("type") == "LineString":
                route_coords = feat["geometry"]["coordinates"]
                break

    overpass = None
    if route_coords is not None and len(route_coords) >= 2:
        cache = out_root / "_overpass" / f"{vehicle}__{drive_id}.json"
        try:
            overpass = fetch_route_osm(route_coords, cache)
        except Exception:
            print(f"WARNING {drive}: overpass fetch failed, falling back to osm.mcap only")
            traceback.print_exc()

    lat0 = float(np.nanmean(lat)) if np.any(pos_ok) else 0.0
    lon0 = float(np.nanmean(lon)) if np.any(pos_ok) else 0.0

    # frame points in local metres
    fx, fy = local_xy(lat, lon, lat0, lon0)

    way_id = np.full(n, -1, dtype=np.int64)
    op_class = np.full(n, None, dtype=object)
    op_access = np.full(n, None, dtype=object)
    op_maxspeed = np.full(n, np.nan, dtype=np.float64)
    op_source = np.full(n, None, dtype=object)
    snap_dist = np.full(n, np.nan, dtype=np.float64)

    signals_xy: list[tuple[float, float]] = []
    stops_xy: list[tuple[float, float]] = []

    if overpass is not None:
        ways = [e for e in overpass.get("elements", []) if e["type"] == "way"]
        for e in overpass.get("elements", []):
            if e["type"] != "node":
                continue
            tag = e.get("tags", {}).get("highway")
            x, y = local_xy(np.array([e["lat"]]), np.array([e["lon"]]), lat0, lon0)
            if tag == "traffic_signals":
                signals_xy.append((x[0], y[0]))
            elif tag == "stop":
                stops_xy.append((x[0], y[0]))

        geoms = []
        meta_rows = []
        for w in ways:
            g = w.get("geometry")
            if not g or len(g) < 2:
                continue
            xs, ys = local_xy(
                np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g]),
                lat0, lon0,
            )
            tags = w.get("tags", {})
            ms, ms_src = parse_way_maxspeed(tags)
            geoms.append(LineString(np.column_stack([xs, ys])))
            meta_rows.append(
                (
                    w["id"],
                    tags.get("highway"),
                    tags.get("access"),
                    np.nan if ms is None else ms,
                    ms_src,
                )
            )

        if geoms:
            tree = STRtree(geoms)
            valid_idx = np.flatnonzero(pos_ok)
            pts = shapely.points(fx[valid_idx], fy[valid_idx])
            pairs = tree.query(pts, predicate="dwithin", distance=SNAP_MAX_DIST_M)
            if pairs.size:
                pt_i, way_i = pairs
                dists = shapely.distance(pts[pt_i], np.array(geoms, dtype=object)[way_i])
                # score: distance minus a bonus when class matches osm.mcap class
                cand_class = np.array(
                    [meta_rows[w][1] for w in way_i], dtype=object
                )
                frame_mcap = mcap_class[valid_idx[pt_i]]
                bonus = np.where(
                    (cand_class != None) & (cand_class == frame_mcap),  # noqa: E711
                    CLASS_MATCH_BONUS_M,
                    0.0,
                )
                score = dists - bonus
                order = np.lexsort((score, pt_i))
                pt_sorted = pt_i[order]
                first = np.ones(len(pt_sorted), dtype=bool)
                first[1:] = pt_sorted[1:] != pt_sorted[:-1]
                sel = order[first]
                for k in sel:
                    fi = valid_idx[pt_i[k]]
                    wid, hw, acc, ms, ms_src = meta_rows[way_i[k]]
                    way_id[fi] = wid
                    op_class[fi] = hw
                    op_access[fi] = acc
                    op_maxspeed[fi] = ms
                    op_source[fi] = ms_src
                    snap_dist[fi] = dists[k]

    # --- resolve final columns -------------------------------------------------
    road_class = np.full(n, None, dtype=object)
    road_class_source = np.full(n, "none", dtype=object)
    max_speed = np.full(n, np.nan, dtype=np.float64)
    max_speed_source = np.full(n, "none", dtype=object)

    for i in range(n):
        if op_class[i] is not None:
            road_class[i] = op_class[i]
            road_class_source[i] = "overpass"
        elif mcap_class[i] is not None:
            road_class[i] = mcap_class[i]
            road_class_source[i] = "mcap"

        if not math.isnan(op_maxspeed[i]):
            max_speed[i] = op_maxspeed[i]
            max_speed_source[i] = op_source[i]
        elif mcap_maxspeed[i] > 0:
            max_speed[i] = mcap_maxspeed[i]
            max_speed_source[i] = "mcap"

    env = np.array(
        [classify_env(road_class[i], op_access[i], max_speed[i]) for i in range(n)],
        dtype=object,
    )
    derived = np.array(
        [
            max_speed[i]
            if not math.isnan(max_speed[i])
            else ENV_DEFAULT_KMH.get(env[i], np.nan)
            for i in range(n)
        ],
        dtype=np.float64,
    )

    # --- distance along route to next traffic light / stop sign ----------------
    dist_signal = np.full(n, np.nan, dtype=np.float64)
    dist_stop = np.full(n, np.nan, dtype=np.float64)
    if route_coords is not None and len(route_coords) >= 2:
        rc = np.asarray(route_coords, dtype=np.float64)
        rx, ry = local_xy(rc[:, 1], rc[:, 0], lat0, lon0)
        route_line = LineString(np.column_stack([rx, ry]))

        def _along(nodes_xy: list[tuple[float, float]]) -> np.ndarray:
            ss = []
            for x, y in nodes_xy:
                p = Point(x, y)
                if route_line.distance(p) <= NODE_ROUTE_MAX_DIST_M:
                    ss.append(route_line.project(p))
            return np.sort(np.asarray(ss, dtype=np.float64))

        s_signals = _along(signals_xy)
        s_stops = _along(stops_xy)

        ok = np.flatnonzero(pos_ok)
        if ok.size:
            frame_pts = shapely.points(fx[ok], fy[ok])
            s_frame = shapely.line_locate_point(route_line, frame_pts)
            for s_nodes, out in ((s_signals, dist_signal), (s_stops, dist_stop)):
                if s_nodes.size == 0:
                    continue
                j = np.searchsorted(s_nodes, s_frame - NEXT_NODE_BEHIND_TOL_M)
                has = j < len(s_nodes)
                d = np.full(ok.size, np.nan)
                d[has] = np.maximum(s_nodes[j[has]] - s_frame[has], 0.0)
                d[d > NEXT_NODE_LOOKAHEAD_M] = np.nan
                out[ok] = d

    df = pl.DataFrame(
        {
            "frame_idx": frames["frame_idx"].cast(pl.Int32),
            "time_stamp_us": frames["time_stamp_us"],
            "latitude": lat,
            "longitude": lon,
            "max_speed_kmh": max_speed.astype(np.float32),
            "road_class": pl.Series(
                [c if c is not None else "unknown" for c in road_class], dtype=pl.Utf8
            ),
            "env_class": pl.Series(list(env), dtype=pl.Utf8),
            "osm_way_id": pl.Series(
                [int(w) if w >= 0 else None for w in way_id], dtype=pl.Int64
            ),
            "dist_to_next_traffic_light_m": dist_signal.astype(np.float32),
            "dist_to_next_stop_sign_m": dist_stop.astype(np.float32),
            "max_speed_derived_kmh": derived.astype(np.float32),
            "max_speed_source": pl.Series(list(max_speed_source), dtype=pl.Utf8),
            "road_class_source": pl.Series(list(road_class_source), dtype=pl.Utf8),
            "snap_dist_m": snap_dist.astype(np.float32),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    known = float(np.mean(~np.isnan(max_speed)))
    return {
        "drive": drive,
        "n_frames": n,
        "coverage_max_speed": known,
        "coverage_road_class": float(np.mean([c is not None for c in road_class])),
        "overpass": overpass is not None,
        "env_counts": {
            k: int(v) for k, v in zip(*np.unique(env.astype(str), return_counts=True))
        },
        "out": str(out_path),
    }


def drives_from_train_yaml(path: Path) -> list[str]:
    txt = path.read_text()
    match = re.search(r"input_id:\n((?:    - .+\n)+)", txt)
    if match is None:
        msg = f"could not find samples.inputs.input_id list in {path}"
        raise ValueError(msg)
    return re.findall(r"    - (\S+/\S+)", match.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drives", nargs="*", default=[], help="Vehicle/drive-id ...")
    ap.add_argument("--from-train-yaml", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1, help="take every k-th drive")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-root", type=Path, default=Path("/nasa/drives/yaak/data"))
    ap.add_argument("--out-root", type=Path, default=Path("caches/map_gt"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    drives = list(args.drives)
    if args.from_train_yaml is not None:
        drives += drives_from_train_yaml(args.from_train_yaml)[:: args.stride]
    if args.limit is not None:
        drives = drives[: args.limit]
    if not drives:
        ap.error("no drives given (use --drives or --from-train-yaml)")

    results = []
    for drive in drives:
        vehicle, drive_id = drive.split("/", 1)
        out_path = args.out_root / vehicle / f"{drive_id}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"skip (exists): {drive}")
            continue
        try:
            res = build_drive(drive, args.data_root, args.out_root)
        except Exception:
            print(f"FAILED {drive}")
            traceback.print_exc()
            continue
        results.append(res)
        print(
            f"{drive}: {res['n_frames']} frames, "
            f"max_speed coverage {res['coverage_max_speed']:.1%}, "
            f"road_class coverage {res['coverage_road_class']:.1%}, "
            f"env {res['env_counts']}"
        )

    if results:
        summary_path = args.out_root / "build_summary.json"
        existing = []
        if summary_path.exists():
            existing = json.loads(summary_path.read_text())
            done = {r["drive"] for r in results}
            existing = [r for r in existing if r["drive"] not in done]
        summary_path.write_text(json.dumps(existing + results, indent=2))
        print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
