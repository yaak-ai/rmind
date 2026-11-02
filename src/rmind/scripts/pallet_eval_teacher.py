"""Score pallet teachers (`pallet_train_teacher`): learned 3D pose vs floor-free PnP, plus heads.

On the held-out Veigel scene (`--synthetic`, a `pallet_synthetic_veigel --labels-out` dir,
test split; pinhole intrinsics from its `.pose.json`), per pallet crop (its tight box):

    learned       the pose head (`teacher.decode_pose`)
    pnp6 / pnp4   floor-free PnP (`pose_free`) on the SYNTHETIC-head keypoints with
                  confidence > `--min-conf`: 6 DoF, and upright (4 DoF, level pallet). The
                  3D layout comes from the predicted type: eur -> the EUR model's layout,
                  other -> whichever non-EUR model reprojects best
    *_oracle      the same PnP with the pallet's true model layout (an upper bound)
    floor_pnp     today's floor-constrained `pose.solve` on the same keypoints, for contrast

Errors: entry-face centre in cm (3D, and |forward| / |left| / |up|), yaw in degrees; median
and p90; per floor vs elevated (stacked / rack: face centre above the floor band) and per
forward-distance bin. Each method on the pallets it returns a pose for (coverage given), and
all methods on the pallets where every method returns one ("common"). Also: loaded accuracy
(trusted Veigel tags), type accuracy, synthetic-head keypoint px on Veigel (all and the
standard `--standard-keys` subset), and on the real val jobs the hand-label-head keypoint px
and load accuracy. `--real-pose` adds a sanity check on real val pallets: the learned pose
(through the fisheye) vs the floor-constrained `pose.solve` on the hand-head keypoints.

    direnv exec . uv run --no-sync python -m rmind.scripts.pallet_eval_teacher \
        --checkpoints .../teacher_v1/hrnet_w18/last.pt --synthetic .../teacher_v1/veigel_relaxed \
        --labels .../entry_labels_v4.json --images .../images/entry --boxes ...detboxes.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from structlog import get_logger
from torch.utils.data import DataLoader

from rmind.pallet.correspondence import POINTS_3D
from rmind.pallet.entry import EntryKeypoints, load_labels
from rmind.pallet.geometry import CAM_FORK, FisheyeCamera
from rmind.pallet.keypoints import Crop, decode
from rmind.pallet.pose import solve
from rmind.pallet.pose_free import FreePose, solve_6dof, solve_upright
from rmind.pallet.teacher import (
    TYPES,
    PalletTeacherNet,
    TeacherCrops,
    decode_pose,
    layouts_from_poses,
)
from rmind.scripts.pallet_train_teacher import LOAD, evaluate, synthetic_meta

logger = get_logger(__name__)
EUR_MODEL = "SM_BlockPallet_A07_WoodNewA_01"
VEIGEL_CAMERA_HEIGHT = 0.44
DISTANCE_BINS = ((0.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99.0))
# the EUR template's face centre: half the deck height above the floor
EUR_FACE_CENTRE_Z = float(POINTS_3D[:, 2].max()) / 2


def run_net(
    net: PalletTeacherNet, loader: DataLoader, device: str
) -> dict[str, np.ndarray]:
    """Per-crop outputs and targets, concatenated over the loader."""
    keep = (
        "keypoints",
        "visible",
        "scale",
        "crop",
        "intrinsics",
        "pose",
        "loaded",
        "type",
    )
    rows: dict[str, list] = {}
    net.eval()
    with torch.no_grad():
        for b in loader:
            out = net.forward_all(b["image"].to(device), b["geometry"].to(device))
            out = {k: v.cpu() for k, v in out.items()}
            for head in ("heatmaps", "heatmaps_syn"):
                px, conf = decode(out[head])
                rows.setdefault(f"{head}_px", []).append(px)
                rows.setdefault(f"{head}_conf", []).append(conf)
            rows.setdefault("pred_pose", []).append(
                decode_pose(out, b["crop"], b["intrinsics"])
            )
            rows.setdefault("pred_pose_fisheye", []).append(
                decode_pose(out, b["crop"], b["intrinsics"], camera=CAM_FORK)
            )
            rows.setdefault("loaded_logit", []).append(out["loaded"])
            rows.setdefault("type_logit", []).append(out["type"])
            for k in keep:
                rows.setdefault(k, []).append(b[k])
    return {k: torch.cat(v).double().numpy() for k, v in rows.items()}


def keypoint_px(r: dict[str, np.ndarray], head: str, mask: np.ndarray) -> dict:
    err = (
        np.linalg.norm(r[f"{head}_px"] - r["keypoints"], axis=-1) * r["scale"][:, None]
    )
    e = err[r["visible"].astype(bool) & mask[:, None]]
    return {
        "median_px": float(np.median(e)),
        "p90_px": float(np.quantile(e, 0.9)),
        "pck5": float((e < 5).mean()),
        "n": len(e),
    }


def frame_pixels(r: dict[str, np.ndarray], i: int, head: str) -> np.ndarray:
    return Crop(*r["crop"][i]).to_frame(r[f"{head}_px"][i])


def pnp(
    pixels: np.ndarray,
    visible: np.ndarray,
    layouts: list[np.ndarray],
    k: np.ndarray,
    *,
    upright: bool,
) -> FreePose | None:
    """Best (lowest rms) floor-free PnP over candidate layouts."""
    best = None
    for layout in layouts:
        s = solve_6dof(pixels, visible, layout, k=k)
        if s is not None and upright:
            s = solve_upright(pixels, visible, layout, s, k=k)
        if s is not None and (best is None or s.rms_px < best.rms_px):
            best = s
    return best


def errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """`(N, 4)` (forward, left, up, yaw rad) -> `(N, 5)` |d| cm x3, 3D cm, |yaw| deg."""
    d = pred - gt
    yaw = np.degrees(np.abs(np.arctan2(np.sin(d[:, 3]), np.cos(d[:, 3]))))
    cm = np.abs(d[:, :3]) * 100
    return np.column_stack([cm, np.linalg.norm(d[:, :3], axis=1) * 100, yaw])


def summarise(err: np.ndarray, ok: np.ndarray, mask: np.ndarray) -> dict:
    sel = ok & mask
    e = err[sel]
    if not len(e):
        return {"n": 0, "coverage": 0.0}
    med, p90 = np.median(e, axis=0), np.quantile(e, 0.9, axis=0)
    return {
        "n": int(sel.sum()),
        "coverage": float(sel.sum() / max(mask.sum(), 1)),
        "fwd_cm": float(med[0]),
        "left_cm": float(med[1]),
        "up_cm": float(med[2]),
        "pos_cm": float(med[3]),
        "pos_cm_p90": float(p90[3]),
        "yaw_deg": float(med[4]),
        "yaw_deg_p90": float(p90[4]),
    }


def floor_pnp(
    r: dict[str, np.ndarray],
    is_other: np.ndarray,
    others: list[np.ndarray],
    eur: np.ndarray,
) -> np.ndarray:
    """Today's pipeline for contrast: `pose.solve` (on the floor, free pitch) on the same
    synthetic-head keypoints and type-chosen layout, the Veigel pinhole at 0.44 m."""
    p = np.full((len(r["pose"]), 4), np.nan)
    for i in range(len(p)):
        fx, fy, cx, cy = r["intrinsics"][i]
        camera = FisheyeCamera(
            fx,
            fy,
            cx,
            cy,
            1920,
            1080,
            k=(0.0,) * 8,
            model="rational",
            position=(0.0, 0.0, VEIGEL_CAMERA_HEIGHT),
            max_theta=np.deg2rad(80.0),  # pinhole: no fold, but rays need theta < 90
        )
        conf = r["heatmaps_syn_conf"][i]
        best = None
        for layout in others if is_other[i] else [eur]:
            half = -layout[:, 2].min()  # face centre above the pallet's base
            est = solve(
                camera,
                frame_pixels(r, i, "heatmaps_syn"),
                conf > ARGS.min_conf,
                confidence=conf,
                model=layout + np.array([0.0, 0.0, half]),
            )
            if est is not None and (best is None or est.rms_px < best[0]):
                best = (est.rms_px, est.pose, half)
        if best is not None:
            _, pose, half = best
            p[i] = [pose.x, pose.y, half - VEIGEL_CAMERA_HEIGHT, pose.yaw]
    return p


def pose_table(
    r: dict[str, np.ndarray], meta: list[dict], layouts: dict[str, np.ndarray]
) -> dict:
    n = len(meta)
    gt = r["pose"]
    is_other = r["type_logit"] > 0
    others = [v for m, v in layouts.items() if m != EUR_MODEL]
    preds: dict[str, np.ndarray] = {"learned": r["pred_pose"]}
    for name, upright, oracle in (
        ("pnp6", False, False),
        ("pnp4", True, False),
        ("pnp6_oracle", False, True),
        ("pnp4_oracle", True, True),
    ):
        p = np.full((n, 4), np.nan)
        for i in range(n):
            vis = r["heatmaps_syn_conf"][i] > ARGS.min_conf
            cands = (
                [layouts[meta[i]["model"]]]
                if oracle
                else (others if is_other[i] else [layouts[EUR_MODEL]])
            )
            s = pnp(
                frame_pixels(r, i, "heatmaps_syn"),
                vis,
                cands,
                r["intrinsics"][i],
                upright=upright,
            )
            if s is not None:
                p[i] = [*s.centre, s.yaw]
        preds[name] = p
    preds["floor_pnp"] = floor_pnp(r, is_other, others, layouts[EUR_MODEL])
    floor = np.array([m["floor"] for m in meta])
    fwd = gt[:, 0]
    groups: dict[str, np.ndarray] = {
        "all": np.ones(n, dtype=bool),
        "floor": floor,
        "elevated": ~floor,
    }
    for lo, hi in DISTANCE_BINS:
        groups[f"fwd_{lo:g}-{hi:g}m"] = (fwd >= lo) & (fwd < hi)
    for lo, hi in DISTANCE_BINS:
        groups[f"elevated_fwd_{lo:g}-{hi:g}m"] = ~floor & (fwd >= lo) & (fwd < hi)
    oks = {k: np.isfinite(p).all(axis=1) for k, p in preds.items()}
    common = np.logical_and.reduce(list(oks.values()))
    table: dict = {}
    for k, p in preds.items():
        err = errors(np.nan_to_num(p), gt)
        table[k] = {g: summarise(err, oks[k], m) for g, m in groups.items()}
        table[f"{k}@common"] = {g: summarise(err, common, m) for g, m in groups.items()}
    return table


def real_pose_check(r: dict[str, np.ndarray]) -> dict:
    """Real val pallets (on the floor): learned pose and floor-free upright PnP (both through
    the fisheye, EUR layout for PnP) against the floor-constrained `pose.solve` on the same
    hand-head keypoints. Agreement, not accuracy: real frames have no metric ground truth."""
    eur = POINTS_3D - np.array([
        0.0,
        0.0,
        EUR_FACE_CENTRE_Z,
    ])  # origin at the face centre
    cam = np.asarray(CAM_FORK.position)
    diffs: dict[str, list] = {"learned": [], "pnp4": []}
    for i in range(len(r["pose"])):
        conf = r["heatmaps_conf"][i]
        pixels, visible = frame_pixels(r, i, "heatmaps"), conf > ARGS.min_conf
        est = solve(CAM_FORK, pixels, visible, confidence=conf, model=POINTS_3D)
        if est is None:
            continue
        # robot frame -> camera forward / left / up (level camera, no yaw)
        floor = np.array([
            est.pose.x - cam[0],
            est.pose.y - cam[1],
            EUR_FACE_CENTRE_Z - cam[2],
            est.pose.yaw,
        ])[None]
        diffs["learned"].append(errors(r["pred_pose_fisheye"][i][None], floor)[0])
        s6 = solve_6dof(pixels, visible, eur, camera=CAM_FORK)
        s4 = (
            None
            if s6 is None
            else solve_upright(pixels, visible, eur, s6, camera=CAM_FORK)
        )
        if s4 is not None:
            diffs["pnp4"].append(errors(np.array([[*s4.centre, s4.yaw]]), floor)[0])
    out = {}
    for k, v in diffs.items():
        if not v:
            out[k] = {"n": 0}
            continue
        med = np.median(np.array(v), axis=0)
        out[k] = {
            "n": len(v),
            "fwd_cm": float(med[0]),
            "left_cm": float(med[1]),
            "up_cm": float(med[2]),
            "pos_cm": float(med[3]),
            "yaw_deg": float(med[4]),
        }
    return out


def classification(r: dict[str, np.ndarray], meta: list[dict]) -> dict:
    loaded_known = r["loaded"] >= 0
    loaded_hit = (r["loaded_logit"] > 0) == (r["loaded"] > 0)
    floor = np.array([m["floor"] for m in meta])
    type_hit = (r["type_logit"] > 0) == (r["type"] > 0)
    out = {
        "loaded_acc": float(loaded_hit[loaded_known].mean()),
        "loaded_n": int(loaded_known.sum()),
        "loaded_acc_on_loaded": float(loaded_hit[r["loaded"] == 1].mean()),
        "type_acc": float(type_hit.mean()),
        "type_acc_floor": float(type_hit[floor].mean()),
        "type_acc_elevated": float(type_hit[~floor].mean()),
    }
    for t, name in enumerate(TYPES):
        out[f"type_recall_{name}"] = float(type_hit[r["type"] == t].mean())
    return out


ARGS: argparse.Namespace


def main() -> None:
    global ARGS  # ruff: ignore[global-statement]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument(
        "--standard-keys",
        type=Path,
        default=Path(
            "/nasa/team-space/nikita/pallet-detector/datasets/synthetic/veigel-entry-500/entry_test.json"
        ),
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--val-jobs",
        nargs="+",
        default=[
            "2026-09-10--15-17-38",
            "2026-10-02--15-48-24",
            "2026-10-22--09-50-09_03",
            "2026-10-22--09-50-09_04",
            "2026-10-22--09-50-09_05",
        ],
    )
    parser.add_argument("--min-conf", type=float, default=0.3)
    parser.add_argument("--real-pose", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    ARGS = args = parser.parse_args()

    syn_labels, syn_boxes, meta = synthetic_meta(args.synthetic, "test")
    layouts = layouts_from_poses(
        json.loads((args.synthetic / "entry_train.pose.json").read_text())
    )
    syn_base = EntryKeypoints(Path("/"), syn_labels, syn_boxes, train=False)
    syn_loader = DataLoader(TeacherCrops(syn_base, meta), batch_size=16, num_workers=6)
    syn_meta = [meta[name]["pose"] for name, *_ in syn_base.items]
    standard = set(json.loads(args.standard_keys.read_text()))
    in_standard = np.array([name in standard for name, *_ in syn_base.items])

    labels = load_labels(args.labels)
    load_tags = json.loads(args.labels.with_suffix(".load.json").read_text())
    real_meta = {n: {"loaded": LOAD[t]} for n, t in load_tags.items() if t in LOAD}
    boxes: dict = {}
    for f in args.boxes:
        boxes |= json.loads(f.read_text())
    val_labels = {
        n: v for n, v in labels.items() if any(n.startswith(j) for j in args.val_jobs)
    }
    real_loader = DataLoader(
        TeacherCrops(
            EntryKeypoints(args.images, val_labels, boxes, train=False), real_meta
        ),
        batch_size=16,
        num_workers=4,
    )

    results = {}
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = PalletTeacherNet(ck["backbone"], pretrained=False)
        net.load_state_dict(ck["state_dict"])
        net.to(args.device)
        key = f"{path.parent.name}/{path.stem}"
        r = run_net(net, syn_loader, args.device)
        res = {
            "veigel_pose": pose_table(r, syn_meta, layouts),
            "veigel_cls": classification(r, syn_meta),
            "veigel_kp_syn_head": keypoint_px(
                r, "heatmaps_syn", np.ones(len(syn_meta), dtype=bool)
            ),
            "veigel_kp_syn_head_standard": keypoint_px(r, "heatmaps_syn", in_standard),
            "veigel_kp_real_head_standard": keypoint_px(r, "heatmaps", in_standard),
            "real_val": evaluate(net, real_loader, args.device),
        }
        if args.real_pose:
            res["real_pose_vs_floor_solve"] = real_pose_check(
                run_net(net, real_loader, args.device)
            )
        results[key] = res
        report(key, res)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))


def _fmt(s: dict) -> str:
    if not s["n"]:
        return "n=0"
    return (
        f"n={s['n']:3d} cov={s['coverage']:.2f} pos={s['pos_cm']:6.1f}/{s['pos_cm_p90']:6.1f}cm "
        f"(f {s['fwd_cm']:5.1f} l {s['left_cm']:5.1f} u {s['up_cm']:5.1f}) "
        f"yaw={s['yaw_deg']:5.2f}/{s['yaw_deg_p90']:5.2f}deg"
    )


def report(key: str, res: dict) -> None:
    lines = [f"== {key}"]
    lines.extend(
        f"{method:20s} {g:24s} {_fmt(s)}"
        for method, groups in res["veigel_pose"].items()
        for g, s in groups.items()
    )
    lines.extend(f"{k} {json.dumps(v)}" for k, v in res.items() if k != "veigel_pose")
    print("\n".join(lines))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
