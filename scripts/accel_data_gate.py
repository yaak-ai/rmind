"""Phase 4.1 step 23 of docs/action_tokenizer_repair_plan.md -- the data gate.

`VehicleMotion.acceleration_x/y/z` was verified populated on the 15-drive
*predict* set (2026-07-15). This script re-establishes that on the *train*
population (619-655 Niro-HQ drives) before anything is built on top of it, and
answers the other three step-23 questions on whichever split(s) are run:

  1. Populated on the train drives? n_unique / range / zero-fraction per
     vehicle generation.
  2. Which axis is longitudinal, and what sign? Correlate a_x/a_y/a_z against
     realized dv over consecutive within-chunk ticks.
  3. The decisive number: at v < 5 km/h AND brake > 0.3 (where dv is proven
     degenerate -- see docs/action_tokenizer_repair_plan.md #10), does measured
     acceleration separate brake levels? Quantified alongside stationary-no-pedal
     grade/pitch contamination.
  4. The invertibility ceiling: fit (acceleration, speed) -> (gas, brake) on a
     held-out drive split, per SPEED_BANDS, against Phase 2a's ceiling table.

No training. Builds each split's rbyte sample table from raw metadata.log --
NO video decode (`action_train`/`action_val` set `streams: {}`; `predict` here
uses the plain `dataset/yaak/predict` config, image streams unused for this
analysis and simply not read).

**Cache safety**: pass a FRESH `--cache-dir` that has never been used by this
config before. rbyte/pipefunc's `run_folder` store returns any existing
per-output-name file regardless of the schema that produced it (confirmed
cross-run contamination bug, see
[[project_rbyte_cache_contamination_risk]]) -- reusing an old cache dir here
would silently serve pre-acceleration `meta`/`aligned` frames.

Usage:
    nix develop --command uv run python scripts/accel_data_gate.py \\
        --config-dir /home/alex/rmind/config \\
        --split train val predict \\
        --cache-dir .rbyte_cache_accel_gate \\
        --out docs/accel_data_gate.json

    # prototype on a handful of drives first (minutes, not hours):
    nix develop --command uv run python scripts/accel_data_gate.py \\
        --split train --max-drives 20 --cache-dir .rbyte_cache_accel_gate_proto
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import rmind  # noqa: F401  registers the `eval` OmegaConf resolver

SPEED_BANDS = ((0, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 130))
PERCENTILES = (0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9)
NEARSTOP_KMH = 5.0
BRAKE_PRESSED = 0.3
STATIONARY_KMH = 0.5
NO_PEDAL = 0.02
ACTION_STEP_FRAMES = 10  # matches action_tokenizer/pretrain.yaml AND predict.yaml
FRAME_HZ = 30.0
DT_NOMINAL_S = ACTION_STEP_FRAMES / FRAME_HZ

FIELDS = ("acceleration_x", "acceleration_y", "acceleration_z")
VM = "meta/VehicleMotion/"


def _load_split(
    *, split: str, config_dir: str, cache_dir: str, max_drives: int | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Instantiate one split's rbyte dataset and return `(arrays, drive_ids)`.

    `arrays[name]` has shape `(n_samples, clip)`; `drive_ids` has shape
    `(n_samples,)`. Import hydra/omegaconf lazily so `--help` doesn't need a
    nix env.
    """
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        if split in {"train", "val"}:
            cfg = compose(
                config_name="train",
                overrides=[
                    "experiment=yaak/action_tokenizer/pretrain",
                    f"paths.rbyte.cache={cache_dir}",
                ],
            )
            ds_cfg = (
                cfg.datamodule.train.dataset
                if split == "train"
                else cfg.datamodule.val.dataset
            )
        elif split == "predict":
            cfg = compose(
                config_name="train",
                overrides=[
                    "+datamodule=yaak/predict",
                    "+paths=yaak/default",
                    f"paths.rbyte.cache={cache_dir}",
                ],
            )
            ds_cfg = cfg.datamodule.predict.dataset
        else:
            msg = f"unknown split {split!r}"
            raise ValueError(msg)

    if max_drives is not None:
        OmegaConf.set_struct(ds_cfg, False)
        ids = list(ds_cfg.samples.inputs.input_id)[:max_drives]
        paths_ = list(ds_cfg.samples.inputs.yaak_metadata_path)[:max_drives]
        ds_cfg.samples.inputs.input_id = ids
        ds_cfg.samples.inputs.yaak_metadata_path = paths_

    t0 = time.time()
    dataset = instantiate(ds_cfg)
    print(f"[{split}] built {len(dataset)} samples in {time.time() - t0:.1f}s")  # noqa: T201

    names = [
        "speed",
        *FIELDS,
        "gas_pedal_normalized",
        "brake_pedal_normalized",
        "steering_angle_normalized",
    ]
    arrays = {name: dataset.data[f"{VM}{name}"].numpy() for name in names}
    drive_ids = dataset.meta["input_id"].cast(pl.String).to_numpy()
    return arrays, drive_ids


def _generation(drive_ids: np.ndarray) -> np.ndarray:
    """`Niro096-HQ/...` -> `Niro`; `G1-00428/...` -> `G1`."""
    vehicle = np.array([d.split("/")[0] for d in drive_ids])
    return np.where(np.char.startswith(vehicle, "G1"), "G1", "Niro")


def question_1(arrays: dict[str, np.ndarray], drive_ids: np.ndarray) -> dict[str, Any]:
    """Populated on the train drives? Per vehicle generation."""
    generation = _generation(drive_ids)
    out: dict[str, Any] = {
        "n_samples": int(drive_ids.size),
        "n_drives": int(np.unique(drive_ids).size),
    }
    for gen in sorted(set(generation.tolist())):
        mask = generation == gen
        gen_out: dict[str, Any] = {
            "n_samples": int(mask.sum()),
            "n_drives": int(np.unique(drive_ids[mask]).size),
        }
        for field in FIELDS:
            v = arrays[field][mask].ravel()
            v = v[np.isfinite(v)]
            gen_out[field] = {
                "n_unique": int(np.unique(v).size) if v.size else 0,
                "min": float(v.min()) if v.size else None,
                "max": float(v.max()) if v.size else None,
                "mean": float(v.mean()) if v.size else None,
                "std": float(v.std()) if v.size else None,
                "zero_frac": float((v == 0.0).mean()) if v.size else None,
                "percentiles": {str(p): float(np.percentile(v, p)) for p in PERCENTILES}
                if v.size
                else None,
            }
        out[gen] = gen_out
    return out


def _tick_pairs(
    arrays: dict[str, np.ndarray], drive_ids: np.ndarray
) -> dict[str, np.ndarray]:
    """Consecutive within-chunk tick pairs -- `(left, right)` are exactly
    `ACTION_STEP_FRAMES` raw frames apart by construction (`gather_every` in
    the dataset config), so `DT_NOMINAL_S` is a fixed per-pair dt, not fitted.
    5 pairs per 6-tick chunk, flattened across all chunks/drives. Includes
    `drive_id` (one per pair, from the LEFT tick) so downstream drive-level
    splits (question 4) see exactly the same finite-value mask as everything
    else here.
    """
    speed = arrays["speed"]
    n_pairs = speed.shape[1] - 1
    left = np.s_[:, :-1]
    right = np.s_[:, 1:]
    dv = speed[right] - speed[left]
    out = {"dv": dv.ravel(), "speed": speed[left].ravel()}
    for field in FIELDS:
        out[field] = arrays[field][left].ravel()
    out["gas"] = arrays["gas_pedal_normalized"][left].ravel()
    out["brake"] = arrays["brake_pedal_normalized"][left].ravel()
    finite = np.all([np.isfinite(v) for v in out.values()], axis=0)
    out = {k: v[finite] for k, v in out.items()}
    out["drive_id"] = np.repeat(drive_ids, n_pairs)[finite]
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:  # noqa: PLR2004
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def question_2(pairs: dict[str, np.ndarray]) -> dict[str, Any]:
    """Which axis is longitudinal, and what sign?"""
    dv = pairs["dv"]
    dvdt = dv / DT_NOMINAL_S
    out = {}
    for field in FIELDS:
        out[field] = {
            "r_vs_dv": _corr(pairs[field], dv),
            "r_vs_dvdt": _corr(pairs[field], dvdt),
        }
    longitudinal = max(FIELDS, key=lambda f: abs(out[f]["r_vs_dv"]))
    out["longitudinal_axis"] = longitudinal
    out["sign"] = "same as dv" if out[longitudinal]["r_vs_dv"] > 0 else "opposite to dv"
    return out


def question_3(pairs: dict[str, np.ndarray], longitudinal_axis: str) -> dict[str, Any]:
    """The decisive number: v<5 & brake>0.3, where dv is provably degenerate."""
    accel = pairs[longitudinal_axis]
    nearstop_braking = (pairs["speed"] < NEARSTOP_KMH) & (
        pairs["brake"] > BRAKE_PRESSED
    )
    stationary_no_pedal = (
        (pairs["speed"] < STATIONARY_KMH)
        & (pairs["gas"] < NO_PEDAL)
        & (pairs["brake"] < NO_PEDAL)
    )

    def _bucket(mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return mask & (pairs["brake"] >= lo) & (pairs["brake"] < hi)

    low = _bucket(nearstop_braking, 0.3, 0.5)
    high = _bucket(nearstop_braking, 0.5, 1.01)

    return {
        "n_nearstop_braking": int(nearstop_braking.sum()),
        "n_stationary_no_pedal": int(stationary_no_pedal.sum()),
        "accel_vs_dv_r_in_band": _corr(
            accel[nearstop_braking], pairs["dv"][nearstop_braking]
        )
        if nearstop_braking.any()
        else None,
        "accel_vs_brake_r_in_band": _corr(
            accel[nearstop_braking], pairs["brake"][nearstop_braking]
        )
        if nearstop_braking.any()
        else None,
        "accel_mean_brake_0.3-0.5": float(accel[low].mean()) if low.any() else None,
        "accel_std_brake_0.3-0.5": float(accel[low].std()) if low.any() else None,
        "n_brake_0.3-0.5": int(low.sum()),
        "accel_mean_brake_0.5-1.0": float(accel[high].mean()) if high.any() else None,
        "accel_std_brake_0.5-1.0": float(accel[high].std()) if high.any() else None,
        "n_brake_0.5-1.0": int(high.sum()),
        "grade_contamination_mean": float(accel[stationary_no_pedal].mean())
        if stationary_no_pedal.any()
        else None,
        "grade_contamination_std": float(accel[stationary_no_pedal].std())
        if stationary_no_pedal.any()
        else None,
    }


def _drive_split(
    drive_ids: np.ndarray, *, seed: int = 0, val_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole drives, never rows (same protocol as
    `trajectory_action_controller.drive_split`)."""
    drives = np.unique(drive_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(drives)
    n_val = max(1, round(len(drives) * val_frac))
    val_drives = set(drives[:n_val].tolist())
    is_val = np.array([d in val_drives for d in drive_ids])
    return ~is_val, is_val


def question_4(pairs: dict[str, np.ndarray], longitudinal_axis: str) -> dict[str, Any]:
    """The invertibility ceiling: (acceleration, speed) -> (gas, brake), per
    SPEED_BANDS, on a held-out drive split. Design `[accel, speed, 1]`,
    separate OLS per target -- mirrors `fit_longitudinal_dynamics`'s pattern
    (`controller.py:164-173`) but inverted and un-joint (this measures the map,
    it is not the deployed controller).
    """
    train_mask, val_mask = _drive_split(pairs["drive_id"])
    accel = pairs[longitudinal_axis]
    speed = pairs["speed"]
    out: dict[str, Any] = {}
    for lo, hi in SPEED_BANDS:
        band = (speed >= lo) & (speed < hi)
        tr = band & train_mask
        va = band & val_mask
        band_key = f"{lo}-{hi}"
        if tr.sum() < 50 or va.sum() < 50:  # noqa: PLR2004
            out[band_key] = {
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "skipped": True,
            }
            continue
        design_tr = np.stack([accel[tr], speed[tr], np.ones(tr.sum())], axis=-1)
        design_va = np.stack([accel[va], speed[va], np.ones(va.sum())], axis=-1)
        band_out = {"n_train": int(tr.sum()), "n_val": int(va.sum())}
        for target_name in ("gas", "brake"):
            target = pairs[target_name]
            coeffs, *_ = np.linalg.lstsq(design_tr, target[tr], rcond=None)
            pred = design_va @ coeffs
            band_out[f"{target_name}_l1"] = float(np.abs(pred - target[va]).mean())
        out[band_key] = band_out
    return out


def run_split(
    split: str, *, config_dir: str, cache_dir: str, max_drives: int | None
) -> dict[str, Any]:
    arrays, drive_ids = _load_split(
        split=split, config_dir=config_dir, cache_dir=cache_dir, max_drives=max_drives
    )
    q1 = question_1(arrays, drive_ids)
    pairs = _tick_pairs(arrays, drive_ids)
    q2 = question_2(pairs)
    q3 = question_3(pairs, q2["longitudinal_axis"])
    q4 = question_4(pairs, q2["longitudinal_axis"])
    return {
        "q1_population": q1,
        "q2_axis": q2,
        "q3_decisive_band": q3,
        "q4_invertibility": q4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", default="/home/alex/rmind/config")
    parser.add_argument(
        "--split", nargs="+", choices=["train", "val", "predict"], default=["train"]
    )
    parser.add_argument(
        "--cache-dir", required=True, help="FRESH cache dir -- see module docstring"
    )
    parser.add_argument(
        "--max-drives", type=int, default=None, help="prototype on a subset"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    for split in args.split:
        report[split] = run_split(
            split,
            config_dir=args.config_dir,
            cache_dir=args.cache_dir,
            max_drives=args.max_drives,
        )

    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")  # noqa: T201
    else:
        print(text)  # noqa: T201


if __name__ == "__main__":
    main()
