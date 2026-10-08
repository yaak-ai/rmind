"""Bimanual (nero-bimanual-26) training helpers: take ids in val batches, arm selection.

The task the 2026-10-07 corpus records is "pick up the cube from the left or right
pickup mark with the corresponding hand": most takes move ONE arm, chosen by where
the cube is, so the policy has to learn arm selection from the cameras. The
ARM-SELECTION metric checks that on val, per active-arm class (left / right /
both, from the shared split JSON's `takes[*].class`).

`side_excursion` / `arm_selection` are a PORT of nutron-cli's
`runtime/training/nutron_act/metrics.py` (same names, constants and keys), so the
ACT and patch families report the same numbers: per frame and side, the
excursion is the largest arm-joint move from the anchor (the frame's measured q)
over the EXECUTED horizon -- the first `EXEC_STEPS` = 50 real chunk steps, ACT's
`n_action_steps` -- for the prediction and for the demonstration.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from rmind.data.nero_robot import flat_rbyte_batch

__all__ = [
    "ARM_DECISIVE_FACTOR",
    "ARM_MOVE_RAD",
    "EXEC_STEPS",
    "INPUT_ID",
    "arm_selection",
    "flat_rbyte_batch_with_ids",
    "side_excursion",
]

#: a side "moves" over the executed horizon when its largest joint excursion from
#: the anchor exceeds this (rad); a frame is decisive for a one-arm class when the
#: demo's active arm moves AND moves more than ARM_DECISIVE_FACTOR x the idle arm
ARM_MOVE_RAD: Final = 0.05
ARM_DECISIVE_FACTOR: Final = 2.0
#: the executed horizon (ACT n_action_steps 50 of a 100-step chunk)
EXEC_STEPS: Final = 50
#: batch key carrying each sample's take id (val loaders of the bimanual datamodules)
INPUT_ID: Final = "input_id"


def flat_rbyte_batch_with_ids(batch: Any) -> dict[str, Any]:
    """`flat_rbyte_batch` plus `input_id`: the take directory of every sample (list[str])."""
    out = flat_rbyte_batch(batch)
    meta = batch.meta if hasattr(batch, "meta") else batch.get("meta")
    if meta is None:
        return out
    # rbyte's BatchMeta is a tensorclass (attribute access, NonTensorStack values)
    ids = getattr(meta, INPUT_ID, None)
    if ids is None:
        ids = meta[INPUT_ID]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    out[INPUT_ID] = [str(v) for v in ids]
    return out


def side_excursion(
    x_abs: np.ndarray, state: np.ndarray, valid: np.ndarray, arm: slice
) -> np.ndarray:
    """(B,) largest |x[k, j] - state[j]| over the valid steps k and the arm joints j.

    In rad; nan for a frame with no valid step. `x_abs` (B, T, D) absolute,
    `state` (B, D) the anchor, `valid` (B, T) bool.
    """
    d = np.abs(x_abs[..., arm] - state[:, None, arm]).max(axis=-1)  # (B, T)
    d = np.where(valid, d, -np.inf).max(axis=1)
    return np.where(np.isfinite(d), d, np.nan)


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def arm_selection(  # noqa: PLR0913, PLR0914, PLR0917, C901
    prefix: str,
    sides: tuple[str, ...],
    exc_pred: np.ndarray,
    exc_demo: np.ndarray,
    take: np.ndarray,
    cls: np.ndarray,
) -> dict[str, float]:
    """The ARM-SELECTION metric, per active-arm class (left / right / both).

    `exc_pred` / `exc_demo` are (N, S) `side_excursion`s per frame, `take` and
    `cls` (N,) the frame's take id and its take's class. Per class `c`:

    * `frames` / `takes`; `pred_exc_<side>_rad` / `demo_exc_<side>_rad`
      (mean excursion per side).
    * one-arm classes (`c` is a side; the other side is idle):
      `active_exc_{pred,demo}_rad`, `idle_exc_{pred,demo}_rad` and
      `idle_exc_ratio` = predicted / demonstrated idle-arm excursion (1 = moves
      the idle arm as little as the operator did; >> 1 = drives the wrong arm too);
      `correct_side_rate` = on DECISIVE frames (demo active arm moves >
      ARM_MOVE_RAD and > ARM_DECISIVE_FACTOR x the idle arm) the fraction where the
      prediction moves the active arm more than the idle one; `take_correct_rate`
      = fraction of the class's takes whose mean predicted excursion is larger on
      the active arm.
    * `both`: `both_move_rate` = on frames where the demo moves both arms
      (> ARM_MOVE_RAD each), the fraction where the prediction does too.

    Pooled over the one-arm classes: `arm_select/correct_side_rate`,
    `arm_select/take_correct_rate`, `arm_select/idle_exc_ratio`.
    """
    out: dict[str, float] = {}
    p = f"{prefix}arm_select/"
    si = {s: i for i, s in enumerate(sides)}
    pooled_correct, pooled_takes = [], []
    pooled_idle_pred, pooled_idle_demo = [], []
    for c in ("left", "right", "both"):
        m = cls == c
        if not m.any():
            continue
        q = f"{p}{c}/"
        out[q + "frames"] = float(m.sum())
        out[q + "takes"] = float(len(set(take[m].tolist())))
        for s, i in si.items():
            out[q + f"pred_exc_{s}_rad"] = _nanmean(exc_pred[m, i])
            out[q + f"demo_exc_{s}_rad"] = _nanmean(exc_demo[m, i])
        if c in si and len(sides) == 2:  # noqa: PLR2004
            a, i = si[c], 1 - si[c]
            out[q + "active_exc_pred_rad"] = _nanmean(exc_pred[m, a])
            out[q + "active_exc_demo_rad"] = _nanmean(exc_demo[m, a])
            out[q + "idle_exc_pred_rad"] = _nanmean(exc_pred[m, i])
            out[q + "idle_exc_demo_rad"] = _nanmean(exc_demo[m, i])
            out[q + "idle_exc_ratio"] = out[q + "idle_exc_pred_rad"] / max(
                out[q + "idle_exc_demo_rad"], 1e-6
            )
            pooled_idle_pred.append(exc_pred[m, i])
            pooled_idle_demo.append(exc_demo[m, i])
            with np.errstate(invalid="ignore"):
                decisive = (
                    m
                    & (exc_demo[:, a] > ARM_MOVE_RAD)
                    & (exc_demo[:, a] > ARM_DECISIVE_FACTOR * exc_demo[:, i])
                )
                correct = exc_pred[:, a] > exc_pred[:, i]
            out[q + "decisive_frames"] = float(decisive.sum())
            if decisive.any():
                out[q + "correct_side_rate"] = float(correct[decisive].mean())
                pooled_correct.append(correct[decisive])
            votes = []
            for t in sorted(set(take[m].tolist())):
                mt = m & (take == t)
                votes.append(_nanmean(exc_pred[mt, a]) > _nanmean(exc_pred[mt, i]))
            out[q + "take_correct_rate"] = float(np.mean(votes))
            pooled_takes.extend(votes)
        elif c == "both":
            with np.errstate(invalid="ignore"):
                decisive = m & (np.nanmin(exc_demo, axis=1) > ARM_MOVE_RAD)
                moved = np.nanmin(exc_pred, axis=1) > ARM_MOVE_RAD
            out[q + "decisive_frames"] = float(decisive.sum())
            if decisive.any():
                out[q + "both_move_rate"] = float(moved[decisive].mean())
    if pooled_correct:
        out[p + "correct_side_rate"] = float(np.concatenate(pooled_correct).mean())
    if pooled_takes:
        out[p + "take_correct_rate"] = float(np.mean(pooled_takes))
    if pooled_idle_pred:
        idle_demo = _nanmean(np.concatenate(pooled_idle_demo))
        out[p + "idle_exc_ratio"] = _nanmean(np.concatenate(pooled_idle_pred)) / max(
            idle_demo, 1e-6
        )
    return out
