"""Fit the FAST trajectory tokenizer's BPE on the whole training set and measure reconstruction on val.

Poses come from the trajectory-tokenizer experiment's own target (heading-cleaned, dead-reckoned
TrajectoryTarget), so they match what the policy will be trained on.

    # fit on all training trajectories, save, evaluate on val
    uv run --extra train python -m rmind.scripts.fit_fast_trajectory \
        --out artifacts/fast_trajectory --vocab-size 4096 --gammas 10 10 100

    # evaluate an already fitted tokenizer on val
    uv run --extra train python -m rmind.scripts.fit_fast_trajectory --load artifacts/fast_trajectory
"""

import argparse
import multiprocessing as mp
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from structlog import get_logger

import rmind  # noqa: F401  (registers the eval resolver)
from rmind.callbacks.loggers.trajectory_reconstruction import CATEGORIES, _categorize
from rmind.components.fast_trajectory import FASTTrajectoryTokenizer
from rmind.models.trajectory_tokenizer import TrajectoryTokenizer

logger = get_logger(__name__)

POSE_XY = 2
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def collect_poses(loader: Iterable[Any], model: TrajectoryTokenizer, limit: int | None) -> np.ndarray:
    out, n = [], 0
    with torch.no_grad():
        for batch in loader:
            poses = model._gather_poses(batch).float().numpy()  # noqa: SLF001
            out.append(poses)
            n += len(poses)
            if limit is not None and n >= limit:
                break
    return np.concatenate(out)[:limit]


def _stats(x: np.ndarray) -> dict[str, float]:
    return {"median": round(float(np.median(x)), 4), "p99": round(float(np.percentile(x, 99)), 4), "max": round(float(x.max()), 4)}


def evaluate(tok: FASTTrajectoryTokenizer, val: np.ndarray) -> None:
    ids = tok.encode(val)
    rec = tok.decode(ids)
    lens = np.array([len(i) for i in ids])
    err = np.linalg.norm(rec[..., :POSE_XY] - val[..., :POSE_XY], axis=-1)  # (N, P) position error, m
    logger.info(
        "validation",
        trajectories=len(val),
        tokens_per_trajectory={"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90)), "max": int(lens.max())},
        bpe_lossless=bool(np.array_equal(tok.coefficients(rec), tok.coefficients(val))),
        average_position_error_m=_stats(err.mean(1)),
        final_position_error_m=_stats(err[:, -1]),
        worst_position_error_m=_stats(err.max(1)),
    )
    if val.shape[-1] > POSE_XY:
        head = np.degrees(np.abs(np.angle(np.exp(1j * (rec[..., POSE_XY] - val[..., POSE_XY])))))
        logger.info("validation heading", final_heading_error_deg=_stats(head[:, -1]))
    cat = _categorize(val)
    for k, (name, _) in enumerate(CATEGORIES):
        m = cat == k
        if m.any():
            logger.info(
                "by manoeuvre",
                manoeuvre=name,
                n=int(m.sum()),
                final_position_error_m_median=round(float(np.median(err[m, -1])), 4),
                tokens_median=float(np.median(lens[m])),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="directory to save the fitted tokenizer")
    parser.add_argument("--load", type=Path, default=None, help="evaluate an already fitted tokenizer instead of fitting")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--gammas", type=float, nargs=3, default=(10.0, 10.0, 100.0), help="rounding scale for x, y (m) and theta (rad)")
    parser.add_argument("--experiment", default="yaak/trajectory_tokenizer/pretrain")
    parser.add_argument("--max-train", type=int, default=None, help="cap on training trajectories (default: all)")
    parser.add_argument("overrides", nargs="*", help="extra hydra overrides")
    args = parser.parse_args()
    if (args.out is None) == (args.load is None):
        parser.error("pass exactly one of --out (fit and save) or --load (evaluate a saved tokenizer)")

    mp.set_start_method("forkserver", force=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={args.experiment}", *args.overrides])
    model = instantiate(cfg.model)  # only its input transform (TrajectoryTarget) is used

    if args.load is None:
        t0 = time.time()
        train = collect_poses(instantiate(cfg.datamodule.train), model, args.max_train)
        logger.info("training trajectories", shape=train.shape, seconds=round(time.time() - t0))
        t0 = time.time()
        tok = FASTTrajectoryTokenizer(num_poses=train.shape[1], gammas=args.gammas).fit(train, vocab_size=args.vocab_size)
        tok.save(args.out)
        logger.info("fitted and saved", seconds=round(time.time() - t0), out=str(args.out))
    else:
        tok = FASTTrajectoryTokenizer.load(args.load)
    assert tok.value_min is not None  # noqa: S101
    assert tok.value_max is not None  # noqa: S101
    logger.info(
        "tokenizer",
        alphabet=tok.value_max - tok.value_min + 1,
        value_range=(tok.value_min, tok.value_max),
        vocab=tok.vocab_size,
        gammas=tok.gammas.tolist(),
    )

    val = collect_poses(instantiate(cfg.datamodule.val), model, None)
    evaluate(tok, val)


if __name__ == "__main__":
    main()
