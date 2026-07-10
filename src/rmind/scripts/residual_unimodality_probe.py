"""Within-code residual unimodality check (offset-bug brief, Phase 1d).

Separates the offset-supervision loss bug from genuine within-cell
multimodality, which teacher forcing will *not* cure. Needs only the frozen
action tokenizer and the action-chunk dataset (no policy model, no images).

For every train-split action chunk:

    target   = tokenizer._normalize(chunk.flatten(-2, -1))     # (b, 24)
    gt_codes = tokenizer(chunk)                                # (b, G)
    residual = target - tokenizer.invert(gt_codes)             # (b, 24)

The residual is reshaped to (b, horizon, fields) and the gas / brake pedal
dims (all horizon steps pooled) are histogrammed per PRIMARY code (q=0).
A code is flagged bimodal if >5% of its residual mass lies beyond 3x the
within-code IQR from the median on either pedal dim; a smoothed-histogram
mode count is logged as a dip-test-like proxy. Top-10 codes by frequency
get matplotlib histograms logged to wandb.

Usage:
    uv run python -m rmind.scripts.residual_unimodality_probe
    uv run python -m rmind.scripts.residual_unimodality_probe \
        --max-batches 20 --no-wandb --out diag_results/residual_unimodality.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils._pytree import MappingKey, tree_leaves, tree_map  # noqa: PLC2701

from rmind.models.action_tokenizer import ActionTokenizer
from rmind.scripts.offset_diag import shutdown_dataloader
from rmind.utils.pytree import key_get_default

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from rbyte.dataloader import TorchDataNodeDataLoader
    from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_TOKENIZER_CKPT = REPO_ROOT / "artifacts" / "model-gkzgn6gk:v9" / "model.ckpt"

EXPECTED_NUM_QUANTIZERS = 4
EXPECTED_CODEBOOK_SIZE = 16

PEDAL_FIELDS = ("gas_pedal", "brake_pedal")
TAIL_IQR_MULTIPLIER = 3.0
TAIL_MASS_THRESHOLD = 0.05
MIN_CODE_COUNT = 100  # codes with fewer samples are excluded from the verdict
TOP_K_HISTOGRAMS = 10


def load_tokenizer(
    ckpt_path: str | Path = DEFAULT_TOKENIZER_CKPT, device: str | torch.device = "cuda"
) -> ActionTokenizer:
    """Load the frozen action tokenizer from a local checkpoint, eval mode.

    Raises:
        ValueError: if the quantizer geometry is not G=4, C=16.
    """
    tokenizer = ActionTokenizer.load_from_checkpoint(
        Path(ckpt_path), map_location="cpu", weights_only=False
    )
    tokenizer = (
        tokenizer.to(torch.device(device)).eval().requires_grad_(requires_grad=False)
    )

    quantizer = tokenizer.quantizer
    if (quantizer.num_quantizers, quantizer.codebook_size) != (
        EXPECTED_NUM_QUANTIZERS,
        EXPECTED_CODEBOOK_SIZE,
    ):
        msg = (
            f"unexpected quantizer geometry: G={quantizer.num_quantizers} "
            f"C={quantizer.codebook_size}, expected G={EXPECTED_NUM_QUANTIZERS} "
            f"C={EXPECTED_CODEBOOK_SIZE}"
        )
        raise ValueError(msg)

    return tokenizer


def build_dataloader(
    batch_size: int = 2048, num_workers: int = 2
) -> TorchDataNodeDataLoader[dict[str, Any]]:
    """Instantiate the action-tokenizer TRAIN dataloader (actions only).

    Composes `experiment=yaak/action_tokenizer/pretrain` (action_clip=6,
    action_step=10, action_stride=10 - the 3hz geometry the tokenizer was
    trained with). First instantiation builds the rbyte sample cache under
    `.rbyte_cache/yaak/action_train/t6d10s10` (metadata-only, NFS-bound).
    Shuffle is disabled and `drop_last` is off so a pass covers the split.
    """
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train", overrides=["experiment=yaak/action_tokenizer/pretrain"]
        )

    # make the cache path independent of the caller's working directory
    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")

    node = cfg.datamodule.train
    node.batch_size = batch_size
    node.num_workers = num_workers
    node.shuffle = False
    node.drop_last = False

    return instantiate(node)


def gather_chunk(tokenizer: ActionTokenizer, batch: dict[str, Any]) -> Tensor:
    """Extract the raw action chunk (b, horizon, fields) from a batch.

    Mirrors `ActionTokenizer._gather_actions` field ordering (pytree leaves
    of `tokenizer.targets`), but reads the RAW values through the Remapper
    only - normalization is left to `tokenizer._normalize` / `tokenizer()`.
    """
    remapped = tokenizer.input_transform[0](batch)
    gathered = tree_map(
        lambda path: key_get_default(
            remapped, tuple(MappingKey(part) for part in path), None
        ),
        tokenizer.targets,
        is_leaf=lambda x: isinstance(x, tuple),
    )
    columns = [t.float() for t in tree_leaves(gathered)]
    return torch.stack(columns, dim=-1)  # (b, horizon, fields)


def pedal_field_indices(tokenizer: ActionTokenizer) -> dict[str, int]:
    """Map pedal field name -> index in the stacked action field dim."""
    leaves = tree_leaves(tokenizer.targets, is_leaf=lambda x: isinstance(x, tuple))
    field_names = [leaf[-1] for leaf in leaves]
    return {field: field_names.index(field) for field in PEDAL_FIELDS}


def count_modes(values: np.ndarray, bins: int = 128, smooth: int = 5) -> int:
    """Dip-test-like proxy: count prominent modes of a smoothed histogram.

    A local maximum counts as a separate mode if it reaches >=5% of the peak
    density and is separated from the previous accepted mode by a valley
    below 60% of the smaller of the two peaks.
    """
    lo, hi = np.quantile(values, [0.005, 0.995])
    if hi <= lo:
        return 1
    hist, _ = np.histogram(values, bins=bins, range=(float(lo), float(hi)))
    kernel = np.ones(smooth) / smooth
    density = np.convolve(hist.astype(np.float64), kernel, mode="same")

    peak = float(density.max())
    if peak <= 0.0:
        return 1

    modes = 0
    last_mode_height = 0.0
    valley = np.inf
    for i in range(len(density)):
        left = density[i - 1] if i > 0 else -np.inf
        right = density[i + 1] if i < len(density) - 1 else -np.inf
        valley = min(valley, float(density[i]))
        is_max = density[i] >= left and density[i] > right
        if not (is_max and density[i] >= 0.05 * peak):
            continue
        height = float(density[i])
        if modes == 0 or valley < 0.6 * min(height, last_mode_height):
            modes += 1
            last_mode_height = height
            valley = np.inf
    return max(modes, 1)


def pedal_stats(values: np.ndarray) -> dict[str, float | int | bool]:
    """Robust per-(code, pedal) residual stats and the tail-mass flag."""
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0.0:
        tail_frac = float(np.mean(np.abs(values - median) > TAIL_IQR_MULTIPLIER * iqr))
        iqr_degenerate = False
    else:  # degenerate spread: any deviation from the median is "tail"
        tail_frac = float(np.mean(values != median))
        iqr_degenerate = True

    return {
        "n_values": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": iqr,
        "min": float(values.min()),
        "max": float(values.max()),
        "tail_frac_beyond_3iqr": tail_frac,
        "iqr_degenerate": iqr_degenerate,
        "n_modes_proxy": count_modes(values),
    }


def plot_code_histograms(
    code: int, residuals: dict[str, np.ndarray], stats: dict[str, Any]
) -> plt.Figure:
    """Side-by-side gas / brake residual histograms for one primary code."""
    fig, axes = plt.subplots(1, len(PEDAL_FIELDS), figsize=(11, 4))
    for ax, field in zip(axes, PEDAL_FIELDS, strict=True):
        values = residuals[field]
        field_stats = stats["pedals"][field]
        ax.hist(values, bins=100, color="tab:blue", alpha=0.85)
        ax.axvline(field_stats["median"], color="black", lw=1, label="median")
        for sign in (-1, 1):
            ax.axvline(
                field_stats["median"] + sign * TAIL_IQR_MULTIPLIER * field_stats["iqr"],
                color="red",
                lw=1,
                ls="--",
                label="median +- 3*IQR" if sign < 0 else None,
            )
        ax.set_yscale("log")
        ax.set_title(
            f"{field} | tail={field_stats['tail_frac_beyond_3iqr']:.3%} "
            f"modes={field_stats['n_modes_proxy']}"
        )
        ax.set_xlabel("residual (normalized units)")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"primary code {code} | n={stats['count']} chunks "
        f"({stats['frequency']:.2%} of split) | bimodal={stats['bimodal']}"
    )
    fig.tight_layout()
    return fig


def run_probe(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0914, PLR0915
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.ckpt, device)
    pedal_idx = pedal_field_indices(tokenizer)
    horizon = None

    loader = build_dataloader(args.batch_size, args.num_workers)
    n_samples_split = len(loader.dataset)
    print(  # noqa: T201
        f"[residual_unimodality] train action split: {n_samples_split} chunks, "
        f"{len(loader)} batches of {args.batch_size}"
    )

    num_codes = tokenizer.quantizer.codebook_size
    per_code: dict[str, list[list[np.ndarray]]] = {
        field: [[] for _ in range(num_codes)] for field in PEDAL_FIELDS
    }
    counts = np.zeros(num_codes, dtype=np.int64)
    residual_l1_sum = 0.0
    n_chunks = 0

    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break

                chunk = gather_chunk(tokenizer, batch).to(device)  # (b, h, f)
                horizon = chunk.shape[1]
                target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001
                gt_codes = tokenizer(chunk)
                residual = target - tokenizer.invert(gt_codes)  # (b, h*f)
                residual = residual.reshape(chunk.shape)  # (b, h, f)

                primary = gt_codes[:, 0].cpu().numpy()
                residual_np = residual.cpu().numpy()
                residual_l1_sum += float(np.abs(residual_np).sum())
                n_chunks += residual_np.shape[0]

                for code in np.unique(primary):
                    mask = primary == code
                    counts[code] += int(mask.sum())
                    for field, idx in pedal_idx.items():
                        per_code[field][code].append(
                            residual_np[mask][:, :, idx].reshape(-1)
                        )

                if (batch_idx + 1) % 100 == 0:
                    rate = n_chunks / (time.perf_counter() - t0)
                    print(  # noqa: T201
                        f"[residual_unimodality] batch {batch_idx + 1}/{len(loader)} "
                        f"({n_chunks} chunks, {rate:.0f} chunks/s)"
                    )
    finally:
        shutdown_dataloader(loader)

    elapsed = time.perf_counter() - t0
    print(  # noqa: T201
        f"[residual_unimodality] processed {n_chunks} chunks in {elapsed:.1f}s"
    )

    code_residuals: dict[int, dict[str, np.ndarray]] = {
        code: {
            field: (
                np.concatenate(per_code[field][code])
                if per_code[field][code]
                else np.empty(0, dtype=np.float32)
            )
            for field in PEDAL_FIELDS
        }
        for code in range(num_codes)
    }

    code_stats: dict[int, dict[str, Any]] = {}
    for code in range(num_codes):
        count = int(counts[code])
        stats: dict[str, Any] = {
            "count": count,
            "frequency": count / n_chunks if n_chunks else 0.0,
            "pedals": {},
            "bimodal": False,
        }
        if count > 0:
            stats["pedals"] = {
                field: pedal_stats(code_residuals[code][field])
                for field in PEDAL_FIELDS
            }
            stats["bimodal"] = any(
                stats["pedals"][field]["tail_frac_beyond_3iqr"] > TAIL_MASS_THRESHOLD
                for field in PEDAL_FIELDS
            )
        code_stats[code] = stats

    eligible = [c for c in range(num_codes) if counts[c] >= MIN_CODE_COUNT]
    bimodal_codes = [c for c in eligible if code_stats[c]["bimodal"]]
    bimodal_fraction = len(bimodal_codes) / len(eligible) if eligible else 0.0

    result: dict[str, Any] = {
        "tokenizer_ckpt": str(args.ckpt),
        "split": "train (action_tokenizer pretrain datamodule, 3hz t6d10s10)",
        "n_chunks_split": n_samples_split,
        "n_chunks_processed": n_chunks,
        "horizon": horizon,
        "residual_l1_mean_per_dim": (
            residual_l1_sum / (n_chunks * (horizon or 1) * tokenizer._action_features)  # noqa: SLF001
            if n_chunks
            else None
        ),
        "tail_iqr_multiplier": TAIL_IQR_MULTIPLIER,
        "tail_mass_threshold": TAIL_MASS_THRESHOLD,
        "min_code_count": MIN_CODE_COUNT,
        "eligible_codes": eligible,
        "bimodal_codes": bimodal_codes,
        "bimodal_code_fraction": bimodal_fraction,
        "verdict": (
            "bimodal" if bimodal_fraction > TAIL_MASS_THRESHOLD else "unimodal"
        ),
        "per_code": {str(c): code_stats[c] for c in range(num_codes)},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[residual_unimodality] wrote {out_path}")  # noqa: T201

    top_codes = sorted(range(num_codes), key=lambda c: -counts[c])[:TOP_K_HISTOGRAMS]
    figures = {
        code: plot_code_histograms(code, code_residuals[code], code_stats[code])
        for code in top_codes
        if counts[code] > 0
    }
    for code, fig in figures.items():
        fig.savefig(out_path.parent / f"residual_hist_code{code:02d}.png", dpi=120)

    if args.wandb:
        os.environ.setdefault("WANDB_DIR", "wandb_logs")
        Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
        import wandb  # noqa: PLC0415

        run = wandb.init(
            entity="yaak",
            project="rmind",
            name="diag-residual-unimodality",
            tags=["diag/joint_policy_offset_bug"],
            job_type="diagnostics",
            config={k: v for k, v in result.items() if k != "per_code"},
        )
        table = wandb.Table(
            columns=[
                "code",
                "count",
                "frequency",
                *(
                    f"{field}/{stat}"
                    for field in PEDAL_FIELDS
                    for stat in (
                        "median",
                        "iqr",
                        "tail_frac_beyond_3iqr",
                        "n_modes_proxy",
                    )
                ),
                "bimodal",
            ]
        )
        for code in range(num_codes):
            stats = code_stats[code]
            row: list[Any] = [code, stats["count"], stats["frequency"]]
            for field in PEDAL_FIELDS:
                pedal = stats["pedals"].get(field, {})
                row.extend(
                    pedal.get(k)
                    for k in ("median", "iqr", "tail_frac_beyond_3iqr", "n_modes_proxy")
                )
            row.append(stats["bimodal"])
            table.add_data(*row)

        run.log({
            "per_code_stats": table,
            "bimodal_code_fraction": bimodal_fraction,
            "n_chunks_processed": n_chunks,
            **{
                f"residual_hist/code{code:02d}": wandb.Image(fig)
                for code, fig in figures.items()
            },
        })
        run.summary["verdict"] = result["verdict"]
        run.summary["bimodal_codes"] = bimodal_codes
        result["wandb_url"] = run.url
        run.finish()
        print(f"[residual_unimodality] wandb: {result['wandb_url']}")  # noqa: T201
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for fig in figures.values():
        plt.close(fig)

    return result


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_TOKENIZER_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", default="diag_results/residual_unimodality.json")
    final = run_probe(parser.parse_args())
    print(  # noqa: T201
        f"[residual_unimodality] verdict={final['verdict']} "
        f"bimodal_code_fraction={final['bimodal_code_fraction']:.3f} "
        f"bimodal_codes={final['bimodal_codes']}"
    )
