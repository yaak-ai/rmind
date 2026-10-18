"""Compare early versus discarded-suffix traction in explicitly selected replay intervals."""

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def traction_statistics(chunks, threshold, duration):
    traction = np.asarray(chunks)[..., 0].reshape(-1, chunks.shape[-2])
    suffix = traction[:, 8:]
    windows = np.lib.stride_tricks.sliding_window_view(suffix, duration, axis=-1)
    sustained = ((windows.min(-1) > threshold) | (windows.max(-1) < -threshold)).any(-1)
    return {
        "early_mean_absolute": float(np.abs(traction[:, 1:8]).mean()),
        "suffix_mean_absolute": float(np.abs(suffix).mean()),
        "suffix_fraction_above_threshold": float((np.abs(suffix) > threshold).mean()),
        "fraction_chunks_sustained_suffix": float(sustained.mean()),
        "fraction_chunks_entirely_below_threshold": float(
            (np.abs(traction).max(-1) < threshold).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "REPLAY_DIR", "START_UTC", "END_UTC"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--duration", type=int, default=15)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(arguments.segment),
        2,
        figsize=(13, 3.8 * len(arguments.segment)),
        squeeze=False,
    )
    results = []
    for index, (name, directory, start, end) in enumerate(arguments.segment):
        directory = Path(directory)
        recording = np.load(directory / "chunks.npz")
        report = json.loads((directory / "report.json").read_text())
        start_ns = datetime.fromisoformat(start).timestamp() * 1e9
        end_ns = datetime.fromisoformat(end).timestamp() * 1e9
        selected = (recording["wall_ns"] >= start_ns) & (recording["wall_ns"] < end_ns)
        if not selected.any():
            msg = f"No inputs in {name}"
            raise ValueError(msg)
        actions, variants = (
            recording["actions"][selected],
            recording["variants"][selected],
        )
        matches = [
            match
            for match in report["matches"]
            if start_ns <= match["wall_ns"] < end_ns
        ]
        residuals = np.asarray([
            next(
                candidate["mae"]
                for candidate in match["candidates"]
                if candidate["slot_shift"] == 0
            )
            for match in matches
        ])
        result = {
            "name": name,
            "source": str(directory),
            "start": start,
            "end": end,
            "inputs": int(selected.sum()),
            "candidate": traction_statistics(
                actions, arguments.threshold, arguments.duration
            ),
            "multi_seed": traction_statistics(
                variants, arguments.threshold, arguments.duration
            ),
            "emitted_action_match_mae_mean": float(residuals.mean()),
            "emitted_action_match_mae_p95": float(np.quantile(residuals, 0.95)),
            "samples_per_context": variants.shape[1],
        }
        results.append(result)
        time = np.arange(actions.shape[1]) / 30
        axes[index, 0].plot(
            time,
            np.abs(actions[..., 0]).mean(0),
            label="Candidate reconstructed chunks",
        )
        axes[index, 0].plot(
            time,
            np.abs(variants[..., 0]).mean((0, 1)),
            label="Fresh-noise distribution",
        )
        axes[index, 0].axvspan(
            1 / 30, 8 / 30, alpha=0.12, color="green", label="Typical executed region"
        )
        axes[index, 0].axhline(
            arguments.threshold,
            linestyle=":",
            color="gray",
            label="Diagnostic threshold (not deadband)",
        )
        axes[index, 0].set(
            title=f"{name}: mean absolute traction",
            xlabel="Action time from image (s)",
            ylabel="Normalized |traction|",
            ylim=(0, 0.5),
        )
        axes[index, 0].legend(fontsize=7)
        representative = len(actions) // 2
        for trajectory in variants[representative]:
            axes[index, 1].plot(
                time, trajectory[:, 0], alpha=0.3, linewidth=0.8, color="tab:blue"
            )
        axes[index, 1].plot(
            time,
            actions[representative, :, 0],
            color="black",
            linewidth=2,
            label="Candidate recorded-noise chunk",
        )
        axes[index, 1].axvspan(1 / 30, 8 / 30, alpha=0.12, color="green")
        axes[index, 1].set(
            title="Middle context: individual sampled trajectories",
            xlabel="Action time from image (s)",
            ylabel="Normalized traction",
            ylim=(-1, 1),
        )
        axes[index, 1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(arguments.output / "traction_horizons.png", dpi=150)
    plt.close(figure)
    payload = {
        "interpretation": "Approximate input replay; no physical counterfactual rollout or exact historical parity claim.",
        "early_indices": [1, 8],
        "suffix_indices": [8, 50],
        "threshold": arguments.threshold,
        "sustained_samples": arguments.duration,
        "threshold_caveat": "Illustrative normalized-command threshold, not a calibrated traction/acceleration deadband.",
        "segments": results,
    }
    (arguments.output / "summary.json").write_text(json.dumps(payload, indent=2))
    (arguments.output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>D12 historical chunk replay</title>'
        "<h1>D12 historical chunk replay — approximate, not exact</h1>"
        "<p>Recorded compressed video and recovered candidate RNG alignment. "
        "Fresh-noise variants are distributional probes, not historical samples. "
        "The 15% command threshold is diagnostic, not a measured actuator deadband.</p>"
        '<img src="traction_horizons.png" style="max-width:100%"><pre>'
        + html.escape(json.dumps(payload, indent=2))
        + "</pre>"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
