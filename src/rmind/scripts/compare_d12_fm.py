"""Compare completed D12 FM runs on identical recorded contexts and noise seeds."""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from rmind.scripts.eval_d12_fm import ACTIONS, summarize

RUNS = {
    "forward_3cam": "2026-10-14-h50",
    "fork_only": "2026-10-14-h50-1cam",
    "backward_3cam": "2026-10-14-h50-backward",
    "balanced_forward_3cam": "2026-10-14-h50-balanced",
}


def assert_paired(reference, candidate):
    for key in ("target", "seconds", "reset", "height"):
        np.testing.assert_array_equal(
            reference[key], candidate[key], err_msg=f"unpaired {key}"
        )
    np.testing.assert_array_equal(reference["frames"][:, 0], candidate["frames"][:, 0])


def steering_slices(arrays):
    prediction = arrays["stream"][:, :, :6, 1]
    target = np.broadcast_to(arrays["target"][:, None, :6, 1], prediction.shape)
    result = {}
    for threshold in (0.05, 0.2, 0.4):
        active = np.abs(target) > threshold
        if active.any():
            result[str(threshold)] = {
                "mae": float(np.abs(prediction - target)[active].mean()),
                "wrong_sign_fraction": float(
                    (prediction[active] * target[active] < 0).mean()
                ),
                "mean_abs_prediction_over_mean_abs_target": float(
                    np.abs(prediction[active]).mean() / np.abs(target[active]).mean()
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    references = {}
    noise_settings = None
    result = {
        "models": {},
        "limitations": "One training seed per model. Four held-out recordings; adjacent contexts are correlated, not independent trials. Raw command errors are not task success rates. Goal/image sensitivity probes use only the original three validation recordings. Idle deadband 0.05 is not calibrated physical motion. Runtimes from local parity are single measurements, not deployment benchmarks.",
    }
    table_rows = []
    for name, directory_name in RUNS.items():
        directory = arguments.root / directory_name
        evaluation = directory / "research-step35000/steps4-fresh"
        fourth = directory / "eval-final-fourth-holdout"
        reports = [
            (evaluation, json.loads((evaluation / "report.json").read_text())),
            (fourth, json.loads((fourth / "report.json").read_text())),
        ]
        loaded = {}
        for path, report in reports:
            settings = (
                report["seed"],
                report["samples"],
                report["fm_steps"],
                report["noise_mode"],
                report["precision"],
            )
            if noise_settings is None:
                noise_settings = settings
            if settings != noise_settings:
                raise ValueError("unmatched sampling protocol")
            if report["checkpoint_sha256"] != reports[0][1]["checkpoint_sha256"]:
                raise ValueError("mixed checkpoint versions")
            for job, metadata in report["jobs"].items():
                if metadata["split"] != "val":
                    continue
                with np.load(path / job / "predictions.npz") as saved:
                    arrays = dict(saved)
                if job in references:
                    assert_paired(references[job], arrays)
                else:
                    if name != "forward_3cam":
                        raise ValueError("different validation jobs")
                    references[job] = arrays
                loaded[job] = arrays
        if set(loaded) != set(references):
            raise ValueError("missing validation jobs")
        pooled = {
            key: np.concatenate([arrays[key] for arrays in loaded.values()])
            for key in (
                "target",
                "stream",
                "window",
                "reset",
                "seconds",
                "height",
                "fm_batch_ms",
            )
        }
        metrics = summarize(pooled)
        sensitivity = json.loads(
            (directory / "sensitivity-step35000/report.json").read_text()
        )
        probe_metrics = sensitivity["metrics"]["val/all"]
        goal_sensitivity = {}
        for field in ("pickup", "dropoff"):
            for radius in (0.1, 0.5, 1.0, 2.0):
                variants = [
                    probe_metrics[key]
                    for key, case in sensitivity["cases"].items()
                    if case["field"] == field and case["radius_m"] == radius
                ]
                goal_sensitivity[f"{field}/{radius}m"] = {
                    key: float(np.mean([variant[key] for variant in variants]))
                    for key in (
                        "paired_near_abs_change",
                        "collapse_to_idle_given_baseline_active",
                        "baseline_drive_lift_abs_mean",
                        "changed_drive_lift_abs_mean",
                    )
                }
                goal_sensitivity[f"{field}/{radius}m"]["per_action"] = {
                    action: {
                        key: float(
                            np.mean([
                                variant["per_action"][action][key]
                                for variant in variants
                                if variant["per_action"][action][key] is not None
                            ])
                        )
                        if any(
                            variant["per_action"][action][key] is not None
                            for variant in variants
                        )
                        else None
                        for key in (
                            "paired_abs_change",
                            "baseline_abs_mean",
                            "changed_abs_mean",
                            "neutral_given_baseline_active",
                        )
                    }
                    for action in ACTIONS
                }
        model = {
            "run": json.loads((directory / "run.json").read_text()),
            "checkpoint_sha256": reports[0][1]["checkpoint_sha256"],
            "all_four_validation": metrics,
            "per_job": {job: summarize(arrays) for job, arrays in loaded.items()},
            "strong_steering": steering_slices(pooled),
            "original_three_train": reports[0][1]["aggregate"]["train"],
            "solver_and_noise_original_three": json.loads(
                (directory / "research-step35000/comparison.json").read_text()
            ),
            "goal_sensitivity_original_three": goal_sensitivity,
            "image_sensitivity_original_three": {
                key: value
                for key, value in probe_metrics.items()
                if key.startswith("image/")
            },
            "export_status": json.loads((directory / "completion.json").read_text()),
            "local_trt_parity": json.loads(
                (directory / "export-final/coppernic-parity.json").read_text()
            ),
        }
        result["models"][name] = model
        near = metrics["stream"]["near"]
        values = [
            name,
            f"{near['mae']:.4f}",
            *[f"{near['per_action'][action]['mae']:.4f}" for action in ACTIONS],
            *[
                f"{near['per_action'][action]['false_active_on_idle_fraction']:.1%}"
                for action in ("traction", "fork1")
            ],
        ]
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in values)
            + "</tr>"
        )
        print(" | ".join(values), flush=True)
    result["protocol"] = {
        "seed": noise_settings[0],
        "samples": noise_settings[1],
        "steps": noise_settings[2],
        "noise": noise_settings[3],
        "precision": noise_settings[4],
        "jobs": sorted(references),
    }
    (arguments.output / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    headers = (
        "Model",
        "Near MAE",
        "Traction MAE",
        "Steering MAE",
        "Fork MAE",
        "Traction false-active on idle",
        "Fork false-active on idle",
    )
    page = "<!doctype html><meta charset='utf-8'><h1>D12 FM final checkpoint comparison</h1>"
    page += "<p>35,000 optimizer steps; all four held-out recordings; eight paired Gaussian starts; four Euler steps; first six 30 Hz commands per 5 Hz replan.</p>"
    page += (
        "<table border='1' cellpadding='8'><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr>"
        + "".join(table_rows)
        + "</table>"
    )
    page += (
        "<p>"
        + html.escape(result["limitations"])
        + "</p><p><a href='comparison.json'>Complete metrics and provenance</a></p>"
    )
    (arguments.output / "report.html").write_text(page)


if __name__ == "__main__":
    main()
