"""Analyze recorded D12 policy rollouts without loading a model or touching the kit."""

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

mpl.use("Agg")
import matplotlib.pyplot as plt

TOPICS = (
    "policy/prediction",
    "policy/inputs",
    "lindelot/status",
    "lindelot/applied",
    "linde/speed",
    "linde/traction",
    "linde/fork",
    "lindelot/vehicle_state",
    "rtls/pose",
    "rtls/pallet_pose",
    "fork/height",
    "cam_fork/frame",
)
IDLE_COMMAND_THRESHOLD = 0.05
STATIONARY_SPEED_THRESHOLD = 0.15
CONSOLE_RELEASE_TOLERANCE_S = 0.35
POSE_STALENESS_THRESHOLD_S = 0.9
OPERATOR_COMMAND_THRESHOLD = 0.1


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def intervals(
    times: Sequence[float] | np.ndarray,
    enabled: Sequence[bool] | np.ndarray,
    max_gap: float = 0.5,
) -> list[tuple[float, float, str]]:
    result = []
    start = None
    for index, timestamp in enumerate(times):
        gap = index > 0 and timestamp - times[index - 1] > max_gap
        if start is not None and (not enabled[index] or gap):
            stop = times[index - 1] if gap else timestamp
            result.append((float(start), float(stop), "gap" if gap else "off"))
            start = None
        if enabled[index] and start is None:
            start = timestamp
    if start is not None:
        result.append((float(start), float(times[-1]), "end"))
    return result


def quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return (
        None
        if not len(values)
        else dict(
            zip(
                ("min", "p50", "p95", "max"),
                np.quantile(values, [0, 0.5, 0.95, 1]).tolist(),
                strict=True,
            )
        )
    )


def past(table: np.ndarray, times: Sequence[float] | np.ndarray) -> np.ndarray:
    if not len(table):
        return np.full((len(times), table.shape[1]), np.nan)
    indices = np.searchsorted(table[:, 0], times, side="right") - 1
    result = table[np.maximum(indices, 0)].copy()
    result[indices < 0] = np.nan
    return result


def pallet_goal_distance(inputs: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(inputs[:, 6:8] - inputs[:, 8:10], axis=1) * 10
    distance[(np.abs(inputs[:, 6:10]) >= 1).any(axis=1)] = np.nan
    return distance


def read_job(directory: Path) -> tuple[dict[str, np.ndarray], list[str], int]:
    tables = defaultdict(list)
    skipped = []
    files = sorted(
        directory.glob("sensor--*.mcap"),
        key=lambda path: int(path.stem.split("--")[-1]),
    )
    for path in files:
        with path.open("rb") as handle:
            try:
                reader = make_reader(handle, decoder_factories=[DecoderFactory()])
                if reader.get_summary() is None:
                    skipped.append(path.name)
                    continue
            except Exception as error:
                skipped.append(f"{path.name}: {error}")
                continue
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=TOPICS
            ):
                timestamp = message.log_time / 1e9
                topic = channel.topic
                if topic == "policy/prediction":
                    heads = {
                        item.name.split(".")[-1]: item.value for item in decoded.heads
                    }
                    commands = {item.name: item.value for item in decoded.command}
                    row = [
                        timestamp,
                        decoded.step_tov_ns,
                        decoded.engage_requested,
                        decoded.published,
                        *[
                            heads.get(name, np.nan)
                            for name in ("traction", "steering", "fork1")
                        ],
                        *[
                            commands.get(name, np.nan)
                            for name in ("traction", "steering", "fork1")
                        ],
                    ]
                elif topic == "policy/inputs":
                    aux = {item.name: list(item.values) for item in decoded.aux}
                    height = aux.get("fork_height", [np.nan] * 3)
                    row = [
                        timestamp,
                        decoded.step_tov_ns,
                        decoded.reset_generation,
                        decoded.sent_to_model,
                        len(decoded.frames),
                        aux.get("speed", [np.nan])[-1],
                        *aux.get("relative_ego_pos", [np.nan] * 2)[-2:],
                        *aux.get("relative_dropoff_pos", [np.nan] * 2)[-2:],
                        *height[-3:],
                    ]
                elif topic == "lindelot/status":
                    row = [
                        timestamp,
                        decoded.override_active,
                        decoded.yoke_disconnected,
                    ]
                elif topic == "lindelot/applied":
                    row = [
                        timestamp,
                        decoded.override_active,
                        decoded.traction_pct / 100,
                        decoded.fork1_pct / 100,
                    ]
                elif topic == "linde/speed":
                    row = [timestamp, decoded.speed_kmh]
                elif topic == "linde/traction":
                    row = [timestamp, decoded.traction_pct / 100]
                elif topic == "linde/fork":
                    row = [timestamp, decoded.fork1_pct / 100]
                elif topic == "lindelot/vehicle_state":
                    row = [timestamp, decoded.operator_on, decoded.operator_on_valid]
                elif topic in {"rtls/pose", "rtls/pallet_pose"}:
                    if not decoded.selected:
                        continue
                    row = [
                        timestamp,
                        decoded.x_m,
                        decoded.y_m,
                        decoded.heading_deg,
                        decoded.rejected,
                    ]
                elif topic == "fork/height":
                    row = [timestamp, decoded.height_m, decoded.valid, decoded.sigma_m]
                else:
                    row = [timestamp, decoded.frame_index, decoded.pts_ns]
                tables[topic].append(row)
    widths = (10, 13, 3, 4, 2, 2, 2, 3, 5, 5, 4, 3)
    result = {}
    for topic, width in zip(TOPICS, widths, strict=True):
        table = np.asarray(tables[topic], dtype=float).reshape(-1, width)
        result[topic] = table[np.argsort(table[:, 0], kind="stable")]
    return result, skipped, len(files)


def journal_events(path: Path | None) -> list[tuple[float, str]]:
    if path is None:
        return []
    events = []
    for line in path.read_text().splitlines():
        match = re.search(r"console requested (ENGAGE|RELEASE)", line)
        if match:
            precise = re.search(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z", line)
            timestamp = precise.group() if precise else line.split()[0]
            events.append((
                datetime.fromisoformat(timestamp).timestamp(),
                match.group(1),
            ))
    return events


def analyze(
    directory: Path, output: Path, events: list[tuple[float, str]]
) -> dict[str, Any]:
    tables, skipped, files = read_job(directory)
    nonempty = [table for table in tables.values() if len(table)]
    if not nonempty:
        return {"job": directory.name, "files": files, "skipped": skipped}
    start = min(table[0, 0] for table in nonempty)
    stop = max(table[-1, 0] for table in nonempty)
    predictions = tables["policy/prediction"]
    inputs = tables["policy/inputs"]
    status = tables["lindelot/status"]
    applied = tables["lindelot/applied"]
    speed = tables["linde/speed"]
    pallet = tables["rtls/pallet_pose"]
    height = tables["fork/height"]
    segments = intervals(status[:, 0], status[:, 1] > 0)
    model_steps = inputs[inputs[:, 3] > 0]
    report: dict[str, Any] = {
        "job": directory.name,
        "start_utc": utc(start),
        "end_utc": utc(stop),
        "duration_s": stop - start,
        "files": files,
        "skipped": skipped,
        "input_frame_counts": sorted(set(inputs[:, 4].tolist())),
        "prediction_count": len(predictions),
        "model_steps": len(model_steps),
        "console_engages": int(
            sum(
                start <= timestamp <= stop and event == "ENGAGE"
                for timestamp, event in events
            )
        ),
        "override_segments": len(segments),
        "override_seconds": sum(end - begin for begin, end, _ in segments),
        "override_duration_s": quantiles([end - begin for begin, end, _ in segments]),
        "camera_period_ms": quantiles(np.diff(tables["cam_fork/frame"][:, 0]) * 1000),
        "model_input_period_ms": quantiles(np.diff(model_steps[:, 0]) * 1000),
        "pallet_gap_s": quantiles(np.diff(pallet[:, 0])),
        "pallet_gaps_over_1s": int((np.diff(pallet[:, 0]) > 1).sum()),
    }
    step_times = {int(row[1]): row[0] for row in model_steps}
    first_outputs = {}
    for row in predictions:
        first_outputs.setdefault(int(row[1]), row[0])
    report["first_output_latency_ms"] = quantiles([
        (timestamp - step_times[step]) * 1000
        for step, timestamp in first_outputs.items()
        if step in step_times
    ])
    if len(inputs):
        distances = pallet_goal_distance(inputs)
        final = distances[inputs[:, 0] >= inputs[-1, 0] - 5]
        report["final_pallet_goal_error_m"] = quantiles(final)
        report["pallet_goal_error_note"] = (
            "Distance between pallet and dropoff tokens in the same ego frame; clipped tokens excluded; UWB noise and tag/target offsets remain."
        )
    if len(predictions):
        active = past(status, predictions[:, 0])[:, 1] > 0
        requested = predictions[:, 2] > 0
        report["actions"] = {}
        for name, mask in (
            ("requested", requested),
            ("override", active),
            ("released", ~requested),
        ):
            values = predictions[mask, 4:7]
            report["actions"][name] = {"samples": len(values)}
            if not len(values):
                continue
            report["actions"][name].update({
                "all_idle_fraction": float(
                    (np.abs(values) < IDLE_COMMAND_THRESHOLD).all(axis=1).mean()
                )
            })
            for index, action in enumerate(("traction", "steering", "fork1")):
                report["actions"][name][action] = {
                    "distribution": quantiles(values[:, index]),
                    "idle_fraction": float(
                        (np.abs(values[:, index]) < IDLE_COMMAND_THRESHOLD).mean()
                    ),
                    "positive_fraction": float(
                        (values[:, index] > IDLE_COMMAND_THRESHOLD).mean()
                    ),
                    "negative_fraction": float(
                        (values[:, index] < -IDLE_COMMAND_THRESHOLD).mean()
                    ),
                    "over_range_fraction": float((np.abs(values[:, index]) > 1).mean()),
                }
    active_speed = past(status, speed[:, 0])[:, 1] > 0
    report["override_speed_native"] = quantiles(np.abs(speed[active_speed, 1]))
    report["override_speed_zero_fraction"] = (
        float((np.abs(speed[active_speed, 1]) < STATIONARY_SPEED_THRESHOLD).mean())
        if active_speed.any()
        else None
    )
    report["actuator_tracking"] = {}
    active_applied = applied[applied[:, 1] > 0]
    if len(active_applied) and len(predictions):
        for action, column, pred_column in (("traction", 2, 7), ("fork1", 3, 9)):
            errors = []
            for lag in np.arange(0, 0.251, 0.01):
                reference = past(predictions, active_applied[:, 0] - lag)
                valid = (reference[:, 3] > 0) & (
                    active_applied[:, 0] - reference[:, 0] < 0.4 + lag
                )
                errors.append(
                    float(
                        np.abs(
                            active_applied[valid, column]
                            - np.clip(reference[valid, pred_column], -1, 1)
                        ).mean()
                    )
                    if valid.any()
                    else float("inf")
                )
            best = int(np.argmin(errors))
            report["actuator_tracking"][action] = {
                "best_lag_ms": best * 10,
                "mae_normalized": errors[best],
            }
    records = []
    for index, (begin, end, boundary) in enumerate(segments):
        latest = {topic: past(table, [end])[0] for topic, table in tables.items()}
        pallet_age = end - latest["rtls/pallet_pose"][0]
        pose_age = end - latest["rtls/pose"][0]
        console_release = any(
            event == "RELEASE" and abs(timestamp - end) < CONSOLE_RELEASE_TOLERANCE_S
            for timestamp, event in events
        )
        if boundary != "off":
            reason = "recording_boundary_or_status_gap"
        elif console_release:
            reason = "console_release"
        elif pallet_age >= POSE_STALENESS_THRESHOLD_S:
            reason = "pallet_stale_aligned"
        elif pose_age >= POSE_STALENESS_THRESHOLD_S or latest["rtls/pose"][4] > 0:
            reason = "pose_stale_or_rejected_aligned"
        elif (
            latest["lindelot/vehicle_state"][2] > 0
            and latest["lindelot/vehicle_state"][1] == 0
        ):
            reason = "driver_absent_snapshot"
        elif (
            abs(latest["linde/traction"][1]) > OPERATOR_COMMAND_THRESHOLD
            or abs(latest["linde/fork"][1]) > OPERATOR_COMMAND_THRESHOLD
        ):
            reason = "operator_input_aligned"
        else:
            reason = "unresolved"
        selected = inputs[(inputs[:, 0] >= begin) & (inputs[:, 0] <= end)]
        selected_speed = speed[(speed[:, 0] >= begin) & (speed[:, 0] <= end)]
        endpoints = past(height, [begin, end])
        record = {
            "segment": index + 1,
            "start_utc": utc(begin),
            "end_utc": utc(end),
            "start_s": begin - start,
            "end_s": end - start,
            "duration_s": end - begin,
            "release_evidence": reason,
            "pallet_age_at_release_s": float(pallet_age),
            "pose_age_at_release_s": float(pose_age),
            "height_start_m": float(endpoints[0, 1]),
            "height_end_m": float(endpoints[1, 1]),
            "height_valid_start": bool(endpoints[0, 2] == 1),
            "height_valid_end": bool(endpoints[1, 2] == 1),
            "speed_abs_max_native": float(np.abs(selected_speed[:, 1]).max())
            if len(selected_speed)
            else None,
        }
        selected_predictions = predictions[
            (predictions[:, 0] >= begin) & (predictions[:, 0] <= end)
        ]
        if len(selected_predictions):
            record["all_idle_fraction"] = float(
                (np.abs(selected_predictions[:, 4:7]) < IDLE_COMMAND_THRESHOLD)
                .all(axis=1)
                .mean()
            )
            for column, action in enumerate(("traction", "steering", "fork1"), 4):
                record[f"{action}_median"] = float(
                    np.median(selected_predictions[:, column])
                )
                record[f"{action}_abs_p95"] = float(
                    np.quantile(np.abs(selected_predictions[:, column]), 0.95)
                )
        if len(selected_speed):
            record["speed_zero_fraction"] = float(
                (np.abs(selected_speed[:, 1]) < STATIONARY_SPEED_THRESHOLD).mean()
            )
        for name, columns in (("pickup", (6, 8)), ("dropoff", (8, 10))):
            if len(selected):
                distance = (
                    np.linalg.norm(selected[:, columns[0] : columns[1]], axis=1) * 10
                )
                record[f"{name}_distance_start_m"] = float(np.median(distance[:3]))
                record[f"{name}_distance_end_m"] = float(np.median(distance[-3:]))
        if len(selected):
            distances = pallet_goal_distance(selected)
            finite_start = distances[:3][np.isfinite(distances[:3])]
            finite_end = distances[-3:][np.isfinite(distances[-3:])]
            record["pallet_goal_error_start_m"] = (
                float(np.median(finite_start)) if len(finite_start) else None
            )
            record["pallet_goal_error_end_m"] = (
                float(np.median(finite_end)) if len(finite_end) else None
            )
        records.append(record)
    report["segments"] = records
    output.mkdir(parents=True, exist_ok=True)
    if records:
        with (output / "segments.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({key for record in records for key in record})
            )
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(
        output / "signals.npz",
        **cast(
            "dict[str, Any]",
            {name.replace("/", "__"): table for name, table in tables.items()},
        ),
    )
    figure, axes = plt.subplots(6, 1, figsize=(18, 15), sharex=True)
    for index, action in enumerate(("traction", "steering", "fork1")):
        axes[index].plot(
            predictions[:, 0] - start,
            predictions[:, 4 + index],
            lw=0.5,
            label=f"predicted {action}",
        )
        if index != 1:
            axes[index].plot(
                applied[:, 0] - start,
                applied[:, 2 if index == 0 else 3],
                lw=0.5,
                alpha=0.7,
                label="applied (includes manual)",
            )
        axes[index].set_ylim(-1.2, 1.2)
        axes[index].legend(loc="upper right")
    axes[3].plot(speed[:, 0] - start, speed[:, 1], label="speed (native, nominal km/h)")
    axes[3].legend(loc="upper right")
    axes[4].plot(height[:, 0] - start, height[:, 1], label="estimated fork height (m)")
    axes[4].legend(loc="upper right")
    grid = np.arange(start, stop, 0.1)
    axes[5].plot(grid - start, grid - past(pallet, grid)[:, 0], label="pallet age (s)")
    axes[5].axhline(1, color="red", ls="--", label="freshness limit")
    axes[5].set_ylim(0, 3)
    axes[5].legend(loc="upper right")
    for axis in axes:
        for begin, end, _ in segments:
            axis.axvspan(begin - start, end - start, color="green", alpha=0.15)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(
        "Seconds since first recorded message; green = MCU override active"
    )
    figure.suptitle(
        directory.name + " UTC — actual rollout, not open-loop imitation error"
    )
    figure.tight_layout()
    figure.savefig(output / "timeline.png", dpi=120)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, nargs="+", required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    events = journal_events(arguments.journal)
    reports = [
        analyze(job, arguments.output / job.name, events) for job in arguments.jobs
    ]
    limitations = [
        "Session/model identity must be checked against deployment evidence; engine hash is not in policy messages.",
        "Override-active comes from MCU telemetry; source identity is inferred from published policy and deployment, not encoded in that telemetry.",
        "Release categories are timestamp-aligned evidence, not recorded reason codes; vehicle-state snapshots are change-driven.",
        "Only emitted predictions are logged; complete sampled chunks and alternative modes are unavailable.",
        "Steering ADC is the operator yoke, not wheel-angle feedback; do not call it policy tracking error.",
        "Native speed is labelled km/h by the decoder but absolute units are not independently calibrated.",
        "Height is an estimator, UWB is noisy, goal distances are token-derived and possibly clamped; no automatic task-success label.",
        "Only complete local MCAP snapshots are analyzed; live final chunks may be skipped.",
    ]
    (arguments.output / "report.json").write_text(
        json.dumps(
            {"jobs": reports, "limitations": limitations}, indent=2, allow_nan=False
        )
        + "\n"
    )
    sections = [
        "<!doctype html><meta charset='utf-8'><title>D12 rollout analysis</title>",
        "<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{max-width:100%}td,th{padding:8px;text-align:left}</style>",
        "<h1>D12 recorded rollout analysis</h1><p>Times are UTC. Green timeline regions indicate MCU override active, not automatically independent attempts or successes.</p>",
        "<p><a href='report.json'>Machine-readable report</a></p>",
        "<table><tr><th>Session</th><th>Override bouts</th><th>Override seconds</th><th>Longest bout</th><th>Final pallet-goal error (median m)</th></tr>",
    ]
    for report in reports:
        if "override_segments" not in report:
            continue
        error = report.get("final_pallet_goal_error_m")
        sections.append(
            f"<tr><td>{html.escape(report['job'])}</td><td>{report['override_segments']}</td>"
            f"<td>{report['override_seconds']:.1f}</td><td>{report['override_duration_s']['max'] if report['override_duration_s'] else 0:.1f}</td>"
            f"<td>{error['p50'] if error else 'unknown'}</td></tr>"
        )
    sections.append("</table><h2>Limitations</h2><ul>")
    sections.extend(f"<li>{html.escape(item)}</li>" for item in limitations)
    sections.append("</ul>")
    for report in reports:
        name = html.escape(report["job"])
        sections.append(
            f"<h2>{name}</h2><a href='{name}/segments.csv'>Per-bout CSV</a> | <a href='{name}/signals.npz'>Extracted signals</a>"
        )
        sections.extend(
            f"<p><img src='{name}/{filename}' alt='{name} {filename}'></p>"
            for filename in (
                "timeline.png",
                "camera_samples.jpg",
                "delivery_samples.jpg",
            )
            if (arguments.output / report["job"] / filename).exists()
        )
    (arguments.output / "report.html").write_text("\n".join(sections))
    for report in reports:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in {"segments", "actions"}
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
