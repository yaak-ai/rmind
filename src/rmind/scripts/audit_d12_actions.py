"""Measure source sampling clocks and raw-to-training steering sign changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from rbyte.samples.mcap import McapReader, ProtobufDecoderFactory

from rmind.scripts.eval_d12_fm import DEFAULT_JOBS


def nearest_offsets(reference, source):
    reference = np.asarray(reference, dtype=np.int64)
    source = np.asarray(source, dtype=np.int64)
    if not len(reference) or not len(source):
        raise ValueError("empty clock")
    following = np.searchsorted(source, reference).clip(0, len(source) - 1)
    preceding = (following - 1).clip(0, len(source) - 1)
    choice = np.where(
        np.abs(source[preceding] - reference) <= np.abs(source[following] - reference),
        preceding,
        following,
    )
    return source[choice] - reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/nasa/data/d12/yaak"))
    parser.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    fields = {
        "cam_fork/frame": {"log_time": pl.Datetime("ns"), "pts_ns": pl.Int64()},
        "lindelot/adc": {
            "log_time": pl.Datetime("ns"),
            "steering_angle_normalised": pl.Float32(),
        },
        "lindelot/vehicle_state": {
            "log_time": pl.Datetime("ns"),
            "traction_command_pct": pl.Float32(),
        },
        "linde/fork": {"log_time": pl.Datetime("ns"), "fork1_pct": pl.Float32()},
    }
    reader = McapReader(decoder_factories=[ProtobufDecoderFactory], fields=fields)
    results = {}
    for job_name in arguments.jobs:
        job = arguments.data_root / job_name
        tables = reader(job / "data_for_training.mcap")
        camera = (
            tables["cam_fork/frame"]
            .sort("log_time")["log_time"]
            .cast(pl.Int64)
            .to_numpy()
        )
        result = {"signals": {}}
        for topic in fields:
            if topic == "cam_fork/frame":
                continue
            source = (
                tables[topic].sort("log_time")["log_time"].cast(pl.Int64).to_numpy()
            )
            offsets = nearest_offsets(camera, source) / 1e6
            result["signals"][topic] = {
                "messages": len(source),
                "period_ms_p50_p95": np.quantile(
                    np.diff(source) / 1e6, [0.5, 0.95]
                ).tolist(),
                "nearest_offset_ms_p05_p50_p95": np.quantile(
                    offsets, [0.05, 0.5, 0.95]
                ).tolist(),
                "nearest_abs_offset_ms_p95": float(np.quantile(np.abs(offsets), 0.95)),
                "future_nearest_fraction": float((offsets > 0).mean()),
            }
        raw_reader = McapReader(
            decoder_factories=[ProtobufDecoderFactory],
            fields={"lindelot/adc": fields["lindelot/adc"]},
        )
        raw = raw_reader(job / "data.mcap")["lindelot/adc"]
        matched = tables["lindelot/adc"].join(raw, on="log_time", suffix="_raw")
        processed = matched["steering_angle_normalised"].to_numpy()
        original = matched["steering_angle_normalised_raw"].to_numpy()
        result["steering_raw_vs_training"] = (
            {
                "matched_messages": len(matched),
                "same_sign_mae": float(np.abs(processed - original).mean()),
                "inverted_sign_mae": float(np.abs(processed + original).mean()),
            }
            if len(matched)
            else {"matched_messages": 0}
        )
        results[job_name] = result
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(
            {
                "jobs": results,
                "limitations": "Clock alignment and preprocessing consistency only; does not establish physical steering calibration, actuator delay, or camera encoder/publisher pairing correctness. Nearest joins can select a future state; causal state-history experiments must instead use latest-past values and validity masks.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
