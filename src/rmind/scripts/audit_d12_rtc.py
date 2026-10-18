"""Re-execute saved controller traces and check paired D12 evidence identities."""

import argparse
import json
from pathlib import Path

import numpy as np

from rmind.scripts.evaluate_d12_rtc import SEEDS, SEGMENTS, sha256, verify, write_json
from rmind.scripts.replay_d12_rtc import Controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--folder", choices=("async", "async-paced"), default="async")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.output
    protocol = verify(root)
    provenance = json.loads((root / "provenance.json").read_text())
    assert sha256(args.controller) == provenance["controller_binary"]["sha256"]
    for name, expected in provenance["source_files"].items():
        assert sha256(root / "source_snapshot" / name) == expected, name
    expected_jobs = {f"{Path(job).name}.npz" for job in protocol["expert_jobs"]}
    for model in ("baseline", "rtc"):
        actual_jobs = {path.name for path in (root / "expert" / model).glob("*.npz")}
        if actual_jobs != expected_jobs:
            raise ValueError(f"Incomplete or unexpected expert cohort: {model}")
    for path in sorted((root / "expert/baseline").glob("*.npz")):
        baseline = np.load(path)
        rtc = np.load(root / "expert/rtc" / path.name)
        for key in ("target", "height", "noise", "recording_ns", "frames", "reset"):
            np.testing.assert_array_equal(baseline[key], rtc[key])
    events = 0
    cells = 0
    plans = 0
    ticks = 0
    files = {}
    for segment in SEGMENTS:
        baseline_meta = json.loads(
            (root / "tokens/baseline" / f"{segment[1]}.json").read_text()
        )
        rtc_meta = json.loads((root / "tokens/rtc" / f"{segment[1]}.json").read_text())
        assert baseline_meta["rows"] == rtc_meta["rows"]
        assert baseline_meta["images_sha256"] == rtc_meta["images_sha256"]
        rows = baseline_meta["rows"]
        for profile in ("zero", "measured", "overrun"):
            for seed in SEEDS:
                paired_noise = None
                paired_requests = None
                for arm in ("baseline", "rtc_unconditioned", "rtc"):
                    model = "baseline" if arm == "baseline" else "rtc"
                    path = (
                        root
                        / args.folder
                        / model
                        / f"{segment[0]}-{arm}-{profile}-{seed}.json"
                    )
                    trace = json.loads(path.read_text())
                    arrays = np.load(path.with_suffix(".npz"))
                    requests = [
                        (
                            plan["row"],
                            plan["request"]["requested_ns"],
                            plan["latency_ns"],
                        )
                        for plan in trace["plans"]
                    ]
                    if paired_requests is None:
                        paired_requests = requests
                    else:
                        assert paired_requests == requests, path
                    if paired_noise is None:
                        paired_noise = arrays["noise"]
                    else:
                        np.testing.assert_array_equal(paired_noise, arrays["noise"])
                    controller = Controller(args.controller, 6 if arm == "rtc" else 0)
                    wall_indices = {
                        int(value): index
                        for index, value in enumerate(arrays["wall_ns"])
                    }
                    for event in trace["events"]:
                        actual = controller.call(event["input"])
                        assert actual == event["output"], (path, event["input"])
                        if event["input"]["op"] == "action" and "ok" in actual:
                            if args.folder == "async-paced":
                                assert actual["ok"] is not None, (
                                    "duplicate emitted tick"
                                )
                            if actual["ok"] is not None:
                                index = wall_indices[event["input"]["now_ns"]]
                                np.testing.assert_array_equal(
                                    arrays["commands"][index],
                                    np.asarray(
                                        actual["ok"]["values"], dtype=np.float32
                                    ),
                                )
                                assert (
                                    arrays["plan_ids"][index]
                                    == actual["ok"]["request_id"]
                                )
                                assert (
                                    arrays["action_indices"][index]
                                    == actual["ok"]["action_index"]
                                )
                    controller.close()
                    for plan in trace["plans"]:
                        request = plan["request"]
                        row = rows[plan["row"]]
                        assert row["wall_ns"] <= request["requested_ns"]
                        assert row["frame_wall_ns"] == request["observation_ns"]
                        assert request["forecast"] <= (6 if arm == "rtc" else 0)
                        actions = np.asarray(plan["actions"])
                        assert actions.shape == (50, 3) and np.isfinite(actions).all()
                        count = request["forecast"]
                        np.testing.assert_allclose(
                            actions[:count],
                            np.asarray(request["prefix"])[:count],
                            atol=1e-5,
                            rtol=0,
                        )
                    events += len(trace["events"])
                    plans += len(trace["plans"])
                    ticks += int(arrays["scored"].sum())
                    cells += 1
                    files[str(path.relative_to(root))] = sha256(path)
                    files[str(path.with_suffix(".npz").relative_to(root))] = sha256(
                        path.with_suffix(".npz")
                    )
    assert cells == 81
    write_json(
        args.report or root / f"{args.folder}-audit.json",
        {
            "passed": True,
            "cells": cells,
            "controller_events_reexecuted_exactly": events,
            "plans_no_future_observation_or_prefix_mismatch": plans,
            "scored_command_ticks": ticks,
            "paired_expert_noise_targets_height_frames_equal": True,
            "paired_recorded_images_equal": True,
            "controller_sha256": sha256(args.controller),
            "files": files,
        },
    )


if __name__ == "__main__":
    main()
