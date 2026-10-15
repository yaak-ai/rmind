"""Repeatable solver-step and cross-replan noise-coupling ablations."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rmind.scripts.eval_d12_fm import DEFAULT_JOBS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS)
    parser.add_argument("--rerun", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    baseline = arguments.output / "steps4-fresh"
    for steps, noise in (
        (4, "fresh"),
        (8, "fresh"),
        (16, "fresh"),
        (4, "fixed"),
        (4, "shift"),
    ):
        name = f"steps{steps}-{noise}"
        output = arguments.output / name
        command = [
            sys.executable,
            "-m",
            "rmind.scripts.eval_d12_fm",
            "--checkpoint",
            str(arguments.checkpoint),
            "--config",
            str(arguments.config),
            "--output",
            str(output),
            "--samples",
            str(arguments.samples),
            "--seed",
            str(arguments.seed),
            "--fm-steps",
            str(steps),
            "--noise-mode",
            noise,
            "--jobs",
            *arguments.jobs,
        ]
        if output != baseline:
            command += ["--reuse-features", str(baseline), "--no-video"]
        elif arguments.rerun:
            command += ["--rerun"]
        subprocess.run(command, check=True)
        report = json.loads((output / "report.json").read_text())
        summary[name] = report["aggregate"]
        (arguments.output / "comparison.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
