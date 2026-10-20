"""Evaluate matching checkpoint steps as independent D12 training runs produce them."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

COMPARISON_VARIANTS = 5


def _require_complete(comparison: dict[str, object], output: Path) -> None:
    if len(comparison) != COMPARISON_VARIANTS:
        msg = f"incomplete prior evaluation at {output}; use a fresh output"
        raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--status", type=Path, required=True)
    arguments = parser.parse_args()
    remaining = list(arguments.run_dirs)
    state = {str(directory): "waiting_for_checkpoint" for directory in remaining}
    while remaining:
        for directory in remaining.copy():
            checkpoints = list(
                (directory / "checkpoints").glob(f"*-step={arguments.step}.ckpt")
            )
            if not checkpoints:
                continue
            if len(checkpoints) != 1:
                msg = f"ambiguous checkpoint in {directory}"
                raise ValueError(msg)
            checkpoint = checkpoints[0]
            initial = checkpoint.stat()
            time.sleep(5)
            if (initial.st_size, initial.st_mtime_ns) != (
                checkpoint.stat().st_size,
                checkpoint.stat().st_mtime_ns,
            ):
                continue
            state[str(directory)] = "evaluating"
            arguments.status.write_text(json.dumps(state, indent=2) + "\n")
            output = directory / f"research-step{arguments.step}"
            sensitivity = directory / f"sensitivity-step{arguments.step}"
            try:
                if not (output / "comparison.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "rmind.scripts.research_d12_fm",
                            "--checkpoint",
                            str(checkpoint),
                            "--config",
                            str(directory / "resolved.yaml"),
                            "--output",
                            str(output),
                            "--rerun",
                        ],
                        check=True,
                    )
                comparison = json.loads((output / "comparison.json").read_text())
                _require_complete(comparison, output)
                if not (sensitivity / "report.json").exists():
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "rmind.scripts.probe_d12_fm",
                            "--checkpoint",
                            str(checkpoint),
                            "--config",
                            str(directory / "resolved.yaml"),
                            "--output",
                            str(sensitivity),
                        ],
                        check=True,
                    )
                state[str(directory)] = "complete"
            except Exception as error:
                state[str(directory)] = f"failed: {error}"
            remaining.remove(directory)
            arguments.status.write_text(json.dumps(state, indent=2) + "\n")
        arguments.status.write_text(json.dumps(state, indent=2) + "\n")
        if remaining:
            time.sleep(60)


if __name__ == "__main__":
    main()
