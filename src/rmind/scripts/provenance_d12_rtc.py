"""Archive the exact dirty source and verify immutable D12 evaluation inputs."""

import argparse
import importlib.metadata
import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rmind.scripts.evaluate_d12_rtc import MODELS, sha256, verify, write_json
from rmind.scripts.train_d12_rtc import resolved_executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--controller-binary", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify(args.output)
    git = resolved_executable("git")
    nvidia_smi = resolved_executable("nvidia-smi")
    root = Path(__file__).resolve().parents[3]
    paths = subprocess.check_output(
        [git, "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    source = [root / name for name in sorted(set(paths)) if (root / name).is_file()]
    archive = args.output / "source.tar.gz"
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as stream:
        for path in source:
            stream.add(path, arcname=str(path.relative_to(root)))
        stream.add(args.controller_source, arcname="external/coppernic/d12_rtc.rs")
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(args.output / "source_snapshot", filter="data")
    original_manifest = MODELS["rtc"] / "inputs.json"
    expected = json.loads(original_manifest.read_text())["dataset"]

    def checked(record: dict[str, Any]) -> dict[str, Any]:
        path = Path(record["path"])
        actual = sha256(path)
        if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
            msg = f"Training/evaluation data drift: {path}"
            raise ValueError(msg)
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        dataset = list(pool.map(checked, expected))
    write_json(
        args.output / "provenance.json",
        {
            "captured_utc": datetime.now(UTC).isoformat(),
            "scope": "Source captured after expert/extraction stages; asynchronous replay executes this snapshot. This record describes this reproduction only; original historical failed attempts remain in their separate immutable evidence directory.",
            "source_archive": {"path": str(archive), "sha256": sha256(archive)},
            "source_files": {
                str(path.relative_to(root)): sha256(path) for path in source
            },
            "controller_source": {
                "path": str(args.controller_source),
                "sha256": sha256(args.controller_source),
            },
            "controller_binary": {
                "path": str(args.controller_binary),
                "sha256": sha256(args.controller_binary),
            },
            "dataset": dataset,
            "protocol_sha256": sha256(args.output / "protocol.json"),
            "models": protocol["models"],
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "numpy",
                    "pytorch-lightning",
                    "omegaconf",
                    "polars",
                    "matplotlib",
                )
            },
            "git_head": subprocess.check_output(
                [git, "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_status": subprocess.check_output(
                [git, "status", "--short"], cwd=root, text=True
            ),
            "gpu_inventory": subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            "latency_caveat": "Historical input-to-first-output trace from OLD deployed model, not newly measured nativeRTC Orin latency; no hardware-time equivalence claimed.",
            "scheduler_caveat": "Both policies use the same actual RTC controller; baseline is an unconditioned head in that controller, not a replay of the legacy player implementation.",
            "strata_caveat": "Supported historicallabels are two stationary/no-acceleration intervals and one successful session. No grounded missed-pocket or dropoff-event interval annotations are available; expert hold/lift/lower are command-defined, not semantic outcome labels.",
        },
    )


if __name__ == "__main__":
    main()
