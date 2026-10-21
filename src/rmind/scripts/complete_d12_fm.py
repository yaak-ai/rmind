"""Wait for training, then export and validate a local TensorRT artifact."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        msg = f"{name} executable not found"
        raise RuntimeError(msg)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--coppernic", type=Path, required=True)
    parser.add_argument("--validation-host")
    parser.add_argument("--training-host")
    parser.add_argument("--ssh-identity", type=Path)
    arguments = parser.parse_args()
    if arguments.wait_pid <= 0:
        parser.error("wait-pid must be positive")
    directory = arguments.run_dir.resolve()
    status = directory / "completion.json"
    status.write_text(json.dumps({"state": "waiting_for_training"}) + "\n")
    identity = (
        ["-i", str(arguments.ssh_identity), "-o", "IdentitiesOnly=yes"]
        if arguments.ssh_identity
        else []
    )
    ssh_base = [
        _executable("ssh"),
        *identity,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
    ]
    try:
        while True:
            if arguments.training_host:
                result = subprocess.run(
                    [
                        *ssh_base,
                        arguments.training_host,
                        shlex.join(["kill", "-0", str(arguments.wait_pid)]),
                    ],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 1:
                    break
                if result.returncode not in {0, 255}:
                    msg = f"unexpected remote training check: {result.stderr.decode()}"
                    raise RuntimeError(msg)
            else:
                try:
                    os.kill(arguments.wait_pid, 0)
                except ProcessLookupError:
                    break
            time.sleep(60)
        checkpoint = directory / "checkpoints/last.ckpt"
        config = OmegaConf.load(directory / "resolved.yaml")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = saved["global_step"]
        if step < config.trainer.max_steps:
            msg = f"Training stopped early at step {step}; not exporting"
            raise RuntimeError(msg)
        del saved
        output = directory / "export-final"
        subprocess.run(
            [
                str(Path.cwd() / ".venv/bin/python"),
                "-m",
                "rmind.scripts.export_d12_fm",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(directory / "resolved.yaml"),
                "--output",
                str(output),
            ],
            check=True,
        )
        engine = output / "model-rtx4090-trt11.engine"
        subprocess.run(
            [
                "/usr/bin/trtexec",
                f"--onnx={output / 'model.onnx'}",
                f"--saveEngine={engine}",
                "--noTF32",
                "--skipInference",
                "--memPoolSize=workspace:2048",
                "--device=0",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(arguments.coppernic / "target/debug/d12-flow-parity"),
                "--engine",
                str(engine),
                "--manifest",
                str(output / "manifest.json"),
                "--output",
                str(output / "coppernic-parity.json"),
            ],
            check=True,
        )
        validation_report = None
        if arguments.validation_host:
            host = arguments.validation_host
            remote = f"/home/max/d12-fm-validation/{directory.name}/final"
            ssh = [*ssh_base, host]
            subprocess.run([*ssh, shlex.join(["mkdir", "-p", remote])], check=True)
            subprocess.run(
                [
                    _executable("rsync"),
                    "-av",
                    "-e",
                    shlex.join(ssh_base),
                    "--exclude=*.engine",
                    str(output) + "/",
                    f"{host}:{remote}/",
                ],
                check=True,
            )
            verifier = Path(__file__).with_name("verify_d12_trt.py")
            subprocess.run(
                [
                    _executable("scp"),
                    *identity,
                    "-o",
                    "BatchMode=yes",
                    str(verifier),
                    f"{host}:{remote}/",
                ],
                check=True,
            )
            command = (
                shlex.join([
                    "python3",
                    "-u",
                    f"{remote}/verify_d12_trt.py",
                    "--manifest",
                    f"{remote}/manifest.json",
                    "--engine",
                    f"{remote}/model-validation-kit.engine",
                    "--report",
                    f"{remote}/trt-parity.json",
                ])
                + f" > {shlex.quote(remote + '/build.log')} 2>&1"
            )
            result = subprocess.run([*ssh, command], check=False)
            validation = output / "validation-kit"
            validation.mkdir(exist_ok=True)
            subprocess.run(
                [
                    _executable("rsync"),
                    "-av",
                    "-e",
                    shlex.join(ssh_base),
                    "--include=*.engine",
                    "--include=*.json",
                    "--include=build.log",
                    "--exclude=*",
                    f"{host}:{remote}/",
                    str(validation) + "/",
                ],
                check=True,
            )
            result.check_returncode()
            validation_report = json.loads((validation / "trt-parity.json").read_text())
            if not validation_report["passed"]:
                msg = "validation-kit parity failed"
                raise RuntimeError(msg)
        status.write_text(
            json.dumps(
                {
                    "state": "local_artifact_validated",
                    "global_step": step,
                    "engine": str(engine),
                    "deployment_target_validated": False,
                    "validation_kit": validation_report,
                    "remaining": "D12 .169 engine build, live Coppernic wiring and target replay",
                },
                indent=2,
            )
            + "\n"
        )
    except Exception as error:
        status.write_text(
            json.dumps({"state": "failed", "error": str(error)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
