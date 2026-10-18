"""Launch the experimental native D12 RTC head from an audited fork-only baseline."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rmind.models.patch_policy_continuous import PatchPolicyContinuous

ACTION_HORIZON = 50
TRAIN_JOB_COUNT = 34
VALIDATION_JOB_COUNT = 4


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")


class SmokeCheckpointQuarantine(pl.Callback):
    @override
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint["test_only"] = True
        checkpoint["rtc_test_fixture"] = {
            "deployable": False,
            "head_trained": False,
            "purpose": "three-step training smoke, not a deployment candidate",
        }


def build_config(baseline: Path, directory: Path) -> DictConfig:
    config = OmegaConf.load(baseline / "resolved.yaml")
    if (
        list(config.model.cameras) != ["cam_fork"]
        or config.model.action_horizon != ACTION_HORIZON
    ):
        msg = "The baseline must be fork-only H50"
        raise ValueError(msg)
    if (
        len(config.datamodule.train.dataset.samples.inputs.job_dir) != TRAIN_JOB_COUNT
        or len(config.datamodule.val.dataset.samples.inputs.job_dir)
        != VALIDATION_JOB_COUNT
    ):
        msg = "Expected the original34/4 whole-job split"
        raise ValueError(msg)
    config.model.fm_head = {
        "_target_": "rmind.components.rtc_flow_head.TokenTimeFlowHead",
        "feat_dim": 512,
        "action_dim": 3,
        "action_horizon": 50,
        "channel_dim": 256,
        "channel_hidden_dim": 512,
        "token_hidden_dim": 64,
        "num_layers": 4,
        "max_prefix": 6,
        "time_embedding_mode": "native",
    }
    config.model.fm_inference_steps = 5
    config.model.optimizer.lr = 3e-5
    config.matmul_precision = "highest"
    config.ckpt_path = None
    config.trainer.precision = "32-true"
    config.trainer.default_root_dir = str(directory)
    config.trainer.logger.save_dir = str(directory)
    config.trainer.callbacks[1].dirpath = str(directory / "checkpoints")
    config.paths.rbyte.cache = str(directory / "rbyte_cache")
    config.wandb.name = "d12-rtc-native-fork-h50-fp32"
    config.wandb.tags += [
        "training_time_rtc",
        "native_time_embedding",
        "fp32",
        "experimental_not_deployed",
    ]
    config.wandb.notes = (
        "Experimental native token-time RTC: fork-only, original34train/4val jobs, "
        "30Hz H50 actions,6-frame history at5Hz, height+missing mask, no subgoals. "
        "Compatible vision/trunk/state weights from ptktynvw step35000; fresh RTC head. "
        "Native analytical time encoding, no external vector/JAX runtime. "
        "Prefix0..6 sampled proportional exp(-length);5 Euler steps. "
        "FP32/noTF32, frozenDINO, originalAdamW betas/decay and35k warmup/cosine, lr3e-5. "
        "D12 experimental authorization, not proxy-quality or deployment certification."
    )
    return config


def warmstart(model: Any, source_state: dict) -> dict:
    destination = model.state_dict()
    shared = {
        name: value
        for name, value in source_state.items()
        if not name.startswith("fm_head.")
    }
    expected = {name for name in destination if not name.startswith("fm_head.")}
    if set(shared) != expected:
        msg = f"Non-head state mismatch: missing={sorted(expected - set(shared))}, extra={sorted(set(shared) - expected)}"
        raise ValueError(msg)
    for name, value in shared.items():
        if value.shape != destination[name].shape:
            msg = f"Warmstart tensor shape mismatch: {name}"
            raise ValueError(msg)
    for name in ("action_mean", "action_std"):
        if not torch.equal(shared[name], destination[name]):
            msg = f"Action normalization changed: {name}"
            raise ValueError(msg)
    fresh_head = {
        name: value.clone()
        for name, value in destination.items()
        if name.startswith("fm_head.")
    }
    result = model.load_state_dict(shared, strict=False)
    if set(result.missing_keys) != set(fresh_head) or result.unexpected_keys:
        msg = "Unexpected partial-load result"
        raise ValueError(msg)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[name], value) for name, value in shared.items()):
        msg = "A shared tensor did not load exactly"
        raise ValueError(msg)
    if any(not torch.equal(loaded[name], value) for name, value in fresh_head.items()):
        msg = "The new RTC head was modified by warmstart"
        raise ValueError(msg)
    return {
        "loaded_nonhead_tensors": len(shared),
        "loaded_nonhead_elements": sum(value.numel() for value in shared.values()),
        "ignored_source_head_keys": sorted(
            name for name in source_state if name.startswith("fm_head.")
        ),
        "fresh_destination_head_keys": sorted(fresh_head),
        "all_shared_tensors_bitwise_equal": True,
        "new_head_unchanged": True,
        "action_normalization_bitwise_equal": True,
        "native_time_mode": model.fm_head.time_embedding_mode,
        "native_time_frequency_sha256": hashlib.sha256(
            model.fm_head.time_frequency.cpu().numpy().tobytes()
        ).hexdigest(),
    }


class TrainingHealth(pl.Callback):
    def __init__(
        self, directory: Path, early_steps: tuple[int, ...] = (1, 100)
    ) -> None:
        super().__init__()
        self.directory = directory
        self.early_steps = set(early_steps)
        self.saved = set()

    @override
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        norms = {}
        for group, prefix in (("head", "fm_head."), ("trunk", "encoder.")):
            gradients = [
                parameter.grad.detach()
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            if not gradients or any(
                not torch.isfinite(value).all() for value in gradients
            ):
                msg = f"Missing or nonfinite {group} gradients"
                raise FloatingPointError(msg)
            norm = float(
                torch.sqrt(sum(value.float().square().sum() for value in gradients))
            )
            if not norm > 0:
                msg = f"Zero {group} gradient norm"
                raise FloatingPointError(msg)
            norms[group + "_grad_norm"] = norm
        row = {
            "utc": datetime.now(UTC).isoformat(),
            "optimizer_step": trainer.global_step + 1,
            **norms,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        }
        with (self.directory / "health.jsonl").open("a", buffering=1) as stream:
            stream.write(json.dumps(row) + "\n")

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is None or not torch.isfinite(loss).all():
            msg = "Nonfinite training loss"
            raise FloatingPointError(msg)
        step = trainer.global_step
        if step in self.early_steps and step not in self.saved:
            directory = self.directory / "checkpoints"
            directory.mkdir(exist_ok=True)
            path = directory / f"early-step={step}.ckpt"
            trainer.save_checkpoint(path)
            self.saved.add(step)
            write_json(
                path.with_suffix(".json"),
                {
                    "step": step,
                    "utc": datetime.now(UTC).isoformat(),
                    "path": str(path),
                    "sha256": digest(path),
                    "loss": float(loss.detach()),
                    "experimental_not_deployed": True,
                },
            )


def resolved_executable(name: str) -> str:
    if name not in {"git", "nvidia-smi"}:
        msg = "Only the fixed git and nvidia-smi provenance commands are supported"
        raise ValueError(msg)
    executable = shutil.which(name)
    if executable is None:
        msg = f"Required provenance executable is missing: {name}"
        raise FileNotFoundError(msg)
    return str(Path(executable).resolve())


def archive_inputs(
    config: DictConfig, baseline: Path, directory: Path, checkpoint: Path
) -> None:
    root = Path(__file__).resolve().parents[3]
    git = resolved_executable("git")
    nvidia_smi = resolved_executable("nvidia-smi")
    paths = subprocess.check_output(
        [git, "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for name in sorted(set(paths)):
            if (root / name).is_file():
                archive.add(root / name, arcname=name)
    with tarfile.open(directory / "source.tar.gz", "r:gz") as archive:
        archive.extractall(directory / "source_snapshot", filter="data")
    records = set()
    for split in ("train", "val"):
        for job in config.datamodule[split].dataset.samples.inputs.job_dir:
            folder = Path(job)
            records |= {folder / "data_for_training.mcap", folder / "dropoff.parquet"}
            records.update(folder.glob("height--*.mcap"))
            records.update(folder.glob("cam_*/frame_info_*.txt"))
            records.update(folder.glob("cam_*/video.mp4"))
    if any(not path.is_file() for path in records):
        msg = "A required dataset input is missing"
        raise FileNotFoundError(msg)

    def record(path: Path) -> dict[str, Any]:
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        dataset = list(pool.map(record, sorted(records)))
    write_json(
        directory / "inputs.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "dataset_scope": "All38 jobs: trainingMCAP, dropoff, height sidecars, camera frame indexes and MP4s; standarddata.mcap is not consumed",
            "checkpoint": record(checkpoint),
            "baseline": [
                record(baseline / name)
                for name in ("resolved.yaml", "action_stats.yaml", "audit.json")
            ],
            "source_archive": record(directory / "source.tar.gz"),
            "resolved_config": record(directory / "resolved.yaml"),
            "git_head": subprocess.check_output(
                [git, "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "provenance_executables": {"git": git, "nvidia-smi": nvidia_smi},
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
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    )
    for name in ("action_stats.yaml", "audit.json"):
        shutil.copy2(baseline / name, directory / name)


def smoke(config: DictConfig, checkpoint: Path, directory: Path) -> None:
    smoke_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    smoke_config.datamodule.train.dataset.samples.inputs.job_dir = [
        str(Path(config.paths.d12) / name)
        for name in (
            "2026-08-25--16-35-31",
            "2026-10-02--15-01-01",
            "2026-10-12--15-40-43",
        )
    ]
    smoke_config.datamodule.val.dataset.samples.inputs.job_dir = [
        str(Path(config.paths.d12) / "2026-10-12--15-59-15")
    ]
    pl.seed_everything(config.seed, workers=True)
    model = instantiate(smoke_config.model)
    audit = warmstart(
        model,
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
    )
    write_json(directory / "warmstart.json", audit)
    data = instantiate(smoke_config.datamodule)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_steps=3,
        limit_train_batches=12,
        limit_val_batches=2,
        num_sanity_val_steps=1,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            TrainingHealth(directory, early_steps=()),
            SmokeCheckpointQuarantine(),
        ],
    )
    trainer.fit(model, datamodule=data)
    path = directory / "smoke.ckpt"
    trainer.save_checkpoint(path)
    restored = (
        PatchPolicyContinuous
        .load_from_checkpoint(path, map_location="cpu")
        .cuda()
        .eval()
    )
    model.cuda().eval()
    features = torch.randn(2, 512, device="cuda")
    noise = torch.randn(2, 50, 3, device="cuda")
    prefix = torch.randn_like(noise)
    mask = torch.arange(50, device="cuda")[None] < torch.tensor(
        [[0], [6]], device="cuda"
    )
    with torch.inference_mode():
        expected = model.fm_head.sample(features, noise, prefix, mask)
        actual = restored.fm_head.sample(features, noise, prefix, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    write_json(
        directory / "passed.json",
        {
            "steps": trainer.global_step,
            "precision": "32-true",
            "head_and_trunk_gradients_checked": True,
            "checkpoint_sha256": digest(path),
            "explicit_noise_reload_max_error": float((actual - expected).abs().max()),
            "native_time_mode": restored.fm_head.time_embedding_mode,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "smoke", "train"), required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    checkpoint = args.baseline / "checkpoints/epoch=5-step=35000.ckpt"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.phase == "prepare":
        directory.mkdir(parents=True, exist_ok=False)
        config = build_config(args.baseline, directory)
        OmegaConf.save(config, directory / "resolved.yaml", resolve=True)
        archive_inputs(config, args.baseline, directory, checkpoint)
        return
    config = OmegaConf.load(directory / "resolved.yaml")
    inputs = json.loads((directory / "inputs.json").read_text())
    if (
        digest(checkpoint) != inputs["checkpoint"]["sha256"]
        or digest(directory / "resolved.yaml") != inputs["resolved_config"]["sha256"]
    ):
        msg = "Frozen checkpoint/config changed"
        raise ValueError(msg)
    if args.phase == "smoke":
        smoke_directory = directory / "smoke"
        smoke_directory.mkdir(exist_ok=False)
        smoke(config, checkpoint, smoke_directory)
        return
    if (
        not (directory / "smoke/passed.json").is_file()
        or (directory / "run.json").exists()
    ):
        msg = "Requires successful smoke and no existing full run"
        raise ValueError(msg)
    pl.seed_everything(config.seed, workers=True)
    model = instantiate(config.model)
    audit = warmstart(
        model,
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
    )
    write_json(directory / "warmstart.json", audit)
    run = wandb.init(
        config=OmegaConf.to_container(config, resolve=True), **config.wandb
    )
    write_json(
        directory / "run.json",
        {
            "id": run.id,
            "url": run.url,
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "experimental_not_deployed": True,
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": digest(Path(__file__).resolve()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "precision": "32-true",
            "tf32": torch.backends.cuda.matmul.allow_tf32,
        },
    )
    data = instantiate(config.datamodule)
    trainer = instantiate(config.trainer)
    trainer.callbacks.append(TrainingHealth(directory))
    trainer.fit(model, datamodule=data)
    run.finish()


if __name__ == "__main__":
    main()
