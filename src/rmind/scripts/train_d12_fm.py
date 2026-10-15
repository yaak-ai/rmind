"""Launch the audited D12 FM experiment, or validate it with real-data training."""

import argparse
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import wandb
from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.scripts.d12_fm_config import configure_cameras, dataset_overrides
from rmind.scripts.train import _train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--camera-count", type=int, choices=(1, 3), default=3)
    parser.add_argument("--camera-layout", choices=("forward", "backward", "fork"))
    parser.add_argument(
        "--sampling", choices=("uniform", "balanced"), default="uniform"
    )
    parser.add_argument("--steering-weight", type=float, default=1.0)
    arguments = parser.parse_args()
    directory = arguments.run_dir.resolve()
    source = Path(__file__).resolve().parents[3]
    statistics = OmegaConf.load(directory / "action_stats.yaml")
    audit = json.loads((directory / "audit.json").read_text())
    layout = arguments.camera_layout or (
        "fork" if arguments.camera_count == 1 else audit.get("camera_layout", "forward")
    )
    if (layout == "backward") != (audit.get("camera_layout") == "backward"):
        raise ValueError("backward cameras require a matching data audit")
    with initialize_config_dir(config_dir=str(source / "config"), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=palletjack/patch_policy/d12_fm",
                f"action_horizon={audit['action_horizon']}",
                f"action_mean={list(statistics.action_mean)}",
                f"action_std={list(statistics.action_std)}",
                *dataset_overrides(layout),
            ],
        )
    configure_cameras(config, layout)
    if "data_root" in audit:
        config.paths.d12 = audit["data_root"]
    if arguments.steering_weight <= 0:
        raise ValueError("steering weight must be positive")
    OmegaConf.update(
        config,
        "model.fm_action_weights",
        [1.0, arguments.steering_weight, 1.0],
        force_add=True,
    )
    if arguments.sampling == "balanced":
        config.datamodule.train._target_ = "rmind.data.d12_sampling.balanced_loader"
        OmegaConf.update(config, "datamodule.train.seed", config.seed, force_add=True)
        OmegaConf.update(
            config, "datamodule.train.balanced_fraction", 0.5, force_add=True
        )
    config.wandb.tags += [
        f"sampling:{arguments.sampling}",
        f"steering_weight:{arguments.steering_weight}",
    ]
    OmegaConf.update(config, "trainer.default_root_dir", str(directory), force_add=True)
    OmegaConf.update(config, "trainer.logger.save_dir", str(directory), force_add=True)
    OmegaConf.update(
        config,
        "trainer.callbacks.1.dirpath",
        str(directory / "checkpoints"),
        force_add=True,
    )
    config.paths.rbyte.cache = str(directory / "rbyte_cache")
    if arguments.resume:
        config.ckpt_path = str(arguments.resume.resolve())
    if arguments.smoke:
        config.datamodule.train.dataset.samples.inputs.job_dir = [
            str(Path(config.paths.d12) / "2026-10-02--15-01-01"),
            str(Path(config.paths.d12) / "2026-10-12--15-40-43"),
        ]
        config.datamodule.val.dataset.samples.inputs.job_dir = [
            str(Path(config.paths.d12) / "2026-10-12--15-59-15")
        ]
        pl.seed_everything(config.seed, workers=True)
        torch.set_float32_matmul_precision(config.matmul_precision)
        model = instantiate(config.model)
        data = instantiate(config.datamodule)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="bf16-mixed",
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
        )
        trainer.fit(model, datamodule=data)
        checkpoint = directory / "smoke.ckpt"
        trainer.save_checkpoint(checkpoint)
        restored = PatchPolicyContinuous.load_from_checkpoint(checkpoint).cuda().eval()
        model.cuda().eval()
        batch = next(iter(data.val_dataloader()))
        batch = model.transfer_batch_to_device(batch, torch.device("cuda"), 0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(11)
            expected = model.eval()(batch)
            torch.manual_seed(11)
            actual = restored(batch)
        for key, value in expected.items(include_nested=True, leaves_only=True):
            torch.testing.assert_close(value, actual[key])
        print(
            json.dumps({
                "smoke": "passed",
                "steps": trainer.global_step,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            }),
            flush=True,
        )
        return
    OmegaConf.save(config, directory / "resolved.yaml", resolve=True)
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=source,
        text=True,
    ).splitlines()
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for path in paths:
            if (source / path).is_file():
                archive.add(source / path, arcname=path)
    resolved = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    run = wandb.init(config=resolved, **config.wandb)
    run.log_code(
        root=str(source),
        include_fn=lambda path: str(Path(path).relative_to(source)) in paths,
    )
    (directory / "run.json").write_text(
        json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
    )
    _train(config)
    run.finish()


if __name__ == "__main__":
    main()
