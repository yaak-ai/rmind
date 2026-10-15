"""Audit the D12 split and fit action statistics using training jobs only."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rmind.scripts.d12_fm_config import configure_cameras, dataset_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--camera-layout", choices=("forward", "backward", "fork"), default="forward"
    )
    parser.add_argument("--data-root", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    config_dir = Path(__file__).resolve().parents[3] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(
            config_name="train",
            overrides=[
                "experiment=palletjack/patch_policy/d12_fm",
                f"action_horizon={arguments.horizon}",
                "action_mean=[0,0,0]",
                "action_std=[1,1,1]",
                *dataset_overrides(arguments.camera_layout),
            ],
        )
    configure_cameras(config, arguments.camera_layout)
    if arguments.data_root:
        config.paths.d12 = str(arguments.data_root.resolve())
    splits = {}
    for split in ("train", "val"):
        dataset = config.datamodule[split].dataset
        splits[split] = [Path(path) for path in dataset.samples.inputs.job_dir]
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("training and validation jobs overlap")
    manifest = {"action_order": ["traction", "steering", "fork1"], "jobs": []}
    train_actions = []
    for split, jobs in splits.items():
        stages = config.datamodule[split].dataset.samples.pipeline.functions
        builder, positioner, windower = [
            instantiate(stage.func) for stage in stages[:3]
        ]

        def inspect_job(job):
            rows = builder(job_dir=job)
            positioned = positioner(row_table=rows)
            samples = windower(frames=positioned)
            if rows.select(manifest["action_order"]).null_count().to_numpy().sum():
                raise ValueError(f"null actions in {job}")
            actions = rows.select(manifest["action_order"]).to_numpy()
            if not np.isfinite(actions).all():
                raise ValueError(f"nonfinite actions in {job}")
            for camera in config.cameras:
                indices = samples[f"{camera}/frame_idx"].to_numpy()
                for frame_index in np.unique(indices):
                    image = job / camera / "frames/256x144" / f"{frame_index:06d}.jpg"
                    if not image.is_file():
                        raise ValueError(f"missing image: {image}")
            heights = np.stack(rows["fork_height"].to_list())
            audit = {
                "job": job.name,
                "split": split,
                "frames": len(rows),
                "windows": len(samples),
                "valid_height_frames": int(heights[:, 2].sum()),
                "height_fraction": float(heights[:, 2].mean()),
                "action_shape": list(samples["traction"].to_numpy().shape[1:]),
                "mcap_bytes": (job / "data_for_training.mcap").stat().st_size,
            }
            return audit, actions

        with ThreadPoolExecutor(max_workers=4) as pool:
            for audit, actions in pool.map(inspect_job, jobs):
                manifest["jobs"].append(audit)
                if split == "train":
                    train_actions.append(actions)
                print(json.dumps(audit), flush=True)
    actions = np.concatenate(train_actions).astype(np.float64)
    statistics = {
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).clip(1e-3).tolist(),
    }
    manifest.update(statistics)
    manifest["action_horizon"] = arguments.horizon
    manifest["camera_layout"] = arguments.camera_layout
    manifest["cameras"] = list(config.cameras)
    manifest["data_root"] = str(config.paths.d12)
    manifest["action_hz"] = 30
    manifest["observation_hz"] = 5
    (arguments.output / "audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    OmegaConf.save(OmegaConf.create(statistics), arguments.output / "action_stats.yaml")
    print(json.dumps(statistics), flush=True)


if __name__ == "__main__":
    main()
