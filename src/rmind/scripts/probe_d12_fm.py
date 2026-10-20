"""Paired counterfactual image and world-frame pickup/dropoff sensitivity probes."""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageFilter

from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.scripts.eval_d12_fm import ACTIONS, DEFAULT_JOBS, load_tokens

ACTIVE_DEADBAND = 0.05
WINDOW_FRAMES = 6
MINIMUM_SAMPLES = 2
ProbeContext = dict[str, Any]
ProbeReport = dict[str, Any]


def world_offset_token(
    offset: tuple[float, float], headings: Sequence[float]
) -> np.ndarray:
    heading_radians = np.deg2rad(np.asarray(headings))
    world_x, world_y = offset
    return (
        np.stack(
            [
                np.cos(heading_radians) * world_x + np.sin(heading_radians) * world_y,
                -np.sin(heading_radians) * world_x + np.cos(heading_radians) * world_y,
            ],
            axis=-1,
        )
        / 10
    )


def paired_metrics(baseline: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    baseline_near = baseline[..., :WINDOW_FRAMES, :]
    changed_near = changed[..., :WINDOW_FRAMES, :]
    baseline_active = (np.abs(baseline_near[..., [0, 2]]) > ACTIVE_DEADBAND).any(
        axis=-1
    )
    changed_idle = np.asarray(
        (np.abs(changed_near[..., [0, 2]]) <= ACTIVE_DEADBAND).all(axis=-1)
    )
    per_action = {}
    for index, action in enumerate(ACTIONS):
        original = baseline_near[..., index]
        perturbed = changed_near[..., index]
        active = np.abs(original) > ACTIVE_DEADBAND
        neutral = np.abs(perturbed) <= ACTIVE_DEADBAND
        per_action[action] = {
            "paired_abs_change": float(np.abs(perturbed - original).mean()),
            "baseline_abs_mean": float(np.abs(original).mean()),
            "changed_abs_mean": float(np.abs(perturbed).mean()),
            "baseline_neutral_fraction": float((~active).mean()),
            "changed_neutral_fraction": float(neutral.mean()),
            "neutral_given_baseline_active": float(neutral[active].mean())
            if active.any()
            else None,
            "baseline_active_count": int(active.sum()),
        }
    return {
        "per_action": per_action,
        "paired_near_abs_change": float(np.abs(changed_near - baseline_near).mean()),
        "paired_per_action_abs_change": dict(
            zip(
                ACTIONS,
                np
                .abs(changed_near - baseline_near)
                .mean(axis=tuple(range(baseline_near.ndim - 1)))
                .tolist(),
                strict=True,
            )
        ),
        "baseline_no_drive_lift_fraction": float((~baseline_active).mean()),
        "changed_no_drive_lift_fraction": float(changed_idle.mean()),
        "collapse_to_idle_given_baseline_active": float(
            changed_idle[baseline_active].mean()
        )
        if baseline_active.any()
        else None,
        "baseline_drive_lift_abs_mean": float(
            np.abs(baseline_near[..., [0, 2]]).mean()
        ),
        "changed_drive_lift_abs_mean": float(np.abs(changed_near[..., [0, 2]]).mean()),
        "baseline_between_sample_std": float(baseline_near.std(axis=-3).mean()),
    }


def summarize_probes(
    stacked: dict[str, np.ndarray], contexts: list[ProbeContext]
) -> dict[str, Any]:
    metrics = {}
    for split in ("train", "val"):
        for category in ("all", "hold", "lift_lower", "drive_low", "drive_high"):
            selected = np.array([
                context["split"] == split
                and (category == "all" or context["category"] == category)
                for context in contexts
            ])
            if selected.any():
                metrics[f"{split}/{category}"] = {
                    name: paired_metrics(
                        stacked["baseline"][selected], values[selected]
                    )
                    for name, values in stacked.items()
                    if name != "baseline"
                }
    return metrics


def plot_goal_sensitivity(report: ProbeReport, output: Path) -> None:
    split = "val/all" if "val/all" in report["metrics"] else "train/all"
    metrics = report["metrics"][split]
    cases = report["cases"]
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for column, action in enumerate(ACTIONS):
        axes[0, column].set_title(
            {"fork1": "Fork lift/lower"}.get(action, action.title())
        )
        for field in ("pickup", "dropoff"):
            radii = sorted({
                case["radius_m"] for case in cases.values() if case["field"] == field
            })
            for row, metric in enumerate((
                "paired_abs_change",
                "changed_abs_mean",
                "neutral_given_baseline_active",
            )):
                values = []
                for radius in radii:
                    directional = [
                        metrics[name]["per_action"][action][metric]
                        for name, case in cases.items()
                        if case["field"] == field
                        and case["radius_m"] == radius
                        and metrics[name]["per_action"][action][metric] is not None
                    ]
                    values.append(
                        float(np.mean(directional)) if directional else np.nan
                    )
                axes[row, column].plot(radii, values, marker="o", label=field)
        baseline = next(iter(metrics.values()))["per_action"][action][
            "baseline_abs_mean"
        ]
        axes[1, column].axhline(
            baseline, color="black", linestyle="--", label="Unperturbed"
        )
        axes[0, column].set_ylabel("Mean |changed - original| command")
        axes[1, column].set_ylabel("Mean |command|")
        axes[1, column].set_ylim(bottom=0)
        axes[2, column].set_ylabel("P(|changed| <= .05 | |original| > .05)")
        axes[2, column].set_ylim(0, 1)
        axes[2, column].set_xlabel("World-frame goal offset (metres)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle(
        f"Goal sensitivity by action — {split}, first 6 commands (0.2 s)\n"
        "Paired same-noise samples; mean of four offset directions. Near-zero steering means centered, not stopped."
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output / "goal_sensitivity.png", dpi=140)
    plt.close(figure)


def refresh_report(output: Path) -> None:
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    with np.load(output / "predictions.npz") as saved:
        report["metrics"] = summarize_probes(dict(saved), report["contexts"])
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_goal_sensitivity(report, output)


def publish_sensitivity(output: Path, run_path: str) -> None:
    import wandb

    entity, project, run_id = run_path.split("/")
    report = json.loads((output / "report.json").read_text())
    columns = [
        "slice",
        "intervention",
        "action",
        "paired_abs_change",
        "baseline_abs_mean",
        "changed_abs_mean",
        "baseline_neutral_fraction",
        "changed_neutral_fraction",
        "neutral_given_baseline_active",
        "baseline_active_count",
    ]
    rows = [
        [split, case, action, *[values[key] for key in columns[3:]]]
        for split, cases in report["metrics"].items()
        for case, metrics in cases.items()
        for action, values in metrics["per_action"].items()
    ]
    with wandb.init(entity=entity, project=project, id=run_id, resume="must") as run:
        run.log({
            "goal_sensitivity": wandb.Image(str(output / "goal_sensitivity.png")),
            "sensitivity_by_action": wandb.Table(columns=columns, data=rows),
        })
        artifact = wandb.Artifact(f"sensitivity-by-action-{run_id}", type="evaluation")
        for name in ("report.json", "goal_sensitivity.png"):
            artifact.add_file(str(output / name), name=name)
        run.log_artifact(artifact)


def image_probe(
    kind: str, selected_camera: str | None = None
) -> Callable[[str, np.ndarray], np.ndarray]:
    def transform(camera: str, images: np.ndarray) -> np.ndarray:
        if selected_camera is not None and camera != selected_camera:
            return images
        if kind == "black":
            return np.zeros_like(images)
        if kind == "stale":
            return np.repeat(images[:1], len(images), axis=0)
        if kind == "dim":
            return np.clip(images.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
        if kind == "bright":
            return np.clip(images.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        if kind == "blur":
            return np.stack([
                np.asarray(Image.fromarray(frame).filter(ImageFilter.GaussianBlur(4)))
                for frame in images
            ])
        raise ValueError(kind)

    return transform


def select_contexts(samples: list[ProbeContext], count: int) -> list[tuple[int, str]]:
    categories = {}
    for index, sample in enumerate(samples):
        drive = np.abs(sample["traction"][-1][:WINDOW_FRAMES]).max() > ACTIVE_DEADBAND
        lift = np.abs(sample["fork1"][-1][:WINDOW_FRAMES]).max() > ACTIVE_DEADBAND
        category = (
            "lift_lower"
            if lift
            else (
                "hold"
                if not drive
                else ("drive_high" if sample["fork_above_300"][-1] else "drive_low")
            )
        )
        categories.setdefault(category, []).append(index)
    chosen = []
    per_category = max(1, count // len(categories))
    for category, indices in categories.items():
        chosen.extend(
            (indices[offset], category)
            for offset in np.linspace(
                0, len(indices) - 1, min(per_category, len(indices)), dtype=int
            )
        )
    return sorted(chosen[:count])


def predict_tokens(
    policy: PatchPolicyContinuous, tokens: torch.Tensor, noise: torch.Tensor
) -> np.ndarray:
    features = policy.encoder(tokens.flatten(1, 2), num_frames=tokens.shape[1])[:, -1]
    if policy.norm is not None:
        features = policy.norm(features)
    return (
        (
            policy.fm_sample(features.expand(len(noise), -1), x0=noise)
            * policy.action_std
            + policy.action_mean
        )
        .cpu()
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wandb-run",
        help="Existing entity/project/run_id to update with per-action plots and table",
    )
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Recompute metrics/plots from saved predictions without inference",
    )
    parser.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS)
    parser.add_argument("--contexts-per-job", type=int, default=12)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--fm-steps", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.refresh_report:
        refresh_report(arguments.output)
        if arguments.wandb_run:
            publish_sensitivity(arguments.output, arguments.wandb_run)
        return
    if arguments.checkpoint is None or arguments.config is None:
        parser.error(
            "--checkpoint and --config are required unless --refresh-report is set"
        )
    if (
        arguments.contexts_per_job < 1
        or arguments.samples < MINIMUM_SAMPLES
        or arguments.fm_steps < 1
    ):
        msg = "positive contexts/steps and at least two samples required"
        raise ValueError(msg)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if (arguments.output / "report.json").exists():
        msg = "choose a new output directory"
        raise ValueError(msg)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.load(arguments.config)
    if not isinstance(config, DictConfig):
        msg = "expected a mapping configuration"
        raise TypeError(msg)
    policy = (
        PatchPolicyContinuous
        .load_from_checkpoint(
            arguments.checkpoint, map_location="cpu", weights_only=False
        )
        .eval()
        .cuda()
    )
    policy.fm_inference_steps = arguments.fm_steps
    image_cases = {
        f"image/{kind}/all": image_probe(kind)
        for kind in ("black", "stale", "dim", "bright", "blur")
    }
    image_cases.update({
        f"image/black/{camera}": image_probe("black", camera)
        for camera in policy.cameras
    })
    offsets = [
        (radius, direction, (radius * offset[0], radius * offset[1]))
        for radius in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
        for direction, offset in (
            ("+x", (1, 0)),
            ("-x", (-1, 0)),
            ("+y", (0, 1)),
            ("-y", (0, -1)),
        )
    ]
    stages = config.datamodule.val.dataset.samples.pipeline.functions
    builder, positioner, windower = (instantiate(stage.func) for stage in stages[:3])
    train_jobs = {
        Path(path).name
        for path in config.datamodule.train.dataset.samples.inputs.job_dir
    }
    val_jobs = {
        Path(path).name for path in config.datamodule.val.dataset.samples.inputs.job_dir
    }
    cases = {"baseline": []}
    contexts = []
    case_info = {}
    with torch.inference_mode():
        for job_name in arguments.jobs:
            if job_name not in train_jobs | val_jobs:
                msg = f"unknown split for {job_name}"
                raise ValueError(msg)
            job = Path(config.paths.d12) / job_name
            rows = builder(job_dir=job)
            samples = list(
                windower(frames=positioner(row_table=rows)).iter_rows(named=True)
            )
            row_map = {
                row["cam_fork/frame_idx"]: row for row in rows.iter_rows(named=True)
            }
            donor = samples[len(samples) // 2]
            donor_tokens = torch.cat(
                load_tokens(policy, job, donor, range(WINDOW_FRAMES), "cuda:0"), dim=1
            )
            state_count = 1 + len(policy.observations)
            for sample_index, category in select_contexts(
                samples, arguments.contexts_per_job
            ):
                sample = samples[sample_index]
                tokens = torch.cat(
                    load_tokens(policy, job, sample, range(WINDOW_FRAMES), "cuda:0"),
                    dim=1,
                )
                headings = [
                    row_map[index]["ego_heading"]
                    for index in sample["cam_fork/frame_idx"]
                ]
                seed = arguments.seed + int.from_bytes(
                    hashlib.sha256(f"{job_name}:{sample_index}".encode()).digest()[:4],
                    "little",
                )
                generator = torch.Generator(device="cuda:0").manual_seed(seed)
                noise = torch.randn(
                    arguments.samples,
                    policy.action_horizon * 3,
                    generator=generator,
                    device="cuda:0",
                )
                baseline = predict_tokens(policy, tokens, noise)
                cases["baseline"].append(baseline)
                contexts.append({
                    "job": job_name,
                    "split": "train" if job_name in train_jobs else "val",
                    "sample_index": sample_index,
                    "fork_frame": sample["cam_fork/frame_idx"][-1],
                    "category": category,
                    "ego_xy": [
                        row_map[sample["cam_fork/frame_idx"][-1]][name]
                        for name in ("ego_x", "ego_y")
                    ],
                    "pickup_xy": [
                        row_map[sample["cam_fork/frame_idx"][-1]][name]
                        for name in ("pallet_x", "pallet_y")
                    ],
                    "dropoff_xy": [
                        row_map[sample["cam_fork/frame_idx"][-1]][name]
                        for name in ("target_x", "target_y")
                    ],
                })
                for name, intervention in image_cases.items():
                    changed = torch.cat(
                        load_tokens(
                            policy,
                            job,
                            sample,
                            range(WINDOW_FRAMES),
                            "cuda:0",
                            image_intervention=intervention,
                        ),
                        dim=1,
                    )
                    cases.setdefault(name, []).append(
                        predict_tokens(policy, changed, noise)
                    )
                changed = tokens.clone()
                changed[:, :, state_count:] = donor_tokens[:, :, state_count:]
                cases.setdefault("image/other_time/all", []).append(
                    predict_tokens(policy, changed, noise)
                )
                for field, label in (
                    ("relative_ego_pos", "pickup"),
                    ("relative_dropoff_pos", "dropoff"),
                ):
                    token_index = 1 + list(policy.observations).index(field)
                    for radius, direction, offset in offsets:
                        name = f"{label}/{radius}/{direction}"
                        values = np.asarray(sample[field]) + world_offset_token(
                            offset, headings
                        )
                        values = torch.tensor(
                            np.clip(values, -1, 1), dtype=torch.float32, device="cuda:0"
                        ).unsqueeze(0)
                        changed = tokens.clone()
                        embeddings = policy.observation_embeddings
                        if embeddings is None:
                            msg = "policy has no observation embeddings"
                            raise RuntimeError(msg)
                        changed[:, :, token_index : token_index + 1] = embeddings[
                            field
                        ](values)
                        cases.setdefault(name, []).append(
                            predict_tokens(policy, changed, noise)
                        )
                        case_info[name] = {
                            "field": label,
                            "radius_m": radius,
                            "direction": direction,
                            "world_offset_m": offset,
                        }
    stacked = {key: np.stack(values) for key, values in cases.items()}
    np.savez_compressed(
        arguments.output / "predictions.npz", **stacked, allow_pickle=False
    )
    metrics = summarize_probes(stacked, contexts)
    with arguments.checkpoint.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_sha256": checksum,
        "config": OmegaConf.to_container(config, resolve=True),
        "fm_steps": arguments.fm_steps,
        "samples": arguments.samples,
        "seed": arguments.seed,
        "contexts": contexts,
        "cases": case_info,
        "metrics": metrics,
        "limitations": "Paired same-noise interventions across all six observation frames, fresh six-frame trunk (not continuous KV). Goals shifted consistently in world coordinates then transformed to each frame's ego coordinates /10. Blackout and other-time images are OOD diagnostics. Sensitivity is NOT counterfactual task correctness; no counterfactual expert actions exist. No spatial rollout inferred from commands. Sample numbers are not mode labels. No model input includes absolute world position directly.",
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plot_goal_sensitivity(report, arguments.output)
    if arguments.wandb_run:
        publish_sensitivity(arguments.output, arguments.wandb_run)


if __name__ == "__main__":
    main()
