"""Recorded-observation open-loop evaluation of D12 flow matching."""

import argparse
import csv
import hashlib
import html
import json
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import simplejpeg
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from rmind.models.d12_flow_decoder import D12FlowDecoderStep
from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep

ACTIONS = ("traction", "steering", "fork1")
DEFAULT_JOBS = (
    "2026-10-12--15-40-43",
    "2026-09-10--15-17-38",
    "2026-10-02--15-49-44",
    "2026-10-12--15-59-15",
)


def action_metrics(predictions, targets):
    if (
        predictions.ndim != 4
        or predictions.shape[-1] != 3
        or targets.shape != predictions.shape[:1] + predictions.shape[2:]
    ):
        raise ValueError("expected predictions N,S,H,3 and targets N,H,3")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("non-finite predictions or targets")
    reference = np.broadcast_to(targets[:, None], predictions.shape)
    errors = np.abs(predictions - reference)
    result = {
        "mae": float(errors.mean()),
        "zero_action_mae": float(np.abs(targets).mean()),
        "clipped_mae": float(np.abs(np.clip(predictions, -1, 1) - reference).mean()),
        "best_of_samples_mae_optimistic": float(
            errors.mean(axis=(2, 3)).min(axis=1).mean()
        ),
        "between_sample_std": float(predictions.std(axis=1).mean()),
        "outside_command_range_fraction": float((np.abs(predictions) > 1).mean()),
        "deadbands": {},
        "per_action": {},
    }
    for threshold in (0.02, 0.05, 0.1):
        result["deadbands"][str(threshold)] = {
            "predicted_all_commands_idle": float(
                (np.abs(predictions) <= threshold).all(axis=-1).mean()
            ),
            "recorded_all_commands_idle": float(
                (np.abs(targets) <= threshold).all(axis=-1).mean()
            ),
            "predicted_no_drive_or_lift": float(
                (np.abs(predictions[..., [0, 2]]) <= threshold).all(axis=-1).mean()
            ),
            "recorded_no_drive_or_lift": float(
                (np.abs(targets[..., [0, 2]]) <= threshold).all(axis=-1).mean()
            ),
        }
    for index, name in enumerate(ACTIONS):
        predicted = predictions[..., index].reshape(-1)
        recorded = reference[..., index].reshape(-1)
        active = np.abs(recorded) > 0.05
        predicted_active = np.abs(predicted) > 0.05
        result["per_action"][name] = {
            "mae": float(np.abs(predicted - recorded).mean()),
            "zero_action_mae": float(np.abs(recorded).mean()),
            "active_mae": float(np.abs(predicted[active] - recorded[active]).mean())
            if active.any()
            else None,
            "active_zero_action_mae": float(np.abs(recorded[active]).mean())
            if active.any()
            else None,
            "recorded_active_fraction": float(active.mean()),
            "predicted_active_fraction": float(predicted_active.mean()),
            "missed_active_fraction": float((~predicted_active[active]).mean())
            if active.any()
            else None,
            "wrong_sign_on_active_fraction": float(
                (predicted[active] * recorded[active] < 0).mean()
            )
            if active.any()
            else None,
            "false_active_on_idle_fraction": float(predicted_active[~active].mean())
            if (~active).any()
            else None,
            "idle_predicted_abs_mean": float(np.abs(predicted[~active]).mean())
            if (~active).any()
            else None,
            "idle_predicted_abs_p90": float(
                np.quantile(np.abs(predicted[~active]), 0.9)
            )
            if (~active).any()
            else None,
            "predicted_abs_p50_p90_p99": np.quantile(
                np.abs(predicted), [0.5, 0.9, 0.99]
            ).tolist(),
            "recorded_abs_p50_p90_p99": np.quantile(
                np.abs(recorded), [0.5, 0.9, 0.99]
            ).tolist(),
            "correlation": float(np.corrcoef(predicted, recorded)[0, 1])
            if predicted.std() > 1e-9 and recorded.std() > 1e-9
            else None,
        }
    return result


def load_tokens(policy, job, sample, positions, device, image_intervention=None):
    data = {}
    for camera in policy.cameras:
        images = [
            simplejpeg.decode_jpeg(
                (
                    job
                    / camera
                    / "frames/256x144"
                    / f"{sample[camera + '/frame_idx'][position]:06d}.jpg"
                ).read_bytes(),
                colorspace="rgb",
                fastdct=True,
                fastupsample=True,
            )
            for position in positions
        ]
        images = np.stack(images)
        if image_intervention is not None:
            images = image_intervention(camera, images)
        data[camera] = (
            torch.from_numpy(np.ascontiguousarray(images)).unsqueeze(0).to(device)
        )
    for name in ("speed", *policy.observations):
        data[name] = torch.tensor(
            [sample[name][position] for position in positions],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    transformed = policy.input_transform({"data": data})
    images = torch.stack(
        [transformed["image"][camera] for camera in policy.cameras], dim=2
    )
    tokens = policy._frame_tokens(
        images,
        transformed["continuous"]["speed"],
        {name: transformed["continuous"][name] for name in policy.observations},
    )
    return list(tokens.split(1, dim=1))


def predict_job(
    policy,
    config,
    job,
    samples_count,
    seed,
    device,
    noise_mode="fresh",
    max_windows=None,
):
    from mcap.reader import make_reader

    stages = config.datamodule.val.dataset.samples.pipeline.functions
    builder, positioner, windower = [instantiate(stage.func) for stage in stages[:3]]
    rows = builder(job_dir=job)
    samples = windower(frames=positioner(row_table=rows))
    timestamps = dict(
        zip(
            rows["cam_fork/frame_idx"].to_list(),
            rows["log_time"].cast(pl.Int64).to_list(),
            strict=True,
        )
    )
    with (job / "data_for_training.mcap").open("rb") as handle:
        statistics = make_reader(handle).get_summary().statistics
    start_ns = statistics.message_start_time
    world = {row["cam_fork/frame_idx"]: row for row in rows.iter_rows(named=True)}
    decoder = D12FlowDecoderStep(policy).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    collected = {
        name: []
        for name in (
            "window",
            "stream",
            "target",
            "seconds",
            "frames",
            "height",
            "reset",
            "warm_parity",
            "recording_ns",
            "ego_xy",
            "pallet_xy",
            "dropoff_xy",
            "heading_deg",
            "fm_batch_ms",
            "stream_features",
            "window_features",
        )
    }
    previous = None
    frame_counter = 0
    noise = None
    for sample_index, sample in enumerate(samples.iter_rows(named=True)):
        if max_windows is not None and sample_index >= max_windows:
            break
        indices = tuple(sample["cam_fork/frame_idx"])
        reset = previous is None or indices[:-1] != previous[1:]
        if reset:
            token_window = load_tokens(policy, job, sample, range(6), device)
            cache = tuple(
                value.to(device)
                for value in policy.encoder.empty_cache(batch_size=1, cache_frames=5)
            )
            frame_counter = 0
            fresh_tokens = token_window
        else:
            fresh_tokens = load_tokens(policy, job, sample, [5], device)
            token_window = token_window[1:] + fresh_tokens
        for tokens in fresh_tokens:
            cosine, sine = (value.to(device) for value in decoder.rope(frame_counter))
            encoded, new_k, new_v = policy.encoder.step(
                tokens[:, 0],
                past_k=cache[0],
                past_v=cache[1],
                cache_bias=cache[2],
                cos=cosine,
                sin=sine,
                readout_only_final_block=True,
            )
            stream_feature = encoded[:, -1]
            if policy.norm is not None:
                stream_feature = policy.norm(stream_feature)
            cache = PatchPolicyDecoderStep.advance(cache, new_k, new_v)
            frame_counter += 1
        window_tokens = torch.cat(token_window, dim=1).flatten(1, 2)
        window_feature = policy.encoder(window_tokens, num_frames=6)[:, -1]
        if policy.norm is not None:
            window_feature = policy.norm(window_feature)
        collected["stream_features"].append(stream_feature[0].cpu().numpy())
        collected["window_features"].append(window_feature[0].cpu().numpy())
        if noise_mode == "fresh" or reset:
            noise = torch.randn(
                samples_count,
                policy.action_horizon * 3,
                generator=generator,
                device=device,
            )
        elif noise_mode == "shift":
            tail = torch.randn(samples_count, 6, 3, generator=generator, device=device)
            noise = torch.cat(
                [noise.reshape(samples_count, -1, 3)[:, 6:], tail], dim=1
            ).flatten(1)
        predictions = {}
        for mode, features in (("window", window_feature), ("stream", stream_feature)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            actions = policy.fm_sample(features.expand(samples_count, -1), x0=noise)
            torch.cuda.synchronize()
            if mode == "stream":
                collected["fm_batch_ms"].append((time.perf_counter() - started) * 1000)
            predictions[mode] = (
                (actions * policy.action_std + policy.action_mean).cpu().numpy()
            )
            collected[mode].append(predictions[mode])
        if reset:
            error = float(np.abs(predictions["window"] - predictions["stream"]).max())
            if error > 1e-4:
                raise RuntimeError(f"warm-cache parity failed: {error}")
            collected["warm_parity"].append(error)
        collected["target"].append(
            np.stack([sample[name][-1] for name in ACTIONS], axis=-1)
        )
        collected["seconds"].append((timestamps[indices[-1]] - start_ns) / 1e9)
        collected["recording_ns"].append(timestamps[indices[-1]])
        row = world[indices[-1]]
        for name, prefix in (
            ("ego_xy", "ego"),
            ("pallet_xy", "pallet"),
            ("dropoff_xy", "target"),
        ):
            collected[name].append([row[prefix + "_x"], row[prefix + "_y"]])
        collected["heading_deg"].append(row["ego_heading"])
        collected["frames"].append([
            sample[camera + "/frame_idx"][-1] for camera in policy.cameras
        ])
        collected["height"].append(sample["fork_height"][-1])
        collected["reset"].append(reset)
        previous = indices
        if sample_index % 50 == 0:
            print(
                json.dumps({
                    "job": job.name,
                    "windows": sample_index + 1,
                    "total": len(samples),
                }),
                flush=True,
            )
    result = {name: np.asarray(values) for name, values in collected.items()}
    result["recording_bounds_ns"] = np.array(
        [start_ns, statistics.message_end_time], dtype=np.int64
    )
    return result


def render(directory, job, arrays, cameras):
    times = (arrays["seconds"][:, None] + np.arange(6)[None] / 30).reshape(-1)
    target = arrays["target"][:, :6].reshape(-1, 3)
    stream = arrays["stream"][:, 0, :6].reshape(-1, 3)
    window = arrays["window"][:, 0, :6].reshape(-1, 3)
    lower = np.quantile(arrays["stream"][:, :, :6], 0.1, axis=1).reshape(-1, 3)
    upper = np.quantile(arrays["stream"][:, :, :6], 0.9, axis=1).reshape(-1, 3)
    figure, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for index, name in enumerate(ACTIONS):
        axis = axes[index]
        axis.plot(times, target[:, index], color="black", linewidth=1, label="recorded")
        axis.plot(
            times,
            stream[:, index],
            color="tab:blue",
            linewidth=0.8,
            label="KV sample 0",
        )
        axis.plot(
            times,
            window[:, index],
            color="tab:orange",
            linewidth=0.7,
            alpha=0.8,
            label="6-frame sample 0",
        )
        axis.fill_between(
            times,
            lower[:, index],
            upper[:, index],
            alpha=0.15,
            color="tab:blue",
            label="KV sample 10–90%",
        )
        axis.axhspan(-0.05, 0.05, color="gray", alpha=0.12)
        axis.set_ylabel(name)
        axis.set_ylim(-1.1, 1.1)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Recording time (seconds); first 6 actions of each 5 Hz replan")
    axes[0].set_title(job.name + " — recorded observations, not closed-loop")
    figure.tight_layout()
    figure.canvas.draw()
    cursor_x = [
        axes[0].transData.transform((stamp, 0))[0] / figure.bbox.width * 768
        for stamp in arrays["seconds"]
    ]
    cursor_y = [
        (1 - axis.get_position().y1, 1 - axis.get_position().y0) for axis in axes
    ]
    figure.savefig(directory / "timeline.png", dpi=130)
    plt.close(figure)
    with (directory / "timeline.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "seconds",
            *[
                f"{mode}_{name}"
                for mode in ("target", "stream_sample0", "window_sample0")
                for name in ACTIONS
            ],
        ])
        writer.writerows(np.column_stack([times, target, stream, window]).tolist())
    overview = Image.open(directory / "timeline.png").convert("RGB").resize((768, 384))
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "768x784",
        "-framerate",
        "5",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        str(directory / "openloop.mp4"),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for position, stamp in enumerate(arrays["seconds"]):
            canvas = Image.new("RGB", (768, 784), "white")
            draw = ImageDraw.Draw(canvas)
            for camera_index, camera in enumerate(cameras):
                frame = int(arrays["frames"][position, camera_index])
                with Image.open(
                    job / camera / "frames/256x144" / f"{frame:06d}.jpg"
                ) as picture:
                    canvas.paste(picture.convert("RGB"), (camera_index * 256, 20))
                draw.text((camera_index * 256 + 4, 3), camera, fill="black")
            height = arrays["height"][position]
            draw.text(
                (8, 170),
                f"recording t={stamp:.2f}s | frame={arrays['frames'][position, 0]} | height valid={int(height[2])}, estimate={height[0] * 2.3:.2f}m",
                fill="black",
            )
            draw.text(
                (8, 188),
                f"H{arrays['target'].shape[1]}: black=recorded; colors=all {arrays['stream'].shape[1]} KV samples (not mode IDs); green=next 0.2s",
                fill="black",
            )
            for action_index, name in enumerate(ACTIONS):
                top = 212 + action_index * 58
                draw.text((6, top + 15), name, fill="black")
                draw.rectangle(
                    (
                        90,
                        top,
                        90 + 6 * 665 / max(arrays["target"].shape[1] - 1, 1),
                        top + 48,
                    ),
                    fill="#edfaed",
                )
                draw.line((90, top + 24, 755, top + 24), fill="#bbbbbb")
                curves = [
                    (
                        values[:, action_index],
                        tuple(
                            int(channel * 255)
                            for channel in plt.get_cmap("tab10")(sample_id % 10)[:3]
                        ),
                    )
                    for sample_id, values in enumerate(arrays["stream"][position])
                ]
                curves.append((arrays["target"][position, :, action_index], "black"))
                for values, color in curves:
                    points = [
                        (
                            90 + step * 665 / max(len(values) - 1, 1),
                            top + 24 - float(np.clip(value, -1, 1)) * 22,
                        )
                        for step, value in enumerate(values)
                    ]
                    draw.line(points, fill=color, width=2)
            canvas.paste(overview, (0, 400))
            for lower_y, upper_y in cursor_y:
                draw.line(
                    (
                        cursor_x[position],
                        400 + lower_y * 384,
                        cursor_x[position],
                        400 + upper_y * 384,
                    ),
                    fill="#d40000",
                    width=2,
                )
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video rendering failed")


def summarize(arrays):
    result = {
        "windows": len(arrays["target"]),
        "cache_resets": int(arrays["reset"].sum()),
        "height_valid_fraction": float(arrays["height"][:, 2].mean()),
    }
    for mode in ("window", "stream"):
        result[mode] = {
            name: action_metrics(
                arrays[mode][:, :, :length], arrays["target"][:, :length]
            )
            for name, length in (
                ("first", 1),
                ("near", 6),
                ("full", arrays["target"].shape[1]),
            )
        }
    result["stream_vs_window_near_mae"] = float(
        np.abs(arrays["stream"][:, :, :6] - arrays["window"][:, :, :6]).mean()
    )
    if "fm_batch_ms" in arrays:
        result["fm_head_batch_ms_median"] = float(np.median(arrays["fm_batch_ms"]))
    result["replan"] = replan_metrics(arrays)
    result["hold_events"] = hold_events(arrays)
    return result


def hold_events(arrays):
    result = {}
    boundaries = np.r_[0, np.flatnonzero(arrays["reset"])[1:], len(arrays["target"])]
    for action_index, name in enumerate(ACTIONS):
        durations = []
        hold_count = 0
        holds_with_activation = 0
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            target = arrays["target"][start:stop, :6, action_index].reshape(-1)
            predicted = (
                arrays["stream"][start:stop, :, :6, action_index]
                .transpose(1, 0, 2)
                .reshape(arrays["stream"].shape[1], -1)
            )
            idle = np.abs(target) <= 0.05
            edges = np.diff(np.r_[False, idle, False].astype(int))
            for beginning, ending in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
            ):
                if ending - beginning < 6:
                    continue
                for sample in predicted:
                    false_active = np.abs(sample[beginning:ending]) > 0.05
                    hold_count += 1
                    holds_with_activation += int(false_active.any())
                    changes = np.diff(np.r_[False, false_active, False].astype(int))
                    durations.extend(
                        (
                            (
                                np.flatnonzero(changes == -1)
                                - np.flatnonzero(changes == 1)
                            )
                            / 30
                        ).tolist()
                    )
        result[name] = {
            "hold_sample_pairs": hold_count,
            "hold_minimum_seconds": 0.2,
            "holds_with_any_false_activation_fraction": holds_with_activation
            / hold_count
            if hold_count
            else None,
            "false_activation_duration_seconds_p50_p95_max": np.quantile(
                durations, [0.5, 0.95, 1]
            ).tolist()
            if durations
            else [0.0, 0.0, 0.0],
        }
    return result


def resample_features(policy, arrays, samples_count, seed, noise_mode, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    result = dict(arrays)
    outputs = {"stream": [], "window": [], "fm_batch_ms": []}
    noise = None
    with torch.inference_mode():
        for position, reset in enumerate(arrays["reset"]):
            if noise_mode == "fresh" or reset:
                noise = torch.randn(
                    samples_count,
                    policy.action_horizon * 3,
                    generator=generator,
                    device=device,
                )
            elif noise_mode == "shift":
                tail = torch.randn(
                    samples_count, 6, 3, generator=generator, device=device
                )
                noise = torch.cat(
                    [noise.reshape(samples_count, -1, 3)[:, 6:], tail], dim=1
                ).flatten(1)
            for mode in ("window", "stream"):
                feature = torch.as_tensor(
                    arrays[mode + "_features"][position], device=device
                ).expand(samples_count, -1)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = policy.fm_sample(feature, x0=noise)
                torch.cuda.synchronize()
                if mode == "stream":
                    outputs["fm_batch_ms"].append(
                        (time.perf_counter() - started) * 1000
                    )
                outputs[mode].append(
                    (prediction * policy.action_std + policy.action_mean).cpu().numpy()
                )
    result.update({key: np.asarray(value) for key, value in outputs.items()})
    reset_error = (
        np
        .abs(result["stream"][arrays["reset"]] - result["window"][arrays["reset"]])
        .reshape(int(arrays["reset"].sum()), -1)
        .max(axis=1)
    )
    if reset_error.max() > 1e-4:
        raise RuntimeError("resampled warm-cache parity failed")
    result["warm_parity"] = reset_error
    return result


def replan_metrics(arrays):
    valid = (~arrays["reset"][1:]) & (np.abs(np.diff(arrays["seconds"]) - 0.2) < 0.04)
    result = {"pairs": int(valid.sum())}
    if not valid.any():
        return result
    for index, name in enumerate(ACTIONS):
        previous = arrays["stream"][:-1, :, 6, index][valid]
        current = arrays["stream"][1:, :, 0, index][valid]
        boundary = arrays["stream"][:-1, :, 5, index][valid]
        result[name] = {
            "overlap_revision_mean": float(np.abs(current - previous).mean()),
            "overlap_revision_p95": float(
                np.quantile(np.abs(current - previous), 0.95)
            ),
            "boundary_jump_mean": float(np.abs(current - boundary).mean()),
            "strong_sign_flip_fraction": float(
                (
                    (current * previous < 0)
                    & (np.abs(current) > 0.1)
                    & (np.abs(previous) > 0.1)
                ).mean()
            ),
        }
    return result


def publish(directory, report, parent_run):
    import wandb

    run = wandb.init(
        entity="yaak",
        project="rmind-palleter",
        job_type="openloop-eval",
        name=f"openloop-{parent_run}-{directory.name}",
        group=f"d12-fm-{parent_run}",
        tags=[
            "d12",
            "flow_matching",
            "openloop",
            f"cameras:{len(report.get('cameras', [1, 2, 3]))}",
            "kv_cache",
        ],
        config={
            key: value
            for key, value in report.items()
            if key not in ("jobs", "aggregate")
        }
        | {"parent_run": parent_run},
    )
    table = wandb.Table(
        columns=[
            "job",
            "split",
            "windows",
            "window_near_mae",
            "stream_near_mae",
            "zero_near_mae",
            "predicted_idle",
            "recorded_idle",
        ]
    )
    for job, summary in report["jobs"].items():
        metrics = summary["stream"]["near"]
        deadband = metrics["deadbands"]["0.05"]
        table.add_data(
            job,
            summary["split"],
            summary["windows"],
            summary["window"]["near"]["mae"],
            metrics["mae"],
            metrics["zero_action_mae"],
            deadband["predicted_all_commands_idle"],
            deadband["recorded_all_commands_idle"],
        )
        if (directory / job / "openloop.mp4").exists():
            run.log({
                f"{summary['split']}/{job}/video": wandb.Video(
                    str(directory / job / "openloop.mp4"), format="mp4"
                )
            })
        if (directory / job / "timeline.png").exists():
            run.log({
                f"{summary['split']}/{job}/timeline": wandb.Image(
                    str(directory / job / "timeline.png")
                )
            })
    run.log({"episodes": table})
    for split, summary in report["aggregate"].items():
        for mode in ("window", "stream"):
            for name in ("mae", "zero_action_mae", "between_sample_std"):
                run.summary[f"{split}/{mode}/near/{name}"] = summary[mode]["near"][name]
    artifact = wandb.Artifact(
        f"d12-openloop-{parent_run}-{directory.name}",
        type="evaluation",
        metadata={
            "checkpoint_sha256": report["checkpoint_sha256"],
            "parent_run": parent_run,
        },
    )
    artifact.add_dir(str(directory))
    for filename in (
        "eval_d12_fm.py",
        "d12_fm_rerun.py",
        "research_d12_fm.py",
        "probe_d12_fm.py",
    ):
        artifact.add_file(
            str(Path(__file__).with_name(filename)), name=f"source/{filename}"
        )
    run.log_artifact(artifact)
    location = {"id": run.id, "url": run.url}
    run.finish()
    (directory / "wandb.json").write_text(json.dumps(location, indent=2) + "\n")
    return location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--wandb-parent-run")
    parser.add_argument("--fm-steps", type=int)
    parser.add_argument(
        "--noise-mode", choices=("fresh", "fixed", "shift"), default="fresh"
    )
    parser.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--reuse-features", type=Path)
    arguments = parser.parse_args()
    if arguments.samples < 2:
        raise ValueError("at least two noise samples required")
    if arguments.max_windows is not None and arguments.max_windows < 1:
        raise ValueError("max-windows must be positive")
    if len(set(arguments.jobs)) != len(arguments.jobs):
        raise ValueError("jobs must be unique")
    directory = arguments.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    config = OmegaConf.load(arguments.config)
    if arguments.render_only:
        report = json.loads((directory / "report.json").read_text())
        for job_name in report["jobs"]:
            destination = directory / job_name
            with np.load(destination / "predictions.npz") as saved:
                arrays = dict(saved)
            if not arguments.no_video:
                render(
                    destination,
                    Path(config.paths.d12) / job_name,
                    arrays,
                    report["cameras"],
                )
            if arguments.rerun:
                from rmind.scripts.d12_fm_rerun import render_rerun

                render_rerun(
                    destination,
                    Path(config.paths.d12) / job_name,
                    arrays,
                    report["cameras"],
                )
        return
    if (directory / "report.json").exists():
        raise ValueError(
            "output already contains an evaluation; use a new output or --render-only"
        )
    policy = (
        PatchPolicyContinuous
        .load_from_checkpoint(
            arguments.checkpoint, map_location="cpu", weights_only=False
        )
        .eval()
        .cuda()
    )
    if arguments.fm_steps is not None:
        if arguments.fm_steps < 1:
            raise ValueError("FM steps must be positive")
        policy.fm_inference_steps = arguments.fm_steps
    split_sets = {
        split: {
            Path(path).name
            for path in config.datamodule[split].dataset.samples.inputs.job_dir
        }
        for split in ("train", "val")
    }
    if split_sets["train"] & split_sets["val"]:
        raise ValueError("overlapping train/validation jobs")
    with arguments.checkpoint.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if arguments.reuse_features:
        source_report = json.loads(
            (arguments.reuse_features / "report.json").read_text()
        )
        if source_report["checkpoint_sha256"] != checksum or source_report[
            "config"
        ] != OmegaConf.to_container(config, resolve=True):
            raise ValueError(
                "cached features must match the exact checkpoint and data configuration"
            )
    report = {
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": checksum,
        "seed": arguments.seed,
        "samples": arguments.samples,
        "precision": "FP32/no-TF32",
        "action_order": ACTIONS,
        "fm_steps": policy.fm_inference_steps,
        "cameras": list(policy.cameras),
        "noise_mode": arguments.noise_mode,
        "reused_features": str(arguments.reuse_features)
        if arguments.reuse_features
        else None,
        "config": OmegaConf.to_container(config, resolve=True),
        "jobs": {},
        "limitations": "Recorded observations only, not a closed-loop success test. Gaussian samples are not discovered mode IDs; no best-sample selection for primary metrics. Deadbands are normalized-command heuristics, not physical motion thresholds. Streaming KV can carry indirect context beyond the six-frame training window. Videos use a fixed 5 Hz playback grid; CSV/Rerun contain recorded timestamps. FM batch timing excludes vision/trunk/transfers and is not serving latency. Spatial views show recorded poses, not inferred motion from action commands.",
    }
    pooled = {"train": [], "val": []}
    for job_name in arguments.jobs:
        split = "train" if job_name in split_sets["train"] else "val"
        if job_name not in split_sets[split]:
            raise ValueError(f"{job_name} is not in {split}")
        job = Path(config.paths.d12) / job_name
        destination = directory / job_name
        destination.mkdir(exist_ok=True)
        with torch.inference_mode():
            seed = arguments.seed + int.from_bytes(
                hashlib.sha256(job_name.encode()).digest()[:4], "little"
            )
            if arguments.reuse_features:
                with np.load(
                    arguments.reuse_features / job_name / "predictions.npz"
                ) as saved:
                    arrays = resample_features(
                        policy,
                        dict(saved),
                        arguments.samples,
                        seed,
                        arguments.noise_mode,
                        "cuda:0",
                    )
            else:
                arrays = predict_job(
                    policy,
                    config,
                    job,
                    arguments.samples,
                    seed,
                    "cuda:0",
                    noise_mode=arguments.noise_mode,
                    max_windows=arguments.max_windows,
                )
        np.savez_compressed(destination / "predictions.npz", **arrays)
        summary = summarize(arrays)
        summary["split"] = split
        report["jobs"][job_name] = summary
        pooled[split].append(arrays)
        (destination / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if not arguments.no_video:
            render(destination, job, arrays, policy.cameras)
        if arguments.rerun:
            from rmind.scripts.d12_fm_rerun import render_rerun

            render_rerun(destination, job, arrays, policy.cameras)
        (directory / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    report["aggregate"] = {
        split: summarize({
            key: np.concatenate([arrays[key] for arrays in values])
            for key in (
                "target",
                "stream",
                "window",
                "reset",
                "seconds",
                "height",
                "fm_batch_ms",
            )
        })
        for split, values in pooled.items()
        if values
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    sections = [
        "<h1>D12 flow matching: open-loop replay</h1>",
        "<p>" + html.escape(report["limitations"]) + "</p>",
        "<p>Checkpoint: " + html.escape(report["checkpoint"]) + "</p>",
    ]
    sections.append(
        "<table border='1' cellpadding='6'><tr><th>Job</th><th>Split</th><th>Window near MAE</th><th>KV near MAE</th><th>Zero-action MAE</th><th>KV idle</th><th>Recorded idle</th></tr>"
    )
    for job, summary in report["jobs"].items():
        near = summary["stream"]["near"]
        deadband = near["deadbands"]["0.05"]
        sections.append(
            f"<tr><td>{job}</td><td>{summary['split']}</td><td>{summary['window']['near']['mae']:.4f}</td><td>{near['mae']:.4f}</td><td>{near['zero_action_mae']:.4f}</td><td>{deadband['predicted_all_commands_idle']:.1%}</td><td>{deadband['recorded_all_commands_idle']:.1%}</td></tr>"
        )
    sections.append("</table>")
    for job in report["jobs"]:
        sections.append(
            f'<h2>{job}</h2><video controls preload="metadata" width="768" src="{job}/openloop.mp4"></video><br><img width="1100" src="{job}/timeline.png"><p><a href="{job}/metrics.json">Metrics</a> | <a href="{job}/timeline.csv">Timeline CSV</a> | <a href="{job}/openloop.rrd">Rerun recording</a></p>'
        )
    (directory / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>D12 FM open-loop</title>"
        + "\n".join(sections)
    )
    if arguments.wandb_parent_run:
        publish(directory, report, arguments.wandb_parent_run)
    print(
        json.dumps(
            {
                split: {
                    mode: report["aggregate"][split][mode]["near"]
                    for mode in ("window", "stream")
                }
                for split in report["aggregate"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
