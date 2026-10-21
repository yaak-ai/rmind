"""Paired recorded-state D12 evaluation; never a physical counterfactual rollout."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.scripts.eval_d12_fm import action_metrics, predict_job

SEEDS = (131, 137, 139)
HEIGHT_VALID_THRESHOLD = 0.5
ACTIVE_COMMAND_THRESHOLD = 0.05
BASE = Path("/nasa/max/d12/flow_matching")
MODELS = {
    "baseline": BASE / "2026-10-14-h50-1cam",
    "rtc": BASE / "2026-10-18-d12-rtc-native-fork-h50",
}
CHECKPOINT = "checkpoints/epoch=5-step=35000.ckpt"
SEGMENTS = (
    (
        "stationary_40s",
        "morning-aligned",
        "2026-10-15T09:40:19.765+00:00",
        "2026-10-15T09:41:00.066+00:00",
    ),
    (
        "stationary_39s",
        "morning-aligned",
        "2026-10-15T09:42:27.869+00:00",
        "2026-10-15T09:43:06.670+00:00",
    ),
    (
        "successful_77s",
        "success",
        "2026-10-15T12:45:05.035+00:00",
        "2026-10-15T12:46:22.135+00:00",
    ),
)


def sha256(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path | str, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def setup() -> None:
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_policy(
    protocol: dict[str, Any], name: str, device: str
) -> PatchPolicyContinuous:
    record = protocol["models"][name]
    if sha256(record["checkpoint"]) != record["sha256"]:
        msg = "Checkpoint changed before loading"
        raise ValueError(msg)
    return (
        PatchPolicyContinuous
        .load_from_checkpoint(
            record["checkpoint"], map_location="cpu", weights_only=False
        )
        .eval()
        .to(device)
    )


def load_config(protocol: dict[str, Any], name: str) -> DictConfig:
    record = protocol["models"][name]
    if sha256(record["config"]) != record["config_sha256"]:
        msg = "Configuration changed before loading"
        raise ValueError(msg)
    return cast("DictConfig", OmegaConf.load(record["config"]))


def explicit_noise(seed: int, identity: str, count: int) -> np.ndarray:
    identity_seed = int.from_bytes(
        hashlib.sha256(identity.encode()).digest()[:4], "little"
    )
    return np.random.default_rng(
        np.random.SeedSequence([seed, identity_seed])
    ).standard_normal((count, 50, 3), dtype=np.float32)


def freeze(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(MODELS["rtc"] / "resolved.yaml")
    write_json(
        directory / "protocol.json",
        {
            "schema": "d12_rtc_offline_v1",
            "seeds": SEEDS,
            "models": {
                name: {
                    "checkpoint": str(path / CHECKPOINT),
                    "sha256": sha256(path / CHECKPOINT),
                    "config": str(path / "resolved.yaml"),
                    "config_sha256": sha256(path / "resolved.yaml"),
                }
                for name, path in MODELS.items()
            },
            "expert_jobs": list(config.datamodule.val.dataset.samples.inputs.job_dir),
            "segments": SEGMENTS,
            "expert_metrics": "Per-action unconditional MAE full/first6, height-valid split, command-defined hold/lift/lower strata; RTC target-prefix3/6 suffix metrics separately oracle-conditioned, not rollout performance",
            "async": {
                "observation_hz": 5,
                "action_hz": 30,
                "horizon": 50,
                "seeds": SEEDS,
                "profiles": ["zero", "measured", "overrun"],
                "arms": ["baseline", "rtc_unconditioned", "rtc"],
                "max_prefix": 6,
                "initial_forecast": 3,
                "freshness_ns": 400000000,
                "overrun_definition": "Measured trace except every50th request250ms, every100th450ms; causal last-completed forecast; failures latch until recorded generation reset",
                "controller": "Unchanged Coppernic RtcController via persistent Rust JSONL wrapper; baseline/unconditioned max_prefix0",
                "gains": [1.0, 1.0, 1.0],
            },
            "latency_source": str(
                BASE / "rtc-robustness-20261016/d12-input-output-latency.json"
            ),
            "latency_sha256": sha256(
                BASE / "rtc-robustness-20261016/d12-input-output-latency.json"
            ),
            "proxy_thresholds": [0.05, 0.10, 0.15],
            "sustained_traction_ticks": 15,
            "interpretation": "Descriptive paired offline comparison, not closed-loop success/physics. Historical failed commands are not targets. No missed-pocket annotation invented. Three Gaussian seeds are not independent training seeds. No automatic deployment gate.",
        },
    )


def verify(directory: Path) -> dict[str, Any]:
    if not __debug__:
        msg = "Evaluation audits require Python without -O"
        raise RuntimeError(msg)
    protocol = json.loads((directory / "protocol.json").read_text())
    for model in protocol["models"].values():
        if sha256(model["checkpoint"]) != model["sha256"]:
            msg = f"Checkpoint changed: {model['checkpoint']}"
            raise ValueError(msg)
        if sha256(model["config"]) != model["config_sha256"]:
            msg = f"Configuration changed: {model['config']}"
            raise ValueError(msg)
    if sha256(protocol["latency_source"]) != protocol["latency_sha256"]:
        msg = "Latency source changed"
        raise ValueError(msg)
    return protocol


def expert(directory: Path, name: str, device: str) -> None:
    protocol = verify(directory)
    output = directory / "expert" / name
    output.mkdir(parents=True, exist_ok=True)
    policy = load_policy(protocol, name, device)
    if policy.fm_head is None:
        msg = "D12 expert evaluation requires a flow-matching head"
        raise ValueError(msg)
    config = load_config(protocol, name)
    report = {}
    with torch.inference_mode():
        for job_name in protocol["expert_jobs"]:
            job = Path(job_name)
            arrays = predict_job(policy, config, job, 2, 131, device)
            del arrays["window"], arrays["stream"]
            count = len(arrays["target"])
            noises = np.stack(
                [explicit_noise(seed, job.name, count) for seed in SEEDS], axis=1
            )
            predictions = {
                prefix: [] for prefix in ((0, 3, 6) if name == "rtc" else (0,))
            }
            flow_mse = {prefix: [] for prefix in predictions}
            for start in range(0, count, 32):
                stop = min(start + 32, count)
                features = torch.as_tensor(
                    arrays["stream_features"][start:stop], device=device
                )
                target = torch.as_tensor(
                    arrays["target"][start:stop], device=device, dtype=torch.float32
                )
                normalized = (target - policy.action_mean) / policy.action_std
                features = features[:, None].expand(-1, len(SEEDS), -1).flatten(0, 1)
                target = (
                    normalized[:, None].expand(-1, len(SEEDS), -1, -1).flatten(0, 1)
                )
                noise = torch.as_tensor(noises[start:stop], device=device).flatten(0, 1)
                times = torch.full((len(features),), 0.5, device=device)
                for prefix, prefix_predictions in predictions.items():
                    mask = (
                        (torch.arange(50, device=device)[None] < prefix)
                        .expand(len(features), -1)
                        .float()
                    )
                    kwargs = (
                        {"clean_prefix": target * mask[..., None], "prefix_mask": mask}
                        if name == "rtc"
                        else {}
                    )
                    prediction = policy.fm_sample(
                        features, x0=noise.flatten(1), **kwargs
                    )
                    raw = prediction * policy.action_std + policy.action_mean
                    prefix_predictions.append(
                        raw.reshape(stop - start, len(SEEDS), 50, 3).cpu().numpy()
                    )
                    if name == "rtc":
                        token_time = torch.where(mask.bool(), 1.0, times[:, None])
                        state = (1 - token_time[..., None]) * noise + token_time[
                            ..., None
                        ] * target
                        velocity = policy.fm_head.velocity(features, state, token_time)
                    else:
                        state = 0.5 * noise + 0.5 * target
                        velocity = policy.fm_head.velocity(
                            features, state.flatten(1), times
                        ).reshape_as(target)
                    error = (
                        (velocity[:, prefix:] - (target - noise)[:, prefix:])
                        .square()
                        .mean(dim=1)
                    )
                    flow_mse[prefix].append(
                        error.reshape(stop - start, len(SEEDS), 3).cpu().numpy()
                    )
            predictions = {
                prefix: np.concatenate(values) for prefix, values in predictions.items()
            }
            metrics = {}
            height_valid = arrays["height"][:, 2] > HEIGHT_VALID_THRESHOLD
            strata = {
                "all": np.ones(count, dtype=bool),
                "height_valid": height_valid,
                "height_missing": ~height_valid,
                "hold_command": (
                    np.abs(arrays["target"][:, :6, [0, 2]]) <= ACTIVE_COMMAND_THRESHOLD
                ).all(axis=(1, 2)),
                "lift_command": (
                    arrays["target"][:, :6, 2] > ACTIVE_COMMAND_THRESHOLD
                ).any(axis=1),
                "lower_command": (
                    arrays["target"][:, :6, 2] < -ACTIVE_COMMAND_THRESHOLD
                ).any(axis=1),
            }
            for prefix, values in predictions.items():
                for stratum, selection in strata.items():
                    if selection.any():
                        for horizon, stop in (
                            ("full", 50),
                            ("near", min(prefix + 6, 50)),
                        ):
                            metrics[f"prefix{prefix}/{stratum}/{horizon}"] = (
                                action_metrics(
                                    values[selection, :, prefix:stop],
                                    arrays["target"][selection, prefix:stop],
                                )
                            )
            report[job.name] = {
                "windows": count,
                "stratum_counts": {
                    key: int(value.sum()) for key, value in strata.items()
                },
                "metrics": metrics,
                "normalized_flow_mse_at_t_half_per_action": {
                    str(prefix): np.concatenate(values).mean(axis=(0, 1)).tolist()
                    for prefix, values in flow_mse.items()
                },
            }
            np.savez_compressed(
                output / f"{job.name}.npz",
                **arrays,
                noise=noises,
                **{
                    f"prediction_prefix{prefix}": values
                    for prefix, values in predictions.items()
                },
                allow_pickle=False,
            )
            print(
                json.dumps({"model": name, "job": job.name, "windows": count}),
                flush=True,
            )
    write_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "expert"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    setup()
    if args.phase == "freeze":
        freeze(args.output)
    else:
        if args.model is None:
            parser.error("expert requires --model")
        expert(args.output, args.model, args.device)


if __name__ == "__main__":
    main()
