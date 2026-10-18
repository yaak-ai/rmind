"""Recorded images with the actual Coppernic RTC controller, without actuation."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from rmind.models.d12_flow_decoder import D12FlowDecoderStep
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep
from rmind.scripts.evaluate_d12_rtc import (
    BASE,
    SEGMENTS,
    SEEDS,
    explicit_noise,
    load_policy,
    setup,
    sha256,
    verify,
    write_json,
)
from rmind.scripts.replay_d12_rollouts import AUX_NAMES, video_images


def tick_ns(tick):
    return (int(tick) * 1_000_000_000 + 29) // 30


class Controller:
    def __init__(self, binary, max_prefix):
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        config = dict(
            horizon=50,
            execution_horizon=6,
            action_hz=30,
            action_dim=3,
            max_prefix=max_prefix,
            initial_forecast=min(3, max_prefix),
            observation_max_age_ns=400_000_000,
            gains=[1.0, 1.0, 1.0],
        )
        self.process.stdin.write(json.dumps(config) + "\n")
        self.process.stdin.flush()
        self.events = []

    def call(self, event):
        self.process.stdin.write(json.dumps(event, allow_nan=False) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("RTC subprocess terminated")
        result = json.loads(response)
        self.events.append({"input": event, "output": result})
        return result

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=5):
            raise RuntimeError("RTC subprocess failed")


def extract(directory, model, device):
    protocol = verify(directory)
    policy = load_policy(protocol, model, device)
    decoder = D12FlowDecoderStep(policy).to(device).eval()
    root = BASE / "rollouts/2026-10-15"
    output = directory / "tokens" / model
    output.mkdir(parents=True, exist_ok=True)
    for replay, job in (
        ("morning-aligned", "2026-10-15--09-33-42"),
        ("success", "2026-10-15--12-43-19"),
    ):
        rows = json.loads((root / "replay" / replay / "inputs.json").read_text())
        images = iter(video_images(root / "raw" / job, rows, 300))
        cached_frame, cached_image = None, None
        tokens = np.lib.format.open_memmap(
            output / f"{replay}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 149, 512),
        )
        image_array = np.lib.format.open_memmap(
            output / f"{replay}-images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows), 144, 256, 3),
        )
        with torch.inference_mode():
            for index, row in enumerate(rows):
                if cached_frame != row["frame"]:
                    cached_frame, cached_image = next(images)
                if cached_frame != row["frame"]:
                    raise ValueError("PTS video ordering mismatch")
                image_array[index] = cached_image
                image = (
                    torch.as_tensor(
                        cached_image.transpose(2, 0, 1).copy(), device=device
                    ).float()[None, None]
                    / 255
                )
                normalized = (
                    image[:, :, None] - decoder.image_mean
                ) / decoder.image_std
                aux = {
                    key: torch.tensor(row["aux"][key], device=device).reshape(1, 1, -1)
                    for key in AUX_NAMES
                }
                value = policy._frame_tokens(
                    normalized, aux["speed"], {key: aux[key] for key in AUX_NAMES[1:]}
                )
                tokens[index] = value[0, 0].cpu().numpy()
                if index % 200 == 0:
                    print(
                        json.dumps({
                            "model": model,
                            "recording": replay,
                            "tokens": index,
                        }),
                        flush=True,
                    )
        tokens.flush()
        image_array.flush()
        write_json(
            output / f"{replay}.json",
            {
                "rows": rows,
                "rows_source": str(root / "replay" / replay / "inputs.json"),
                "rows_sha256": sha256(root / "replay" / replay / "inputs.json"),
                "tokens_sha256": sha256(output / f"{replay}.npy"),
                "model": model,
                "model_artifacts": protocol["models"][model],
                "images_sha256": sha256(output / f"{replay}-images.npy"),
                "videos": {
                    str(root / "raw" / job / f"cam_fork--{chunk}.mp4"): sha256(
                        root / "raw" / job / f"cam_fork--{chunk}.mp4"
                    )
                    for chunk in sorted({row["frame"] // 300 for row in rows})
                },
                "preprocessing": "Existing PTS mapping, lossy MP4, half-pixel uint8 576x324 then256x144; approximate serving images",
            },
        )


def load_tokens(path, model_artifacts):
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("model_artifacts") != model_artifacts:
        raise ValueError(
            "Token model provenance differs or is missing; re-extract tokens"
        )
    if sha256(path) != metadata["tokens_sha256"]:
        raise ValueError("Token array changed")
    return np.load(path, mmap_mode="r"), metadata


def sample_plan(policy, feature, noise, request, conditional):
    kwargs = {}
    if conditional:
        prefix = torch.tensor(request["prefix"], device=feature.device)[None]
        mask = torch.tensor(request["mask"], device=feature.device)[None]
        kwargs = {
            "clean_prefix": (prefix - policy.action_mean) / policy.action_std,
            "prefix_mask": mask,
        }
    predicted = policy.fm_sample(
        feature, x0=torch.tensor(noise, device=feature.device).reshape(1, 150), **kwargs
    )
    return (predicted * policy.action_std + policy.action_mean)[0].cpu().numpy()


def replay_one(
    policy,
    tokens,
    rows,
    segment,
    arm,
    profile,
    seed,
    latency,
    binary,
    device,
    clock="fixed",
):
    start = int(datetime.fromisoformat(segment[2]).timestamp() * 1e9)
    end = int(datetime.fromisoformat(segment[3]).timestamp() * 1e9)
    origin = start - 1_000_000_000
    walls = np.asarray([row["wall_ns"] for row in rows], dtype=np.int64)
    total = int((end - origin) * 30 // 1_000_000_000) + 1
    noise = explicit_noise(seed, segment[0], (total + 5) // 6)
    controller = Controller(binary, 6 if arm == "rtc" else 0)
    decoder = D12FlowDecoderStep(policy).to(device).eval()
    generation, cache, counter, previous_row = None, None, 0, None
    pending = None
    commands = np.full((total, 3), np.nan, dtype=np.float32)
    plan_ids = np.full(total, -1, dtype=np.int64)
    action_indices = np.full(total, -1, dtype=np.int64)
    errors, plans = [], []
    clock_origin, clock_tick = origin, 0
    timestamps = []
    for tick in range(total):
        now = (
            clock_origin + tick_ns(clock_tick)
            if clock == "paced"
            else origin + tick_ns(tick)
        )
        timestamps.append(now)
        if pending is not None and pending[0] <= now:
            completed, event = pending
            result = controller.call(event)
            if "error" in result:
                errors.append({"tick": tick, "op": "complete", **result})
            pending = None
        if clock == "paced" or tick % 6 == 0:
            index = int(np.searchsorted(walls, now, side="right") - 1)
            if index >= 0 and index != previous_row:
                row = rows[index]
                if row["generation"] != generation:
                    generation = row["generation"]
                    controller.call({"op": "reset", "generation": generation})
                    clock_origin, clock_tick = now, 0
                    cache = tuple(value.to(device) for value in decoder.empty_cache())
                    counter = 0
                    pending = None
                request_result = controller.call({
                    "op": "request",
                    "now_ns": now,
                    "observation_ns": int(row["frame_wall_ns"]),
                    "generation": generation,
                })
                if clock == "fixed" or request_result.get("error") not in (
                    "busy",
                    "not_due",
                ):
                    previous_row = index
                if "ok" in request_result:
                    request = request_result["ok"]
                    frame = torch.tensor(np.asarray(tokens[index]), device=device)[None]
                    cosine, sine = (value.to(device) for value in decoder.rope(counter))
                    encoded, new_k, new_v = policy.encoder.step(
                        frame,
                        past_k=cache[0].transpose(0, 1),
                        past_v=cache[1].transpose(0, 1),
                        cache_bias=cache[2],
                        cos=cosine,
                        sin=sine,
                        readout_only_final_block=True,
                    )
                    feature = encoded[:, -1]
                    if policy.norm is not None:
                        feature = policy.norm(feature)
                    cache = PatchPolicyDecoderStep.advance(
                        cache, new_k.transpose(0, 1), new_v.transpose(0, 1)
                    )
                    counter += 1
                    call_index = tick // 6
                    actions = sample_plan(
                        policy, feature, noise[call_index], request, arm == "rtc"
                    )
                    delay = (
                        0.0
                        if profile == "zero"
                        else float(latency[(call_index + seed) % len(latency)])
                    )
                    if profile == "overrun" and call_index % 50 == 49:
                        delay = 0.45 if call_index % 100 == 99 else 0.25
                    completion = now + int(np.ceil(delay * 1e9))
                    event = {
                        "op": "complete",
                        "now_ns": completion,
                        "generation": generation,
                        "request_id": request["id"],
                        "actions": actions.tolist(),
                    }
                    plans.append({
                        "tick": tick,
                        "row": index,
                        "request": request,
                        "latency_ns": completion - now,
                        "actions": actions.tolist(),
                    })
                    if completion == now:
                        result = controller.call(event)
                        if "error" in result:
                            errors.append({"tick": tick, "op": "complete", **result})
                    else:
                        pending = (completion, event)
                else:
                    errors.append({"tick": tick, "op": "request", **request_result})
        if generation is not None:
            result = controller.call({
                "op": "action",
                "now_ns": now,
                "generation": generation,
            })
            if "ok" in result and result["ok"] is not None:
                command = result["ok"]
                commands[tick] = command["values"]
                plan_ids[tick] = command["request_id"]
                action_indices[tick] = command["action_index"]
            elif "error" in result:
                errors.append({"tick": tick, "op": "action", **result})
        clock_tick += 1
    controller.close()
    return {
        "commands": commands,
        "plan_ids": plan_ids,
        "action_indices": action_indices,
        "noise": noise,
        "wall_ns": np.asarray(timestamps, dtype=np.int64),
        "scored": np.asarray(timestamps, dtype=np.int64) >= start,
    }, {"events": controller.events, "errors": errors, "plans": plans}


def command_metrics(arrays):
    values = arrays["commands"][arrays["scored"]]
    ids = arrays["plan_ids"][arrays["scored"]]
    valid = np.isfinite(values).all(axis=1)
    pairs = valid[1:] & valid[:-1]
    seams = pairs & (ids[1:] != ids[:-1])
    changes = np.abs(np.diff(values, axis=0))
    result = {
        "ticks": len(values),
        "command_coverage": float(valid.mean()),
        "seams": int(seams.sum()),
        "per_action": {},
    }
    for index, name in enumerate(("traction", "steering", "fork1")):
        command = values[:, index]
        result["per_action"][name] = {
            "mean_absolute": float(np.abs(command[valid]).mean())
            if valid.any()
            else None,
            "adjacent_delta_mean": float(changes[pairs, index].mean())
            if pairs.any()
            else None,
            "seam_delta_mean": float(changes[seams, index].mean())
            if seams.any()
            else None,
            "fraction_abs_le_threshold": {
                str(threshold): float((np.abs(command[valid]) <= threshold).mean())
                if valid.any()
                else None
                for threshold in (0.05, 0.10, 0.15)
            },
        }
    traction = values[:, 0]
    sustained = []
    for start in range(max(0, len(values) - 14)):
        window = traction[start : start + 15]
        sustained.append(
            bool(
                np.isfinite(window).all()
                and ((window > 0.15).all() or (window < -0.15).all())
            )
        )
    result["sustained_traction_500ms_fraction"] = (
        float(np.mean(sustained)) if sustained else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("extract", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("baseline", "rtc"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--clock", choices=("fixed", "paced"), default="fixed")
    args = parser.parse_args()
    setup()
    if args.phase == "extract":
        extract(args.output, args.model, args.device)
        return
    protocol = verify(args.output)
    if args.controller is None:
        parser.error("replay requires --controller")
    latency = json.loads(Path(protocol["latency_source"]).read_text())["seconds"]
    policy = load_policy(protocol, args.model, args.device)
    output = (
        args.output / ("async" if args.clock == "fixed" else "async-paced") / args.model
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with torch.inference_mode():
        for segment in SEGMENTS:
            token_path = args.output / "tokens" / args.model / f"{segment[1]}.npy"
            tokens, metadata = load_tokens(token_path, protocol["models"][args.model])
            for arm in (
                ("baseline",)
                if args.model == "baseline"
                else ("rtc_unconditioned", "rtc")
            ):
                for profile in ("zero", "measured", "overrun"):
                    for seed in SEEDS:
                        identity = f"{segment[0]}-{arm}-{profile}-{seed}"
                        arrays, trace = replay_one(
                            policy,
                            tokens,
                            metadata["rows"],
                            segment,
                            arm,
                            profile,
                            seed,
                            latency,
                            args.controller,
                            args.device,
                            args.clock,
                        )
                        np.savez_compressed(output / f"{identity}.npz", **arrays)
                        write_json(output / f"{identity}.json", trace)
                        summary[identity] = {
                            **command_metrics(arrays),
                            "errors": dict(
                                Counter(item["error"] for item in trace["errors"])
                            ),
                            "controller_sha256": sha256(args.controller),
                        }
                        print(
                            json.dumps({"completed": identity, **summary[identity]}),
                            flush=True,
                        )
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
