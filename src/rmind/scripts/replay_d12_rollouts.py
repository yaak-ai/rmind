"""Replay recorded D12 inputs with explicit image, noise, and cache provenance."""

import argparse
import hashlib
import json
import operator
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

AUX_NAMES = (
    "speed",
    "fork_above_300",
    "relative_ego_pos",
    "relative_dropoff_pos",
    "fork_height",
)
ACTION_HORIZON = 50
OFFSET_PAIR_LENGTH = 2


def read_recording(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    inputs, predictions, frames, skipped = [], [], {}, []
    files = sorted(
        directory.glob("sensor--*.mcap"),
        key=lambda path: int(path.stem.split("--")[-1]),
    )
    for path in files:
        with path.open("rb") as handle:
            reader = make_reader(handle, decoder_factories=[DecoderFactory()])
            try:
                if reader.get_summary() is None:
                    skipped.append(path.name)
                    continue
            except Exception as error:
                skipped.append(f"{path.name}: {error}")
                continue
            for _, channel, message, decoded in reader.iter_decoded_messages(
                topics=["policy/inputs", "policy/prediction", "cam_fork/frame"]
            ):
                if channel.topic == "policy/inputs" and decoded.sent_to_model:
                    inputs.append({
                        "wall_ns": message.log_time,
                        "step_ns": decoded.step_tov_ns,
                        "generation": decoded.reset_generation,
                        "input_frame": next(
                            item.frame_index
                            for item in decoded.frames
                            if item.camera == "cam_fork"
                        ),
                        "frame_pts_ns": next(
                            item.pts_ns
                            for item in decoded.frames
                            if item.camera == "cam_fork"
                        ),
                        "aux": {item.name: list(item.values) for item in decoded.aux},
                    })
                elif channel.topic == "policy/prediction":
                    heads = {
                        item.name.split(".")[-1]: item.value for item in decoded.heads
                    }
                    predictions.append({
                        "wall_ns": message.log_time,
                        "step_ns": decoded.step_tov_ns,
                        "seq": decoded.seq,
                        "published": decoded.published,
                        "actions": [
                            heads[name] for name in ("traction", "steering", "fork1")
                        ],
                    })
                elif channel.topic == "cam_fork/frame":
                    frames[decoded.pts_ns] = (message.log_time, decoded.frame_index)
    inputs.sort(key=operator.itemgetter("wall_ns"))
    predictions.sort(key=operator.itemgetter("wall_ns"))
    for index, row in enumerate(inputs):
        row["request_index"] = index
        row["frame_wall_ns"], row["frame"] = frames.get(
            row["frame_pts_ns"], (None, None)
        )
    return inputs, predictions, skipped


def resize_bilinear_u8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    horizontal = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    vertical = (
        np.arange(height, dtype=np.float32) + 0.5
    ) * source_height / height - 0.5
    left, top = np.floor(horizontal).astype(int), np.floor(vertical).astype(int)
    horizontal_weight = (horizontal - left)[None, :, None].astype(np.float32)
    vertical_weight = (vertical - top)[:, None, None].astype(np.float32)
    right, bottom = (
        np.clip(left + 1, 0, source_width - 1),
        np.clip(top + 1, 0, source_height - 1),
    )
    left, top = np.clip(left, 0, source_width - 1), np.clip(top, 0, source_height - 1)
    upper = image[top[:, None], left].astype(np.float32)
    upper += horizontal_weight * (image[top[:, None], right].astype(np.float32) - upper)
    lower = image[bottom[:, None], left].astype(np.float32)
    lower += horizontal_weight * (
        image[bottom[:, None], right].astype(np.float32) - lower
    )
    return (
        np.rint(upper + vertical_weight * (lower - upper)).clip(0, 255).astype(np.uint8)
    )


def video_images(
    directory: Path, rows: list[dict[str, Any]], frames_per_video: int
) -> Iterator[tuple[int, np.ndarray]]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        msg = "Video replay requires ffprobe and ffmpeg on PATH"
        raise FileNotFoundError(msg)
    selected = defaultdict(list)
    for row in rows:
        selected[row["frame"] // frames_per_video].append(
            row["frame"] % frames_per_video
        )
    for chunk, selected_indices in sorted(selected.items()):
        path = directory / f"cam_fork--{chunk}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        probe = json.loads(
            subprocess.check_output([
                str(Path(ffprobe).resolve()),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames",
                "-of",
                "json",
                str(path),
            ])
        )["streams"][0]
        if int(probe["nb_frames"]) != frames_per_video:
            msg = f"Cannot assume fixed video index mapping: {path} has {probe['nb_frames']} frames"
            raise ValueError(msg)
        indices = sorted(set(selected_indices))
        select = "+".join(f"eq(n\\,{index})" for index in indices)
        command = [
            str(Path(ffmpeg).resolve()),
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-vf",
            f"select={select}",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
            if process.stdout is None:
                msg = "ffmpeg stdout pipe was not created"
                raise RuntimeError(msg)
            for index in indices:
                byte_count = probe["width"] * probe["height"] * 3
                raw = process.stdout.read(byte_count)
                if len(raw) != byte_count:
                    msg = f"Incomplete decoded frame: {path}:{index}"
                    raise ValueError(msg)
                image = np.frombuffer(raw, dtype=np.uint8).reshape(
                    probe["height"], probe["width"], 3
                )
                capture = resize_bilinear_u8(image, 324, 576)
                yield (
                    chunk * frames_per_video + index,
                    resize_bilinear_u8(capture, 144, 256),
                )
            if process.wait() != 0:
                msg = f"ffmpeg failed: {path}"
                raise RuntimeError(msg)


def candidate_execution_indices(
    predictions: list[dict[str, Any]], row: dict[str, Any], rate: int = 30
) -> np.ndarray:
    timestamps = np.asarray(
        [prediction["wall_ns"] for prediction in predictions], dtype=np.int64
    )
    return np.floor((timestamps - row["frame_wall_ns"]) * rate / 1e9).astype(int)


def noise_index(
    request_index: int, initial_offset: int, changes: Sequence[tuple[int, int]]
) -> int:
    offset = initial_offset
    for boundary, changed_offset in sorted(changes):
        if request_index >= boundary:
            offset = changed_offset
    return request_index + offset


def selected_video_hashes(
    directory: Path, rows: list[dict[str, Any]], frames_per_video: int
) -> dict[str, str]:
    if frames_per_video <= 0:
        msg = "frames_per_video must be positive"
        raise ValueError(msg)
    result = {}
    for chunk in sorted({row["frame"] // frames_per_video for row in rows}):
        path = directory / f"cam_fork--{chunk}.mp4"
        with path.open("rb") as stream:
            result[str(path.resolve())] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def feature_array_digest(features: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()


def noise_provenance(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": checksum,
        "generator_verified": False,
        "declared_generator": "Rust StdRng seed287, rand0.9.5/rand_distr0.5.1; not verified from these supplied bytes",
    }


def load_feature_cache(
    path: Path, fingerprint: str, expected_sha256: str | None = None
) -> np.ndarray:
    if expected_sha256 is not None:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                msg = "External feature cache checksum changed"
                raise ValueError(msg)
    with np.load(path, allow_pickle=False) as saved:
        if "fingerprint" not in saved or str(saved["fingerprint"]) != fingerprint:
            msg = "Feature provenance changed or absent; rebuild explicitly"
            raise ValueError(msg)
        features = saved["features"]
        if "features_sha256" not in saved or str(
            saved["features_sha256"]
        ) != feature_array_digest(features):
            msg = "Feature array checksum changed or absent; rebuild explicitly"
            raise ValueError(msg)
        return features


def main() -> None:
    import torch

    from rmind.models.d12_flow_decoder import D12FlowDecoderStep
    from rmind.models.patch_policy_continuous import PatchPolicyContinuous
    from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rust-noise", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-inputs", type=int)
    parser.add_argument("--noise-offset", type=int, default=0)
    parser.add_argument(
        "--noise-offset-change", action="append", default=[], metavar="REQUEST:OFFSET"
    )
    parser.add_argument("--features-from", type=Path)
    parser.add_argument("--features-sha256")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--noise-audit-radius", type=int, default=4)
    parser.add_argument("--frames-per-video", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.features_from is not None and (
        arguments.features_sha256 is None or arguments.rebuild_features
    ):
        parser.error(
            "--features-from requires --features-sha256 and cannot be rebuilt in place"
        )
    if arguments.features_sha256 is not None and arguments.features_from is None:
        parser.error("--features-sha256 requires --features-from")
    changes = [
        tuple(map(int, value.split(":"))) for value in arguments.noise_offset_change
    ]
    if any(len(change) != OFFSET_PAIR_LENGTH for change in changes):
        msg = "Noise offset changes must be REQUEST:OFFSET"
        raise ValueError(msg)
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows, predictions, skipped = read_recording(arguments.job)
    rows = rows[: arguments.max_inputs]
    if any(row["frame_wall_ns"] is None for row in rows):
        msg = "Selected inputs include frames absent from complete MCAPs; limit the replay prefix"
        raise ValueError(msg)
    with arguments.checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    video_hashes = selected_video_hashes(
        arguments.job, rows, arguments.frames_per_video
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "checkpoint": checkpoint_sha256,
                "rows": rows,
                "videos": video_hashes,
                "image_pipeline": "PTS-matched compressed RGB; half-pixel uint8 576x324 then 256x144 v1",
                "frames_per_video": arguments.frames_per_video,
                "cache": "all submitted inputs, generation resets, five-frame streaming v1",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    by_step = defaultdict(list)
    for prediction in predictions:
        by_step[prediction["step_ns"]].append(prediction)
    noise_input = noise_provenance(arguments.rust_noise)
    noise = np.fromfile(arguments.rust_noise, dtype="<f4").reshape(-1, 150)
    noise_indices = np.asarray([
        noise_index(
            index, arguments.noise_offset, cast("list[tuple[int, int]]", changes)
        )
        for index in range(len(rows))
    ])
    if (noise_indices < 0).any() or (noise_indices >= len(noise)).any():
        msg = "Insufficient Rust noise stream"
        raise ValueError(msg)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    policy = (
        PatchPolicyContinuous
        .load_from_checkpoint(
            arguments.checkpoint, map_location="cpu", weights_only=False
        )
        .eval()
        .to(arguments.device)
    )
    if (
        tuple(policy.cameras) != ("cam_fork",)
        or policy.action_horizon != ACTION_HORIZON
    ):
        msg_0 = "This replay contract requires the fork-only H50 model"
        raise ValueError(msg_0)
    decoder = D12FlowDecoderStep(policy).eval().to(arguments.device)
    features_path = arguments.features_from or arguments.output / "features.npz"
    if arguments.features_from is not None and not features_path.is_file():
        raise FileNotFoundError(features_path)
    if features_path.exists() and not arguments.rebuild_features:
        features = load_feature_cache(
            features_path, fingerprint, arguments.features_sha256
        )
    else:
        images = iter(video_images(arguments.job, rows, arguments.frames_per_video))
        cached_frame, cached_image = None, None
        cache, previous_generation, frame_counter, features = None, None, 0, []
        with torch.inference_mode():
            for index, row in enumerate(rows):
                if cached_frame != row["frame"]:
                    cached_frame, cached_image = next(images)
                if cached_frame != row["frame"]:
                    msg = "Video/input frame order mismatch"
                    raise ValueError(msg)
                if row["generation"] != previous_generation:
                    cache = cast(
                        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]",
                        tuple(
                            value.to(arguments.device)
                            for value in decoder.empty_cache()
                        ),
                    )
                    previous_generation, frame_counter = row["generation"], 0
                if cached_image is None or cache is None:
                    msg = "Replay requires an image and initialized generation cache"
                    raise ValueError(msg)
                image = (
                    torch.tensor(
                        cached_image.transpose(2, 0, 1).copy(), device=arguments.device
                    ).float()[None, None]
                    / 255
                )
                aux = {
                    name: torch.tensor(
                        row["aux"][name], device=arguments.device
                    ).reshape(1, 1, -1)
                    for name in AUX_NAMES
                }
                normalized = (
                    image[:, :, None] - decoder.image_mean
                ) / decoder.image_std
                tokens = policy._frame_tokens(
                    normalized,
                    aux["speed"],
                    {name: aux[name] for name in AUX_NAMES[1:]},
                )
                cosine, sine = (
                    value.to(arguments.device) for value in decoder.rope(frame_counter)
                )
                encoded, new_k, new_v = policy.encoder.step(
                    tokens[:, 0],
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
                features.append(feature[0].cpu().numpy())
                cache = PatchPolicyDecoderStep.advance(
                    cache, new_k.transpose(0, 1), new_v.transpose(0, 1)
                )
                frame_counter += 1
                if index % 100 == 0:
                    print(f"encoded {index}/{len(rows)}", flush=True)
        features = np.stack(features)
        np.savez(
            features_path,
            features=features,
            step_ns=[row["step_ns"] for row in rows],
            fingerprint=fingerprint,
            features_sha256=feature_array_digest(features),
        )
    historical, variants, matches = [], [], []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            feature = torch.tensor(features[index : index + 1], device=arguments.device)
            stream_index = noise_indices[index]
            start = torch.tensor(
                noise[stream_index : stream_index + 1].copy(), device=arguments.device
            )
            actions = (
                (
                    policy.fm_sample(feature, x0=start) * policy.action_std
                    + policy.action_mean
                )[0]
                .cpu()
                .numpy()
            )
            historical.append(actions)
            random_start = torch.randn(arguments.samples, 150, device=arguments.device)
            samples = (
                (
                    policy.fm_sample(
                        feature.expand(arguments.samples, -1), x0=random_start
                    )
                    * policy.action_std
                    + policy.action_mean
                )
                .cpu()
                .numpy()
            )
            variants.append(samples)
            recorded = by_step[row["step_ns"]]
            if recorded:
                observed = np.asarray([
                    prediction["actions"] for prediction in recorded
                ])
                nominal = candidate_execution_indices(recorded, row)
                candidates = []
                for shift in (-1, 0, 1):
                    slots = nominal + shift
                    valid = (slots >= 0) & (slots < ACTION_HORIZON)
                    if valid.any():
                        residual = actions[slots[valid]].clip(-1, 1) - observed[valid]
                        candidates.append({
                            "slot_shift": shift,
                            "mae": float(np.abs(residual).mean()),
                            "max_error": float(np.abs(residual).max()),
                        })
                offsets = np.arange(
                    -arguments.noise_audit_radius, arguments.noise_audit_radius + 1
                )
                stream_indices = offsets + stream_index
                available = (stream_indices >= 0) & (stream_indices < len(noise))
                offsets, stream_indices = offsets[available], stream_indices[available]
                alternatives = (
                    (
                        policy.fm_sample(
                            feature.expand(len(offsets), -1),
                            x0=torch.tensor(
                                noise[stream_indices].copy(), device=arguments.device
                            ),
                        )
                        * policy.action_std
                        + policy.action_mean
                    )
                    .cpu()
                    .numpy()
                )
                valid = (nominal >= 0) & (nominal < ACTION_HORIZON)
                errors = np.abs(
                    alternatives[:, nominal[valid]].clip(-1, 1) - observed[None, valid]
                ).mean(axis=(1, 2))
                matches.append({
                    "request_index": index,
                    "wall_ns": row["wall_ns"],
                    "count": len(recorded),
                    "nominal_slots": nominal.tolist(),
                    "candidates": candidates,
                    "noise_offset_errors": dict(
                        zip(offsets.astype(str), errors.tolist(), strict=True)
                    ),
                })
    if noise_provenance(arguments.rust_noise) != noise_input:
        msg_0 = "Noise source changed during replay"
        raise ValueError(msg_0)
    if (
        selected_video_hashes(arguments.job, rows, arguments.frames_per_video)
        != video_hashes
    ):
        msg_0 = "Video source changed during replay"
        raise ValueError(msg_0)
    np.savez_compressed(
        arguments.output / "chunks.npz",
        actions=np.stack(historical),
        variants=np.stack(variants),
        wall_ns=[row["wall_ns"] for row in rows],
        step_ns=[row["step_ns"] for row in rows],
        generation=[row["generation"] for row in rows],
        frame=[row["frame"] for row in rows],
        noise_indices=noise_indices,
    )
    report = {
        "historical_exactness": "UNVERIFIED: candidate reconstruction, not historical ground truth",
        "images": "Compressed recorded RGB, half-pixel bilinear uint8 1920x1080 -> 576x324 -> 256x144; original UYVY absent",
        "video_mapping": f"Assumed frame_index // {arguments.frames_per_video}; selected videos checked for frame count",
        "noise": "Supplied explicit noise bytes; candidate request-to-inference mapping, not directly logged",
        "noise_input": noise_input,
        "noise_offset": arguments.noise_offset,
        "noise_offset_changes": changes,
        "feature_cache": str(features_path),
        "cache": "Generation changes reset five-frame cache and RoPE counter; rejected or unlogged calls not yet reconstructed",
        "execution_index": "Wall timestamps referenced to recorded camera frame; candidates +/- one action index reported",
        "action_clipping": "Raw chunks retained; emitted-action comparison clips to [-1,1], matching ActionPlan.at",
        "checkpoint": str(arguments.checkpoint),
        "inputs": len(rows),
        "skipped_mcaps": skipped,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_fingerprint": fingerprint,
        "source_video_sha256": video_hashes,
        "feature_array_sha256": feature_array_digest(features),
        "first_prediction_seq": predictions[0]["seq"] if predictions else None,
        "matches": matches,
    }
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2))
    (arguments.output / "inputs.json").write_text(json.dumps(rows))
    print(
        json.dumps({
            "inputs": len(rows),
            "matched_chunks": len(matches),
            "output": str(arguments.output),
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
