"""Generate real-frame streaming FM fixtures spanning cache eviction and resets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import simplejpeg
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rmind.models.d12_flow_decoder import OUTPUT_NAMES, D12FlowDecoderStep
from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--reset-at", type=int, default=8)
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.frames < 1 or arguments.start < 0:
        parser.error("frames must be positive and start nonnegative")
    if not 0 < arguments.reset_at < arguments.frames:
        parser.error("reset-at must be inside the sequence")
    if arguments.output.exists():
        parser.error("choose a new output directory")
    manifest = json.loads(arguments.manifest.read_text())
    checkpoint = Path(manifest["checkpoint"])
    with checkpoint.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    if checksum != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from exported engine")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = (
        PatchPolicyContinuous
        .load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        .eval()
        .to(arguments.device)
    )
    policy.fm_inference_steps = manifest["fm"]["steps"]
    decoder = D12FlowDecoderStep(policy).eval().to(arguments.device)
    config = OmegaConf.load(arguments.config)
    stages = config.datamodule.val.dataset.samples.pipeline.functions
    builder, positioner, windower = [instantiate(stage.func) for stage in stages[:3]]
    job = Path(config.datamodule.val.dataset.samples.inputs.job_dir[-1])
    samples = windower(frames=positioner(row_table=builder(job_dir=job)))
    if arguments.start + arguments.frames > len(samples):
        raise ValueError("not enough contiguous contexts")
    arguments.output.mkdir(parents=True)
    generator = torch.Generator(device=arguments.device).manual_seed(arguments.seed)
    sequence = {
        "schema": "d12_flow_sequence_v1",
        "checkpoint_sha256": checksum,
        "seed": arguments.seed,
        "steps": [],
        "limitations": "Recorded validation frames with explicit same-noise references. Resets are synthetic. Timing is logical 5Hz, not original camera PTS. Does not validate NVMM preprocessing, live state alignment or control dispatch.",
    }
    cache = None
    frame_index = 0
    generation = 0
    previous_camera_frame = None
    with torch.inference_mode():
        for step in range(arguments.frames):
            reset = step in (0, arguments.reset_at)
            if reset:
                cache = tuple(
                    value.to(arguments.device) for value in decoder.empty_cache()
                )
                frame_index = 0
                generation += 1
            sample = samples.row(arguments.start + step, named=True)
            camera_frame = sample["cam_fork/frame_idx"][-1]
            if (
                previous_camera_frame is not None
                and camera_frame - previous_camera_frame != 6
            ):
                raise ValueError("reference sequence crosses a camera gap")
            previous_camera_frame = camera_frame
            images = []
            for camera in policy.cameras:
                pixels = simplejpeg.decode_jpeg(
                    (
                        job
                        / camera
                        / "frames/256x144"
                        / f"{sample[f'{camera}/frame_idx'][-1]:06d}.jpg"
                    ).read_bytes(),
                    colorspace="rgb",
                )
                images.append(
                    torch
                    .from_numpy(pixels)
                    .permute(2, 0, 1)
                    .reshape(1, 1, 3, 144, 256)
                    .to(arguments.device)
                    .float()
                    / 255
                )
            batch_data = {
                name: torch.tensor(
                    sample[name], dtype=torch.float32, device=arguments.device
                ).unsqueeze(0)
                for name in ("speed", *policy.observations)
            }
            for camera, values in zip(policy.cameras, images, strict=True):
                batch_data[camera] = (
                    (values.permute(0, 1, 3, 4, 2) * 255)
                    .to(torch.uint8)
                    .expand(-1, 6, -1, -1, -1)
                )
            transformed = policy.input_transform({"data": batch_data})
            observations = [
                transformed["continuous"][name][:, -1:]
                for name in (
                    "speed",
                    "fork_above_300",
                    "relative_ego_pos",
                    "relative_dropoff_pos",
                    "fork_height",
                )
            ]
            rotation = tuple(
                value.to(arguments.device) for value in decoder.rope(frame_index)
            )
            noise = torch.randn(
                1,
                policy.action_horizon * 3,
                generator=generator,
                device=arguments.device,
            )
            inputs = (*images, *observations, *cache, *rotation, noise)
            outputs = decoder(*inputs)
            directory = arguments.output / f"step{step:03d}"
            directory.mkdir()
            fixtures = {
                "job": job.name,
                "frame_indices": [camera_frame],
                "inputs": [],
                "outputs": [],
            }
            for section, names, tensors in (
                ("inputs", decoder.input_names, inputs),
                ("outputs", OUTPUT_NAMES, outputs),
            ):
                for name, tensor in zip(names, tensors, strict=True):
                    array = tensor.detach().float().cpu().numpy()
                    if not np.isfinite(array).all():
                        raise ValueError(f"nonfinite reference {step}:{name}")
                    array.astype("<f4").tofile(directory / f"{name}.f32")
                    fixtures[section].append({
                        "name": name,
                        "shape": list(array.shape),
                        "dtype": "float32",
                        "file": f"{name}.f32",
                    })
            step_manifest = {**manifest, "fixture": fixtures}
            (directory / "manifest.json").write_text(
                json.dumps(step_manifest, indent=2) + "\n"
            )
            sequence["steps"].append({
                "manifest": f"step{step:03d}/manifest.json",
                "reset": reset,
                "generation": generation,
                "frame_index": frame_index,
                "frame_ns": 1_000_000_000 + step * 200_000_000,
                "camera_frame": camera_frame,
            })
            cache = PatchPolicyDecoderStep.advance(cache, outputs[1], outputs[2])
            frame_index += 1
            print(
                f"step={step} frame={camera_frame} generation={generation} reset={reset}",
                flush=True,
            )
    (arguments.output / "sequence.json").write_text(
        json.dumps(sequence, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
