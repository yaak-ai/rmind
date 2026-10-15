"""Export a D12 FM checkpoint with a streaming contract and numerical gates."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import simplejpeg
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rmind.models.d12_flow_decoder import OUTPUT_NAMES, D12FlowDecoderStep
from rmind.models.patch_policy_continuous import PatchPolicyContinuous
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(arguments.seed)
    policy = (
        PatchPolicyContinuous
        .load_from_checkpoint(
            arguments.checkpoint, map_location="cpu", weights_only=False
        )
        .eval()
        .to(arguments.device)
    )
    decoder = D12FlowDecoderStep(policy).eval().to(arguments.device)
    config = OmegaConf.load(arguments.config)
    stages = config.datamodule.val.dataset.samples.pipeline.functions
    builder, positioner, windower = [instantiate(stage.func) for stage in stages[:3]]
    job = Path(config.datamodule.val.dataset.samples.inputs.job_dir[-1])
    rows = builder(job_dir=job)
    samples = windower(frames=positioner(row_table=rows))
    sample = samples.row(min(50, len(samples) - 1), named=True)
    batch_data = {}
    for camera in policy.cameras:
        images = [
            simplejpeg.decode_jpeg(
                (job / camera / "frames/256x144" / f"{index:06d}.jpg").read_bytes(),
                colorspace="rgb",
            )
            for index in sample[f"{camera}/frame_idx"]
        ]
        batch_data[camera] = (
            torch.from_numpy(np.stack(images)).unsqueeze(0).to(arguments.device)
        )
    for name in ("speed", *policy.observations):
        batch_data[name] = torch.tensor(
            sample[name], dtype=torch.float32, device=arguments.device
        ).unsqueeze(0)
    batch = {"data": batch_data}
    cache = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple(value.to(arguments.device) for value in decoder.empty_cache()),
    )
    noise = torch.randn(1, policy.action_horizon * 3, device=arguments.device)
    with torch.no_grad():
        transformed = policy.input_transform(batch)
        for index in range(6):
            images = [
                batch_data[camera][:, index : index + 1].permute(0, 1, 4, 2, 3).float()
                / 255
                for camera in policy.cameras
            ]
            observations = [
                transformed["continuous"][name][:, index : index + 1]
                for name in (
                    "speed",
                    "fork_above_300",
                    "relative_ego_pos",
                    "relative_dropoff_pos",
                    "fork_height",
                )
            ]
            rotation = tuple(
                value.to(arguments.device) for value in decoder.rope(index)
            )
            inputs = (*images, *observations, *cache, *rotation, noise)
            outputs = decoder(*inputs)
            cache = PatchPolicyDecoderStep.advance(cache, outputs[1], outputs[2])
        features = policy._features(batch)[:, -1]
        reference = (
            policy.fm_sample(features, x0=noise) * policy.action_std
            + policy.action_mean
        )
        eager_error = float((outputs[0] - reference).abs().max())
        if eager_error > 1e-4:
            raise RuntimeError(f"window/streaming parity failed: {eager_error}")
        changed = decoder(*inputs[:-1], -noise)[0]
        if torch.equal(outputs[0], changed):
            raise RuntimeError("flow start noise does not affect actions")
    graph = arguments.output / "model.onnx"
    torch.onnx.export(
        decoder,
        inputs,
        str(graph),
        input_names=list(decoder.input_names),
        output_names=list(OUTPUT_NAMES),
        dynamo=True,
        external_data=True,
        opset_version=18,
    )
    onnx.checker.check_model(str(graph))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(graph), options, providers=["CPUExecutionProvider"]
    )
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in zip(decoder.input_names, inputs, strict=True)
    }
    actual = session.run(list(OUTPUT_NAMES), arrays)
    errors = {
        name: float(np.max(np.abs(value - expected.detach().cpu().numpy())))
        for name, value, expected in zip(OUTPUT_NAMES, actual, outputs, strict=True)
    }
    if max(errors.values()) > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {errors}")
    fixtures = {"inputs": [], "outputs": []}
    for section, names, tensors in [
        ("inputs", decoder.input_names, inputs),
        ("outputs", OUTPUT_NAMES, outputs),
    ]:
        for name, tensor in zip(names, tensors, strict=True):
            filename = f"{name}.f32"
            tensor.detach().float().cpu().numpy().tofile(arguments.output / filename)
            fixtures[section].append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "float32",
                "file": filename,
            })
    manifest = {
        "schema": "d12_flow_matching_v1",
        "action_head": "fm",
        "onnx": "model.onnx",
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.file_digest(
            arguments.checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        "cameras": list(policy.cameras),
        "image": {
            "height": 144,
            "width": 256,
            "layout": "BTCHW",
            "normalization": "unit_rgb",
            "crop": None,
        },
        "observations": {
            "history": 6,
            "hz": 5,
            "goal_scale_m": 10.0,
            "fork_height_scale_m": 2.3,
            "fork_height_fresh_ns": 250000000,
        },
        "actions": {
            "order": ["traction", "steering", "fork1"],
            "hz": 30,
            "horizon": policy.action_horizon,
            "near_horizon": policy.near_horizon,
            "space": "raw_normalized_commands",
            "mean": policy.action_mean.cpu().tolist(),
            "std": policy.action_std.cpu().tolist(),
        },
        "fm": {
            "steps": policy.fm_inference_steps,
            "x0": "sample",
            "seed": arguments.seed,
            "noise_width": policy.action_horizon * 3,
        },
        "cache": {
            "frames": 5,
            "layers": policy.encoder.num_layers,
            "heads": policy.encoder.num_heads,
            "tokens_per_frame": policy.encoder.tokens_per_frame,
            "head_dim": policy.encoder.head_dim,
            "rope_base": policy.encoder.rope_base,
        },
        "fixture": {
            "job": job.name,
            "frame_indices": sample["cam_fork/frame_idx"],
            **fixtures,
        },
        "parity": {"window_vs_streaming_max_abs": eager_error, "onnx_max_abs": errors},
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest["parity"]), flush=True)


if __name__ == "__main__":
    main()
