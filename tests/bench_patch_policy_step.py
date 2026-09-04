"""Benchmark: one COMBINED PatchPolicy training step, real config, real shapes.

`bench_causal_frame.py` measures the ViT and the trunk separately and the gate
(docs/decoder_only_kv_cache.md §14) sums their peaks. That sum is a lower bound:
it omits the tokenizer, the heads and losses, the input transform, and the fp32
master weights the real step carries. This runs the whole `PatchPolicy` from its
own experiment config -- forward, backward, optimizer step -- so the number is
the one that actually has to fit on the training card.

Not a pytest module (pytest only collects `test_*.py`). Needs a GPU, network
(the frozen goal-encoder and action-tokenizer W&B artifacts, and the pretrained
timm checkpoint), and `nix develop`:

    NIX_LD_LIBRARY_PATH=$NVLIBS TRITON_LIBCUDA_PATH=$NVLIBS \\
      uv run python tests/bench_patch_policy_step.py 8 dinov2_registers_causal_3cam

where `$NVLIBS` holds symlinks to the host's `libcuda`/`libnvidia-*` only -- see
docs/decoder_only_kv_cache.md §14.4's environment note.

The batch is synthetic (`torch.testing.make_tensor` at the yaak batch's real
paths and dtypes), so the LOSS VALUES mean nothing in absolute terms. Two things
about them do mean something: that they stay finite, and that they fall on a
repeated fixed batch -- this is an overfit-one-batch check
(plans/effervescent-chasing-seahorse.md, Verification 4) at production shape,
which is what proves gradient actually reaches the LoRA adapters and the
per-camera registers rather than the trunk quietly carrying the whole run.

`--lr`: the cosine-with-warmup `LambdaLR` steps itself once at construction, so
every param group's lr is **0.0** until the Lightning trainer advances the
scheduler. Left alone, nothing moves and `lora_A` never leaves its zero-init
gradient -- the run looks frozen and the check is vacuous. The default forces a
usable lr instead; pass `--lr 0` to see the configured behaviour.
"""

import argparse
import os
import pathlib
import time
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rbyte.types import Batch
from tensordict import TensorDict
from torch.testing import make_tensor

import rmind.utils  # noqa: F401  -- registers the `eval` omegaconf resolver

# str, not Path: hydra's `initialize_config_dir` does `path.find("://")` on it
CONFIG_DIR = str((pathlib.Path(__file__).parent / ".." / "config").resolve())
IMAGE_HW = (324, 576)  # pre-crop; the config's CenterCrop/Resize run in-graph
NUM_WAYPOINTS = 10


def emit(line: str) -> None:
    """`print` is auto-removed by the repo's ruff config (`select = ["ALL"]`)."""
    os.write(1, f"{line}\n".encode())


def build(experiment: str, overrides: list[str]) -> tuple[Any, Any, int]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment=yaak/patch_policy/{experiment}", *overrides],
        )
    model = hydra.utils.instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    # `ChunkFields` unfolds the action fields by `action_horizon` and then
    # narrows every field to `episode_length`, so the batch must be longer than
    # the clip it produces or the unfold has nothing to consume.
    return model.cuda().train(), cfg, cfg.episode_length + cfg.action_horizon - 1


def make_synthetic_batch(cameras: tuple[str, ...], batch: int, frames: int) -> Batch:
    device = torch.device("cuda")
    kw = {"device": device}
    data = {
        camera: make_tensor(
            (batch, frames, *IMAGE_HW, 3), dtype=torch.uint8, low=0, high=256, **kw
        )
        for camera in cameras
    } | {
        "meta/VehicleMotion/speed": make_tensor(
            (batch, frames), dtype=torch.float32, low=0.0, high=130.0, **kw
        ),
        "meta/VehicleMotion/gas_pedal_normalized": make_tensor(
            (batch, frames), dtype=torch.float32, low=0.0, high=1.0, **kw
        ),
        "meta/VehicleMotion/brake_pedal_normalized": make_tensor(
            (batch, frames), dtype=torch.float32, low=0.0, high=1.0, **kw
        ),
        "meta/VehicleMotion/steering_angle_normalized": make_tensor(
            (batch, frames), dtype=torch.float32, low=-1.0, high=1.0, **kw
        ),
        "meta/VehicleState/turn_signal": make_tensor(
            (batch, frames), dtype=torch.int64, low=0, high=3, **kw
        ),
        "waypoints/xy_normalized": make_tensor(
            (batch, frames, NUM_WAYPOINTS, 2),
            dtype=torch.float32,
            low=0.0,
            high=20.0,
            **kw,
        ),
    }
    return Batch(
        data=TensorDict(data, batch_size=[batch], device=device),
        batch_size=[batch],
        device=device,
    )


def encoder_grad_norms(encoder: Any) -> str:
    """Per-camera register + LoRA gradient norms, or "" for a frozen encoder.

    `lora_A` reading exactly 0 on step 0 is CORRECT, not a disconnected graph:
    `lora_B` is zero-initialized, so `dL/dA` carries a factor of `B` and
    vanishes until the first optimizer step moves `B` off zero. It must be
    non-zero from step 1 onward -- if it stays at 0, the adapter is a permanent
    no-op (most likely the lr is still 0, see the module docstring).
    """
    if not hasattr(encoder, "camera_reg_token"):
        return ""

    def total(suffix: str) -> float:
        params = [
            p
            for name, p in encoder.named_parameters()
            if name.endswith(suffix) and p.grad is not None
        ]
        return float(sum(p.grad.norm() ** 2 for p in params) ** 0.5) if params else 0.0

    grad = encoder.camera_reg_token.grad
    per_camera = (
        [round(float(grad[c].norm()), 6) for c in range(grad.shape[0])]
        if grad is not None
        else None
    )
    return (
        f"  reg={per_camera} lora_A={total('lora_A'):.6f} lora_B={total('lora_B'):.6f}"
    )


def main() -> None:  # noqa: PLR0914
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=int, nargs="?", default=8)
    parser.add_argument("experiment", nargs="?", default="dinov2_registers_causal_3cam")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra hydra override, repeatable -- e.g. --override window=6 "
        "--override episode_length=16. Use this to dry-run the exact geometry of "
        "a run you are about to launch, before paying for the sample index build.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="override every param group's lr; 0 keeps the configured schedule, "
        "whose warmup makes step 0's lr exactly 0 (see module docstring)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        msg = "no CUDA device"
        raise SystemExit(msg)

    model, cfg, frames = build(args.experiment, args.override)
    emit(
        f"{torch.cuda.get_device_name(0)} arm={args.experiment} "
        f"tokens_per_frame={model.encoder.tokens_per_frame} "
        f"episode_length={cfg.episode_length} window={cfg.window} "
        f"batch={args.batch}"
    )
    batch = make_synthetic_batch(tuple(model.cameras), args.batch, frames)

    optimizer = model.configure_optimizers()["optimizer"]
    if args.lr:
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    emit(f"lr={[g['lr'] for g in optimizer.param_groups]}")

    torch.cuda.reset_peak_memory_stats()
    inputs = batch.to_dict(retain_none=False)
    for step in range(args.steps):
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            metrics = model._compute_metrics(inputs)  # noqa: SLF001
        loss = metrics["policy", "loss"].sum(reduce=True)
        loss.backward()
        grads = encoder_grad_norms(model.image_encoder)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        emit(
            f"step {step}: loss={loss.item():.4f} "
            f"finite={bool(torch.isfinite(loss))} "
            f"{1000 * (time.perf_counter() - started):.0f} ms "
            f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB{grads}"
        )

    encoder = model.image_encoder
    if hasattr(encoder, "camera_reg_token"):
        # DrivoR Fig. 4. Only meaningful with PER-CAMERA registers, and only
        # informative after real training -- on synthetic noise every camera
        # sees the same gradient direction, so high similarity here says
        # nothing about whether the views stay distinct on real data.
        reg = encoder.camera_reg_token.detach()
        flat = F.normalize(reg.reshape(reg.shape[0], -1), dim=-1)
        emit(f"per-camera register cosine similarity:\n{(flat @ flat.T).cpu().numpy()}")


if __name__ == "__main__":
    main()
