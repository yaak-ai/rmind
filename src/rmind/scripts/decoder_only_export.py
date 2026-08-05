"""Export the PatchPolicy baseline and the KV-cached decoder step to ONNX.

Both graphs come from the same hydra experiment config with a **randomly
initialized** trunk and heads (the frozen goal encoder and action tokenizer are
their real wandb artifacts; the ViT is its pretrained timm checkpoint). Shapes,
op counts and layer counts are therefore exactly the deployment graph's -- only
the weight VALUES are arbitrary, which is irrelevant for latency and memory and
is what makes this an *architecture* measurement rather than a checkpoint one.

Exporting both from one script is deliberate: the baseline reproduction is gate
zero for the comparison, and it is only a valid control if it differs from the
decoder graph in nothing but the trunk formulation.

    python -m rmind.scripts.decoder_only_export \
        --arm small --mode baseline --context 6 --out baseline_small_n6.onnx
    python -m rmind.scripts.decoder_only_export \
        --arm big --mode decoder --context 32 --out decoder_big_n32.onnx

Measuring the results (delta-dev1, AGX Orin, TRT 10.7) -- see
`/nasa/max/skills/trt-export/SKILL.md` and docs/decoder_only_kv_cache.md:

    # pin the GPU clock, or every number is ~55 ms pessimistic
    sudo sh -c 'echo performance > $DEVFREQ/governor'
    sudo sh -c 'echo 918000000 > $DEVFREQ/min_freq'
    # ... and confirm by sampling cur_freq THROUGH an idle gap, not after a run

    # build fp32 only: the reference baseline is fp32 and fp32 is the only
    # precision that has ever reached 0/200 on parity
    /home/max/Code/drivr/.venv/bin/python \
        /home/max/Code/drivr/scripts/build_trt_engine.py \
        --onnx MODEL.onnx --precision fp32 --workspace-gb 6

    /usr/src/tensorrt/bin/trtexec --loadEngine=MODEL.trt \
        --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000
"""

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.fx.experimental._config as _fx_config  # noqa: PLC2701
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from structlog import get_logger
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_flatten_with_path  # noqa: PLC2701

from rmind.components.transformer.causal_frame import (
    CACHE_ATTENTION_MODES,
    CacheAttention,
    CausalFrameTransformer,
)
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep
from rmind.utils.patch import monkeypatched

# tensordict's global dicts mutated during export tracing cause a spurious
# "pending unbacked symbol u0" error even though the exported graph is valid
# (same workaround as rmind.scripts.export_onnx).
_fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]

logger = get_logger(__name__)

# (experiment, trunk width, layers, heads). `_big` is 12L/768d/12H -- note the
# hand-off §7 says "8 layers, 512-d; 768-d in the `_big` arm", which understates
# the depth; dinov2_dinowm_big.yaml sets num_layers: 12.
ARMS = {
    "small": ("yaak/patch_policy/dinov2_dinowm", 512, 8, 8),
    "big": ("yaak/patch_policy/dinov2_dinowm_big", 768, 12, 12),
}
IMAGE_HW = 224  # dinov2 arms; 256 for dinov3
NUM_PATCHES = 256
TOKENS_PER_FRAME = NUM_PATCHES + 1  # speed token prepended
NUM_WAYPOINTS = 10
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def build_policy(arm: str, *, episode_length: int) -> tuple[Any, int, int, int]:
    """Instantiate the arm's `PatchPolicy` with the deployment input transform."""
    from torchvision.transforms.v2 import Normalize  # noqa: PLC0415

    experiment, dim, layers, heads = ARMS[arm]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", f"episode_length={episode_length}"],
        )
    model = instantiate(OmegaConf.to_container(cfg.model, resolve=True))
    model.sample_codes = False  # argmax decoding, as `load_for_export` does
    # deployment supplies already-cropped/resized [0,1] frames; ImageNet norm only
    model.input_transform[2]["image"] = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # the frozen artifacts were checkpointed on cuda and raw.yaml passes no
    # map_location; export is CPU-side
    return model.cpu().eval(), dim, layers, heads


def baseline_args(episode_length: int) -> tuple[dict[str, Any]]:
    """The current serving interface: `episode_length` frames re-encoded per tick."""
    return (
        {
            "data": {
                "cam_front_left": torch.rand(1, episode_length, 3, IMAGE_HW, IMAGE_HW),
                "meta/VehicleMotion/speed": torch.rand(1, episode_length, 1) * 130,
                "waypoints/xy_normalized": torch.rand(
                    1, episode_length, NUM_WAYPOINTS, 2
                )
                * 2
                - 1,
            }
        },
    )


def decoder_model_and_args(
    arm: str, context: int, cache_attention: CacheAttention = "concat"
) -> tuple[Module, tuple[dict[str, Tensor]], tuple[int, int, int, int]]:
    """The decoder step: ONE new frame against a cache of `context - 1` frames."""
    policy, dim, layers, heads = build_policy(arm, episode_length=1)
    policy.encoder = CausalFrameTransformer(
        dim_model=dim,
        num_layers=layers,
        num_heads=heads,
        tokens_per_frame=TOKENS_PER_FRAME,
        window=context,
        cache_attention=cache_attention,
    ).eval()
    step = PatchPolicyDecoderStep(policy=policy).eval()

    cache_frames = context - 1
    past_k, past_v, cache_bias = step.empty_cache(cache_frames=cache_frames)
    # export against a WARM cache -- the steady state. The graph is identical for a
    # cold cache and costs the same; only `cache_bias` differs.
    generator = torch.Generator().manual_seed(0)
    past_k = torch.randn(past_k.shape, generator=generator)
    past_v = torch.randn(past_v.shape, generator=generator)
    cache_bias = torch.zeros_like(cache_bias)
    rope_cos, rope_sin = step.rope(context - 1)

    args = (
        {
            "image": torch.rand(1, 1, 3, IMAGE_HW, IMAGE_HW),
            "speed": torch.rand(1, 1, 1) * 130,
            "waypoints": torch.rand(1, 1, NUM_WAYPOINTS, 2) * 2 - 1,
            "past_k": past_k,
            "past_v": past_v,
            "cache_bias": cache_bias,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        },
    )
    return step, args, (layers, heads, dim // heads, cache_frames)


def export(model: Module, args: tuple[Any, ...], out: Path, *, verify: bool) -> None:
    eager_out: Any = None
    for patch in (False, True):
        with monkeypatched(obj=torch.compiler, name="_is_exporting_flag", patch=patch):
            eager_out = model(*args)
    paths_and_leaves, _ = tree_flatten_with_path(eager_out)
    output_names = [
        ".".join(mk.key for mk in path)  # ty:ignore[unresolved-attribute]
        for path, _ in paths_and_leaves
    ]
    logger.debug("inferred output_names", output_names=output_names)

    exported = torch.export.export(mod=model, args=tuple(args), strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=exported,
        f=out,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=verify,
        report=False,
        output_names=output_names,
        artifacts_dir=out.parent,
    )
    logger.info("exported", path=out.as_posix(), mb=round(out.stat().st_size / 1e6, 1))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--mode", choices=["baseline", "decoder"], required=True)
    parser.add_argument(
        "--context",
        type=int,
        default=6,
        help="frames attended in TOTAL (decoder: 1 new + context-1 cached)",
    )
    parser.add_argument(
        "--attention",
        choices=CACHE_ATTENTION_MODES,
        default="concat",
        help="how the step attends over [cache, own frame] (decoder mode only). "
        "`concat` materializes a copy of the whole cache every tick; `split` "
        "merges two attentions with online softmax and copies nothing; "
        "`split_kt` additionally holds the K cache pre-transposed. All three are "
        "the same attention -- see tests/test_causal_frame.py.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ORT-vs-eager check; reports per-element RELATIVE error, which is "
        "alarming and meaningless on near-zero K/V entries -- compare absolute "
        "error against the tensor scale instead",
    )
    args = parser.parse_args()

    if args.out.exists():
        logger.info("exists, skipping", path=args.out.as_posix())
        return

    torch.manual_seed(1337)
    if args.mode == "baseline":
        model, *_ = build_policy(args.arm, episode_length=args.context)
        export_args: tuple[Any, ...] = baseline_args(args.context)
    else:
        model, export_args, shapes = decoder_model_and_args(
            args.arm, args.context, args.attention
        )
        layers, heads, head_dim, cache_frames = shapes
        logger.info(
            "cache",
            attention=args.attention,
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            cache_frames=cache_frames,
            cached_keys=cache_frames * TOKENS_PER_FRAME,
            past_k_shape=tuple(export_args[0]["past_k"].shape),
        )
    logger.info(
        "parameters",
        millions=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    )
    export(model, export_args, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
