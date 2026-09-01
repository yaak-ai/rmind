"""Export the PatchPolicy baseline and the KV-cached decoder step to ONNX.

Without `--artifact`/`--ckpt` both graphs come from the hydra experiment config
with a **randomly initialized** trunk and heads (the frozen goal encoder and
action tokenizer are their real wandb artifacts; the ViT is its pretrained timm
checkpoint). Shapes, op counts and layer counts are therefore exactly the
deployment graph's -- only the weight VALUES are arbitrary, which is irrelevant
for latency and memory and is what makes that an *architecture* measurement
rather than a checkpoint one.

Exporting both from one script is deliberate: the baseline reproduction is gate
zero for the comparison, and it is only a valid control if it differs from the
decoder graph in nothing but the trunk formulation.

    python -m rmind.scripts.decoder_only_export \
        --arm small --mode baseline --context 6 --out baseline_small_n6.onnx
    python -m rmind.scripts.decoder_only_export \
        --arm big --mode decoder --context 32 --out decoder_big_n32.onnx

**Trained weights** (`--artifact`, the parity/precision case). The architecture
then comes from the CHECKPOINT's own hparams, not from `--arm`, and the trunk is
used as trained instead of being replaced by a fresh one:

    python -m rmind.scripts.decoder_only_export --mode decoder \
        --artifact yaak/rmind/model-do8m9ot8:v0 --out decoder_do8m9ot8_n16.onnx

`--context` then defaults to the trunk's own trained `window`, which is the only
context the checkpoint is *valid* at: `step` reads the cache size off `past_k`'s
shape and has no intrinsic maximum length, so a checkpoint will happily run
against any cache and silently extrapolate (docs/decoder_only_kv_cache.md §10.3).
Passing a different `--context` is legitimate for a LATENCY curve and nothing
else -- such engines must not be served, and the script says so.

`attention_impl` (`flex` in the causal training arm) is irrelevant here by
construction: `CausalFrameTransformer.step` is always SDPA, because a
`torch.compile`d Triton kernel is not exportable (§11.6.7).

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

from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep
from rmind.utils.patch import monkeypatched

# tensordict's global dicts mutated during export tracing cause a spurious
# "pending unbacked symbol u0" error even though the exported graph is valid
# (same workaround as rmind.scripts.export_onnx).
_fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]

logger = get_logger(__name__)

# (experiment, trunk width, layers, heads). `_big` is 12L/768d/12H -- note the
# hand-off §7 says "8 layers, 512-d; 768-d in the `_big` arm", which understates
# the depth; dinov2_dinowm_big.yaml sets num_layers: 12. `small_3cam` is `small`
# plus cam_left_forward/cam_right_forward (config/experiment/yaak/patch_policy/
# dinov2_dinowm_causal_3cam.yaml) -- same width/layers/heads, 3x the per-frame
# patch tokens.
ARMS = {
    "small": ("yaak/patch_policy/dinov2_dinowm", 512, 8, 8),
    "big": ("yaak/patch_policy/dinov2_dinowm_big", 768, 12, 12),
    "small_3cam": ("yaak/patch_policy/dinov2_dinowm_causal_3cam", 512, 8, 8),
    # same geometry as `small_3cam` -- the composed intra-frame position table
    # is (769, 512) in every arm -- but the table is FACTORIZED, so the decode
    # graph gains a constant index_select/matmul that has to fold away. Exists
    # so that is verifiable before committing to a training run.
    "small_3cam_pano": ("yaak/patch_policy/dinov2_dinowm_causal_3cam_pano", 512, 8, 8),
}
IMAGE_HW = 224  # dinov2 arms; 256 for dinov3
NUM_PATCHES = 256
NUM_WAYPOINTS = 10
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def tokens_per_frame(
    num_cameras: int, *, use_readout_token: bool = False, num_register_tokens: int = 0
) -> int:
    """`num_cameras * NUM_PATCHES + 1` -- the speed token plus every camera's
    patches (257 at `num_cameras=1`, 769 at `num_cameras=3`). With
    `use_readout_token` the frame additionally carries `num_register_tokens`
    register tokens plus one readout token (260 at `num_cameras=1`, 2 registers).

    Only the RANDOM-INIT export path needs this: on a checkpoint the geometry
    comes from the trained trunk's own `tokens_per_frame`.
    """
    k = NUM_PATCHES * num_cameras + 1
    if use_readout_token:
        k += num_register_tokens + 1
    return k


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


def _intra_position_kwargs(encoder: Any) -> dict[str, Any]:
    """The intra-frame position arm of a config-built trunk, as ctor kwargs.

    Empty for anything that is not a `CausalFrameTransformer` (or predates the
    configurable arms), which keeps the default `flat`/`norm_gain` behaviour.
    """
    trunk = getattr(encoder, "_orig_mod", encoder)  # unwrap `compiled`
    if not isinstance(trunk, CausalFrameTransformer) or not hasattr(
        trunk, "intra_position_factorization"
    ):
        return {}
    return {
        "intra_position_scaling": trunk.intra_position_scaling,
        "intra_position_factorization": trunk.intra_position_factorization,
        "intra_position_target_norm": trunk.intra_position_target_norm,
        "num_cameras": trunk.num_cameras,
        "patch_grid": trunk.patch_grid,
        "num_prefix_tokens": trunk.num_prefix_tokens,
        "num_suffix_tokens": trunk.num_suffix_tokens,
        "camera_yaw_deg": trunk.camera_yaw_deg,
        "camera_hfov_deg": trunk.camera_hfov_deg,
    }


def baseline_args(
    episode_length: int, cameras: tuple[str, ...] = ("cam_front_left",)
) -> tuple[dict[str, Any]]:
    """The current serving interface: `episode_length` frames re-encoded per tick."""
    return (
        {
            "data": {
                **{
                    camera: torch.rand(1, episode_length, 3, IMAGE_HW, IMAGE_HW)
                    for camera in cameras
                },
                "meta/VehicleMotion/speed": torch.rand(1, episode_length, 1) * 130,
                "waypoints/xy_normalized": torch.rand(
                    1, episode_length, NUM_WAYPOINTS, 2
                )
                * 2
                - 1,
            }
        },
    )


def load_trained_policy(*, artifact: str | None, ckpt: str | None) -> Any:
    """A TRAINED `PatchPolicy` with the deployment input transform.

    `load_for_export` applies exactly the deployment conventions the random path
    replicates by hand: `sample_codes=False` (argmax decoding, deterministic) and
    the in-model crop/resize pipeline replaced by ImageNet `Normalize` only, so
    the host owes an already-cropped `[0, 1]` frame.
    """
    from rmind.models.patch_policy import PatchPolicy  # noqa: PLC0415

    if artifact is not None:
        return PatchPolicy.load_for_export(artifact).cpu().eval()

    from torchvision.transforms.v2 import Normalize  # noqa: PLC0415

    model = PatchPolicy.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False
    )
    model.sample_codes = False
    model.input_transform[2]["image"] = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return model.cpu().eval()


def decoder_model_and_args(  # noqa: PLR0914
    arm: str,
    context: int | None,
    *,
    artifact: str | None = None,
    ckpt: str | None = None,
) -> tuple[
    PatchPolicyDecoderStep, tuple[dict[str, Tensor]], tuple[int, int, int, int, int]
]:
    """The decoder step: ONE new frame against a cache of `context - 1` frames."""  # noqa: DOC501
    if artifact is not None or ckpt is not None:
        # Trained: the architecture comes from the checkpoint's hparams and the
        # trunk is used AS TRAINED. Replacing it (the random path below) would
        # silently discard every trained trunk weight.
        policy = load_trained_policy(artifact=artifact, ckpt=ckpt)
        trunk = policy.encoder
        if not isinstance(trunk, CausalFrameTransformer):
            msg = (
                "checkpoint's encoder is a "
                f"{type(trunk).__name__}, not a CausalFrameTransformer: this "
                "checkpoint is not from a decoder-only (causal) arm and has no "
                "cache-safe positional encoding to export"
            )
            raise TypeError(msg)
        dim, layers, heads = trunk.dim_model, trunk.num_layers, trunk.num_heads
        if context is None:
            context = trunk.window
            if context is None:
                msg = "--context is required: the checkpoint's trunk has window=None"
                raise ValueError(msg)
        elif trunk.window is not None and context != trunk.window:
            # Not fatal -- a latency curve legitimately wants other contexts --
            # but the resulting engine is NOT servable with this checkpoint.
            logger.warning(
                "context != trained window: LATENCY ARTIFACT ONLY, do not serve "
                "this engine -- the trunk has no intrinsic maximum length, so it "
                "will run against this cache and silently extrapolate "
                "(docs/decoder_only_kv_cache.md §10.3)",
                context=context,
                trained_window=trunk.window,
            )
        logger.info(
            "trained trunk",
            dim=dim,
            layers=layers,
            heads=heads,
            trained_window=trunk.window,
            rope_base=trunk.rope_base,
            attention_impl=f"{trunk.attention_impl} (step is always sdpa)",
        )
    else:
        if context is None:
            msg = "--context is required for a randomly initialized export"
            raise ValueError(msg)
        # `episode_length=context`, not 1: the config's `max_sequence_length` is
        # `episode_length * tokens_per_frame` and the trunk cross-checks it
        # against `window * tokens_per_frame` at construction, so a 1-frame clip
        # makes any multi-camera arm raise before it can be replaced below.
        policy, dim, layers, heads = build_policy(arm, episode_length=context)
        configured = policy.encoder
        policy.encoder = CausalFrameTransformer(
            dim_model=dim,
            num_layers=layers,
            num_heads=heads,
            # derive the frame width from the policy's OWN layout -- cameras, and
            # the register + readout tokens a `use_readout_token` arm appends
            tokens_per_frame=tokens_per_frame(
                len(policy.cameras),
                use_readout_token=getattr(policy, "use_readout_token", False),
                num_register_tokens=getattr(policy, "num_register_tokens", 0),
            ),
            window=context,
            # carry the arm's intra-frame position parameterization over. The
            # composed table is the same shape in every arm, but the factorized
            # ones put an index_select/matmul in the decode graph, and a random
            # export exists precisely to check that those fold away -- silently
            # exporting a flat table here would make this gate vacuous.
            **_intra_position_kwargs(configured),
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
            **{
                f"image_{camera}": torch.rand(1, 1, 3, IMAGE_HW, IMAGE_HW)
                for camera in policy.cameras
            },
            "speed": torch.rand(1, 1, 1) * 130,
            "waypoints": torch.rand(1, 1, NUM_WAYPOINTS, 2) * 2 - 1,
            "past_k": past_k,
            "past_v": past_v,
            "cache_bias": cache_bias,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        },
    )
    return (
        step,
        args,
        (layers, heads, dim // heads, cache_frames, step.trunk.tokens_per_frame),
    )


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
    parser.add_argument(
        "--arm",
        choices=sorted(ARMS),
        default="small",
        help="architecture for a RANDOM export; ignored with --artifact/--ckpt, "
        "where the architecture comes from the checkpoint's hparams",
    )
    parser.add_argument("--mode", choices=["baseline", "decoder"], required=True)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument(
        "--artifact", help="trained wandb model artifact, e.g. yaak/rmind/model-<id>:v0"
    )
    weights.add_argument("--ckpt", help="trained local checkpoint path")
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="frames attended in TOTAL (decoder: 1 new + context-1 cached). "
        "Defaults to the checkpoint trunk's trained `window`; required for a "
        "random export. Any other value is a latency artifact -- not servable.",
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
        if args.context is None:
            msg = "--context is required in baseline mode"
            raise ValueError(msg)
        model = (
            load_trained_policy(artifact=args.artifact, ckpt=args.ckpt)
            if (args.artifact or args.ckpt)
            else build_policy(args.arm, episode_length=args.context)[0]
        )
        export_args: tuple[Any, ...] = baseline_args(args.context, model.cameras)
    else:
        model, export_args, shapes = decoder_model_and_args(
            args.arm, args.context, artifact=args.artifact, ckpt=args.ckpt
        )
        layers, heads, head_dim, cache_frames, k = shapes
        logger.info(
            "cache",
            layers=layers,
            heads=heads,
            head_dim=head_dim,
            cache_frames=cache_frames,
            tokens_per_frame=k,
            cached_keys=cache_frames * k,
        )
    logger.info(
        "parameters",
        millions=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    )
    export(model, export_args, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
