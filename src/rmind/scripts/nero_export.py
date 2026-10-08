"""Export a robot-native `NeroPatchPolicy` for serving (contract v3) -- with gates.

    python -m rmind.scripts.nero_export --ckpt model.ckpt --stats STATS_DIR --out ART
    python -m rmind.scripts.nero_export --experiment yaak/nero_robot/causal \\
        --random --stats STATS_DIR --out ART          # architecture/latency only

Writes into `--out`:

| file                         | what |
|------------------------------|------|
| `policy.onnx`                | `NeroPatchPolicyDecoderStep`, fp32, opset from torch's dynamo exporter |
| `tokenizer.pt`               | the frozen tokenizer (state_dict + hparams; in-graph, shipped for audit) |
| `action_standardizer.json`   | the tokenizer's relative-mode action standardizer (nutron_standardizer v1) |
| `state_standardizer.json`    | the policy's state standardizer (in-graph) |
| `hand_standardizer.json`     | the hand token's per-column affine (in-graph; only with a hand token) |
| `policy_manifest.json`       | the contract v3 payload minus schema/version/family; nutron-cli's `patch_contract_from_manifest` turns it into `policy_contract.json` |
| `export_report.json`         | the gates, latency and FLOP numbers below |

GATES (non-zero exit on failure):

1. **streaming == windowed** (#269's serving correctness gate): a
   `--gate-frames` (40) frame episode streamed through the decoder step with a
   ring of `window - 1` frames must equal ONE windowed forward of the trained
   model over the whole episode: max |diff| <= `--tol` (1e-4) on the
   standardized chunk and identical argmax codes, every frame.
2. **ONNX == eager**: ONNX Runtime (CPU) against the eager step on the first
   `--ort-frames` streamed frames (warm cache included): same tolerance on
   `actions`, identical `codes`, and new_k/new_v.
3. nutron-cli's own `patch_contract_from_manifest` validates the manifest and
   `binding_problems` checks the REAL ONNX bindings -- what serving runs before
   the first step. The checkout is `--nutron-cli` (default `$NUTRON_CLI_ROOT`).
   A refusal is recorded in `failures` (the report is still written); without a
   checkout the gate is reported `skipped` in `nutron_cli.status` and
   `warnings` -- never as a pass -- and `--require-nutron-cli` makes that a
   failure.

With `--ckpt`, the checkpoint's sha256 goes into `export_report.json`;
`mac_factory` (the Mac torch backend) refuses a `$NERO_POLICY_CKPT` that hashes
differently, and in every case one whose relative mode, window/KV geometry,
token layout, hand groups, tokenizer or standardizer shas differ from the
contract (`contract_mismatches`).

Latency is measured per decoder step on the local GPU (torch eager fp32, and
ONNX Runtime CUDA when available) -- an RTX 5090 number, NOT the Orin budget;
the Orin measurement is the user's (pitfall 3). The analytic FLOP count per step
is reported so the two can be related.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from structlog import get_logger
from torch import Tensor

from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.data import nero_image
from rmind.data.nero_robot import (
    RELATIVE_ANCHOR,
    AxisStandardizer,
    HandTokenStandardizer,
    flat_names,
    relative_mask,
)
from rmind.datamodules.nero_robot_random import nero_robot_batch
from rmind.models.nero_patch_policy import NeroPatchPolicy
from rmind.models.nero_patch_policy_decoder import NeroPatchPolicyDecoderStep

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
MASK_BIAS = -1e4
NATIVE_WH = {"base": (1920, 1080), "side_left": (1280, 800), "side_right": (1280, 800)}
INPUT_ORDER = (
    "images",
    "state",
    "side_valid",
    "past_k",
    "past_v",
    "cache_bias",
    "rope_cos",
    "rope_sin",
    "hand_token",
)
OUTPUT_ORDER = ("actions", "new_k", "new_v", "codes")
# mac_factory refuses a checkpoint the export did not pin, unless this is "1"
ALLOW_UNPINNED_ENV = "NERO_ALLOW_UNPINNED_CKPT"


# ------------------------------------------------------------------- loading


def build_from_experiment(experiment: str, overrides: list[str]) -> NeroPatchPolicy:
    from hydra import compose, initialize_config_dir  # noqa: PLC0415
    from hydra.utils import instantiate  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="train", overrides=[f"experiment={experiment}", *overrides]
        )
    return instantiate(OmegaConf.to_container(cfg.model, resolve=True))


def load_policy(ckpt: Path) -> NeroPatchPolicy:
    return NeroPatchPolicy.load_from_checkpoint(
        ckpt, map_location="cpu", weights_only=False
    )


def hand_features_module(nutron_cli: Path | None) -> Any:
    """The shared builder: rbyte's vendored copy, else nutron-cli's own file.

    Raises:
        ImportError: when neither is available.
    """
    try:
        from rbyte.samples.nero._vendor import hand_features  # noqa: PLC0415, PLC2701
    except ImportError:
        pass
    else:
        return hand_features
    if nutron_cli is not None:
        return import_file(nutron_cli / "runtime" / "jetson" / "hand_features.py")
    msg = "no hand_features available (install rbyte>=robot ingestion or pass --nutron-cli)"
    raise ImportError(msg)


def import_file(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_nutron_{path.stem}", path)
    if spec is None or spec.loader is None:
        msg = f"cannot import {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------- gates


def image_hw(policy: NeroPatchPolicy) -> tuple[int, int]:
    backbone = (
        policy.image_encoder[0]
        if hasattr(policy.image_encoder, "__getitem__")
        else None
    )
    size = getattr(backbone, "img_size", None) or getattr(policy, "image_hw", None)
    if size is None:
        return (140, 224)
    return (int(size[0]), int(size[1]))


@torch.no_grad()
def streaming_gate(  # noqa: PLR0914
    policy: NeroPatchPolicy,
    step: NeroPatchPolicyDecoderStep,
    batch: dict[str, Any],
    *,
    frames: int,
) -> dict[str, Any]:
    """Max |streamed - windowed| over `frames` frames, and code agreement.

    Codes are compared for EVERY valid side (`forward_all_codes`, torch-side):
    the bound `(1, Q)` output carries only the first valid side's, so a
    second-side code mismatch would otherwise pass unseen. `code_agreement` is
    over (frame, valid side) pairs; `bound_code_agreement` checks the bound
    output against the first valid side.
    """
    features = policy._features(batch)  # noqa: SLF001
    windowed, windowed_codes = policy._predict_chunk_and_codes(features[0])  # noqa: SLF001
    device = step.camera_cond.device
    past = step.empty_cache(device=device)
    worst, agree, total, bound_agree = 0.0, 0, 0, 0
    for frame in range(frames):
        inputs = step.frame_inputs(batch, frame)
        cos, sin = (x.to(device) for x in step.rope(frame))
        args = (
            inputs["images"],
            inputs["state"],
            inputs["side_valid"],
            *past,
            cos,
            sin,
            inputs.get("hand_token"),
        )
        actions, new_k, new_v, codes = step.forward_all_codes(*args)
        bound = step(*args)[3]
        past = step.advance(past, new_k, new_v)
        want = windowed[frame].permute(1, 0, 2).reshape(1, actions.shape[1], -1)
        worst = max(worst, float((actions - want).abs().max()))
        valid_sides = [
            i
            for i, v in enumerate(inputs["side_valid"][0].tolist())
            if v > 0.5  # noqa: PLR2004
        ]
        for side in valid_sides:
            agree += int(torch.equal(codes[0, side], windowed_codes[frame, side]))
            total += 1
        bound_agree += int(torch.equal(bound[0], windowed_codes[frame, valid_sides[0]]))
    return {
        "max_abs": worst,
        "code_agreement": agree / total,
        "bound_code_agreement": bound_agree / frames,
        "frames": frames,
        "sides_checked": total // frames,
    }


def ort_inputs(
    step: NeroPatchPolicyDecoderStep,
    inputs: dict[str, Tensor],
    past: tuple[Tensor, ...],
    frame: int,
) -> dict[str, Tensor]:
    cos, sin = step.rope(frame)
    named = dict(inputs)
    named |= {
        "past_k": past[0],
        "past_v": past[1],
        "cache_bias": past[2],
        "rope_cos": cos,
        "rope_sin": sin,
    }
    return named


def export_onnx(
    step: NeroPatchPolicyDecoderStep, example: dict[str, Tensor], out: Path
) -> None:
    names = [n for n in INPUT_ORDER if n in example]
    args = tuple(example[n] for n in names)
    exported = torch.export.export(step, args=args, strict=False)
    torch.onnx.export(
        model=exported,
        f=out,
        dynamo=True,
        external_data=False,
        optimize=True,
        input_names=list(names),
        output_names=list(OUTPUT_ORDER),
        report=False,
    )


@torch.no_grad()
def ort_gate(
    step: NeroPatchPolicyDecoderStep,
    onnx_path: Path,
    batch: dict[str, Any],
    *,
    frames: int,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort  # noqa: PLC0415

    session = ort.InferenceSession(
        str(onnx_path), providers=providers or ["CPUExecutionProvider"]
    )
    bound = {i.name for i in session.get_inputs()}
    past = step.empty_cache()
    worst = {"actions": 0.0, "new_k": 0.0, "new_v": 0.0}
    codes_equal = True
    for frame in range(frames):
        named = ort_inputs(step, step.frame_inputs(batch, frame), past, frame)
        eager = step(*(named[n] for n in INPUT_ORDER if n in named))
        feeds = {k: v.numpy() for k, v in named.items() if k in bound}
        outputs = dict(
            zip(OUTPUT_ORDER, session.run(list(OUTPUT_ORDER), feeds), strict=True)
        )
        for i, name in enumerate(("actions", "new_k", "new_v")):
            diff = (torch.from_numpy(outputs[name]) - eager[i]).abs().max()
            worst[name] = max(worst[name], float(diff))
        codes_equal &= bool((torch.from_numpy(outputs["codes"]) == eager[3]).all())
        past = step.advance(past, eager[1], eager[2])
    return {"max_abs": worst, "codes_equal": codes_equal, "frames": frames}


@torch.no_grad()
def latency(
    step: NeroPatchPolicyDecoderStep,
    example: dict[str, Tensor],
    *,
    onnx_path: Path | None,
    iters: int = 50,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = [n for n in INPUT_ORDER if n in example]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_step = step.to(device)
        args = [example[n].to(device) for n in names]
        times = []
        for i in range(iters + 10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_step(*args)
            torch.cuda.synchronize()
            if i >= 10:  # noqa: PLR2004
                times.append((time.perf_counter() - t0) * 1e3)
        out["torch_cuda_fp32_ms"] = {
            "p50": statistics.median(times),
            "p95": sorted(times)[int(0.95 * len(times)) - 1],
            "device": torch.cuda.get_device_name(),
        }
        step.cpu()
    if onnx_path is not None:
        import onnxruntime as ort  # noqa: PLC0415

        if "CUDAExecutionProvider" in ort.get_available_providers():
            session = ort.InferenceSession(
                str(onnx_path), providers=["CUDAExecutionProvider"]
            )
            feeds = {n: example[n].numpy() for n in names}
            times = []
            for i in range(iters + 10):
                t0 = time.perf_counter()
                session.run(None, feeds)
                if i >= 10:  # noqa: PLR2004
                    times.append((time.perf_counter() - t0) * 1e3)
            out["ort_cuda_fp32_ms"] = {
                "p50": statistics.median(times),
                "p95": sorted(times)[int(0.95 * len(times)) - 1],
            }
    return out


def trunk_window(trunk: CausalFrameTransformer) -> int:
    """The trunk's KV window, which the streaming export requires.

    Raises:
        TypeError: for an unwindowed trunk (`window=None`).
    """
    if trunk.window is None:
        msg = "the streaming export needs a windowed trunk, got window=None"
        raise TypeError(msg)
    return int(trunk.window)


def flops_per_step(
    policy: NeroPatchPolicy, *, cache_frames: int, vit_gflops: float
) -> dict[str, float]:
    """Analytic per-step FLOPs (2 x MACs): trunk linear + attention, plus the ViTs."""
    trunk = cast("CausalFrameTransformer", policy.encoder)
    d, layers, t = trunk.dim_model, trunk.num_layers, trunk.tokens_per_frame
    mlp = 4 * d
    linear = layers * t * (4 * d * d + 2 * d * mlp) * 2
    attention = layers * t * (cache_frames + 1) * t * d * 2 * 2
    vit = vit_gflops * 1e9 * len(policy.cameras)
    return {
        "trunk_linear_gflop": linear / 1e9,
        "trunk_attention_gflop": attention / 1e9,
        "vit_gflop": vit / 1e9,
        "total_gflop": (linear + attention + vit) / 1e9,
    }


# ------------------------------------------------------------------ manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_layout(
    policy: NeroPatchPolicy, hw: tuple[int, int], patch_size: int
) -> list[list[Any]]:
    per_cam = (hw[0] // patch_size) * (hw[1] // patch_size)
    layout: list[list[Any]] = [["state", 1]]
    if policy.use_hand:
        layout.append(["hand", policy.n_hand_tokens])
    return layout + [[f"patch:{c}", per_cam] for c in policy.cameras]


def policy_relative_mask(policy: NeroPatchPolicy, n_sides: int) -> list[bool]:
    return [bool(v) for v in relative_mask(policy.relative_mode).tolist()] * n_sides


def manifest(  # noqa: PLR0913
    *,
    policy: NeroPatchPolicy,
    step: NeroPatchPolicyDecoderStep,
    hw: tuple[int, int],
    patch_size: int = 14,
    camera_cond: Tensor,
    camera_cond_placeholder: bool,
    stats: dict[str, Any],
    hand_spec: dict[str, Any] | None,
    n_next_actions: int,
    min_start_index: int,
    onnx_io: dict[str, dict[str, dict[str, Any]]],
    files: dict[str, str],
) -> dict[str, Any]:
    trunk = step.trunk
    tokenizer = policy.tokenizer
    window = trunk_window(trunk)
    layout = token_layout(policy, hw, patch_size)
    cameras = [
        {"name": c} | nero_image.geometry(NATIVE_WH[c], hw).as_dict()
        for c in policy.cameras
    ]
    sides = ["left", "right"]
    side_valid = stats.get("side_valid", [True, False])
    return (
        {
            "policy_type": "nero_patch_policy",
            "model": {
                "format": "onnx",
                "file": files["model"],
                "sha256": None,
                "precision": "fp32",
            },
            "camera_fps": 30,
            "frame_stride": 3,
            "action_fps": 30,
            "window_frames": window,
            "tokens_per_frame": int(trunk.tokens_per_frame),
            "token_layout": layout,
            "cameras": cameras,
            "image": {
                "preprocessing_id": nero_image.PREPROCESSING_ID,
                "preprocessing_sha256": nero_image.preprocessing_sha256(),
                "patch_size": patch_size,
                "value_range": "unit",
                "channel_order": "rgb",
                "normalize_in_graph": True,
                "pad_value": float(nero_image.PAD_VALUE),
            },
            "camera_cond": {
                "values": camera_cond.reshape(len(policy.cameras), 13).tolist(),
                "placeholder": camera_cond_placeholder,
            },
            "sides": sides,
            "side_valid": [bool(v) for v in side_valid],
            "state_names": flat_names(sides),
            "action_names": flat_names(sides),
            "chunk_size": int(tokenizer.action_horizon),
            "chunk_t0_offset_steps": 0,
            "tokenizer": {
                "file": files["tokenizer"],
                "sha256": None,
                "num_quantizers": int(tokenizer.quantizer.num_quantizers),
                "codebook_size": int(tokenizer.quantizer.codebook_size),
                "keyframe_stride": int(tokenizer.keyframe_stride),
                "interpolation": "linear",
                "in_graph": True,
            },
            "standardizers": {
                "state": {
                    "file": files["state_standardizer"],
                    "sha256": None,
                    "in_graph": True,
                },
                "action": {
                    "file": files["action_standardizer"],
                    "sha256": None,
                    "in_graph": False,
                },
            }
            # the hand token's feature-column affine: in-graph (serving feeds the
            # raw hf.build_token vector), shipped + hashed so serving refuses an
            # artifact whose file does not match. Present iff the graph has one.
            | (
                {
                    "hand": {
                        "file": files["hand_standardizer"],
                        "sha256": None,
                        "in_graph": True,
                    }
                }
                if "hand_standardizer" in files
                else {}
            ),
            "absolute_action_stats": {k: stats[k] for k in ("min", "max", "q50")},
            "relative_mode": policy.relative_mode,
            "relative_mask": policy_relative_mask(policy, len(sides)),
            "relative_anchor": RELATIVE_ANCHOR,
            "n_next_actions": n_next_actions,
            "min_start_index": min_start_index,
            "max_missed_ticks": 0,
            "hand": hand_spec,
        }
        | (
            # contract v3 `hand_sides` (WP0): one side-tagged token per entry; absent
            # = the one untagged token every pre-bimanual artifact has
            {"hand_sides": list(policy.hand_sides)} if policy.hand_sides else {}
        )
        | {
            # the graph has no goal input whether the policy trained "no_goal"
            # (learned no_goal concatenated in-graph) or "none" (no goal channel)
            "goal_mode": "none",
            "depth": None,
            "kv": {
                "num_layers": int(trunk.num_layers),
                "num_heads": int(trunk.num_heads),
                "head_dim": int(trunk.head_dim),
                "cache_frames": window - 1,
                "dtype": "float32",
                "mask_bias": MASK_BIAS,
                "rope_base": float(trunk.rope_base),
                "rope_counter": "frames_since_reset",
                "ring": "oldest_first_shift",
            },
            "io": onnx_io,
        }
    )


def onnx_io(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    import onnx  # noqa: PLC0415

    model = onnx.load(str(path), load_external_data=False)
    types = {1: "float32", 7: "int64", 6: "int32", 10: "float16"}

    def binding(v: Any) -> dict[str, Any]:
        t = v.type.tensor_type
        return {
            "name": v.name,
            "shape": [int(d.dim_value) for d in t.shape.dim],
            "dtype": types.get(t.elem_type, str(t.elem_type)),
        }

    return {
        "inputs": {v.name: binding(v) for v in model.graph.input},
        "outputs": {v.name: binding(v) for v in model.graph.output},
    }


def nutron_gate(
    pc_path: Path, manifest_path: Path, out: Path, onnx_path: Path
) -> dict[str, Any]:
    """Gate 3: nutron-cli's `patch_contract_from_manifest` + `binding_problems` on
    the REAL ONNX bindings. `status`: ok | refused | binding_problems."""
    import onnxruntime  # noqa: PLC0415

    pc = import_file(pc_path)
    try:
        contract = pc.patch_contract_from_manifest(manifest_path, out)
        pc.write_patch_contract(out / "policy_contract.json", contract)
    except pc.ContractError as exc:
        return {"status": "refused", "error": str(exc)}
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_types = {
        "tensor(float)": "float32",
        "tensor(int64)": "int64",
        "tensor(int32)": "int32",
        "tensor(float16)": "float16",
    }
    problems = pc.binding_problems(
        contract,
        {
            i.name: (list(i.shape), ort_types.get(i.type, i.type))
            for i in session.get_inputs()
        },
        {
            o.name: (list(o.shape), ort_types.get(o.type, o.type))
            for o in session.get_outputs()
        },
    )
    return {
        "status": "binding_problems" if problems else "ok",
        "contract": contract.summary(),
        "binding_problems": problems,
    }


def save_tokenizer(policy: NeroPatchPolicy, path: Path) -> None:
    tokenizer = policy.tokenizer
    torch.save(
        {
            "state_dict": tokenizer.state_dict(),
            "hparams": json.loads(json.dumps(dict(tokenizer.hparams), default=str)),
            "standardizer_sha256": tokenizer.standardizer.digest,
        },
        path,
    )


# ----------------------------------------------------- Mac torch backend


class RoleDecoderStep(torch.nn.Module):
    """`NeroPatchPolicyDecoderStep` behind nutron-cli's torch-backend calling
    convention (`patch_runtime.TorchModuleBackend`): `forward({io role: tensor})`
    -> `{actions, new_k, new_v, codes}`."""

    def __init__(self, step: NeroPatchPolicyDecoderStep) -> None:
        super().__init__()
        self.step = step

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self.step(*(inputs[n].float() for n in INPUT_ORDER if n in inputs))
        return dict(zip(OUTPUT_ORDER, outputs, strict=True))


def _contract_dict(contract: Any) -> dict[str, Any]:
    if hasattr(contract, "to_dict"):
        return contract.to_dict()
    if hasattr(contract, "data"):
        return json.loads(json.dumps(contract.data))
    return json.loads(json.dumps(dict(contract)))


def contract_mismatches(policy: NeroPatchPolicy, contract: Any) -> list[str]:  # noqa: PLR0914, PLR0915
    """Everything a served contract v3 states about the MODEL, checked against a
    loaded policy: [] = this policy is the one the artifact describes.

    nutron-cli's `TorchModuleBackend` reports the contract's own io block as its
    signature, so on the Mac torch path nothing else ties `$NERO_POLICY_CKPT` to
    the artifact; a checkpoint with another relative mode, tokenizer, action /
    state / hand standardizer or hand groups would otherwise serve, and
    `PatchSession` would unstandardize + re-anchor its outputs with the
    contract's (wrong) affine and mask. Standardizers are compared by the sha256
    of their nutron_standardizer JSON re-serialized from the policy against the
    contract's file sha (they are the same bytes `run` ships).
    """
    d = _contract_dict(contract)
    out: list[str] = []

    def check(key: str, have: Any, want: Any) -> None:
        if have != want:
            out.append(f"{key}: checkpoint {have!r}, contract {want!r}")

    trunk = policy.encoder
    if not isinstance(trunk, CausalFrameTransformer):
        return [f"policy.encoder is {type(trunk).__name__}, not a causal decoder"]
    window = trunk_window(trunk)
    tokenizer = policy.tokenizer
    n_sides = len(d.get("sides", ()))
    check("policy_type", "nero_patch_policy", d.get("policy_type"))
    check("action space", "robot" if policy.robot else "glove", "robot")
    # the graph has no goal input for no_goal/none; an image-goal policy has none
    # to serve, whatever the contract says
    check(
        "goal_mode",
        "image" if policy.goal_mode == "image" else "none",
        d.get("goal_mode"),
    )
    check("depth", "stream" if policy.use_depth else None, d.get("depth"))
    check(
        "cameras", list(policy.cameras), [c.get("name") for c in d.get("cameras", ())]
    )
    check("window_frames", window, d.get("window_frames"))
    kv = d.get("kv") or {}
    check("kv.cache_frames", window - 1, kv.get("cache_frames"))
    check("kv.num_layers", int(trunk.num_layers), kv.get("num_layers"))
    check("kv.num_heads", int(trunk.num_heads), kv.get("num_heads"))
    check("kv.head_dim", int(trunk.head_dim), kv.get("head_dim"))
    check("kv.rope_base", float(trunk.rope_base), kv.get("rope_base"))
    check("tokens_per_frame", int(trunk.tokens_per_frame), d.get("tokens_per_frame"))
    cams = d.get("cameras") or [{}]
    hw = cams[0].get("input_hw")
    patch_size = (d.get("image") or {}).get("patch_size")
    if hw and patch_size:
        check(
            "token_layout",
            token_layout(policy, (int(hw[0]), int(hw[1])), int(patch_size)),
            d.get("token_layout"),
        )
    check("relative_mode", policy.relative_mode, d.get("relative_mode"))
    check(
        "relative_mask", policy_relative_mask(policy, n_sides), d.get("relative_mask")
    )
    check("chunk_size", int(tokenizer.action_horizon), d.get("chunk_size"))
    tok = d.get("tokenizer") or {}
    check(
        "tokenizer.num_quantizers",
        int(tokenizer.quantizer.num_quantizers),
        tok.get("num_quantizers"),
    )
    check(
        "tokenizer.codebook_size",
        int(tokenizer.quantizer.codebook_size),
        tok.get("codebook_size"),
    )
    check(
        "tokenizer.keyframe_stride",
        int(tokenizer.keyframe_stride),
        tok.get("keyframe_stride"),
    )
    hand = d.get("hand")
    check("hand token", bool(policy.use_hand), hand is not None)
    if policy.use_hand and hand is not None:
        check("hand.groups", list(policy.hand_groups), list(hand.get("groups", ())))
    check("hand_sides", list(policy.hand_sides), list(d.get("hand_sides") or ()))
    stds = d.get("standardizers") or {}
    state_std = cast(
        "AxisStandardizer", policy.state_standardizer or AxisStandardizer()
    )
    have = {"action": tokenizer.standardizer.digest, "state": state_std.digest}
    if policy.hand_standardizer is not None:
        have["hand"] = policy.hand_standardizer.digest
    for name in sorted(set(have) | set(stds)):
        want = (stds.get(name) or {}).get("sha256")
        if name not in have:
            out.append(f"standardizers.{name}: contract has one, checkpoint has none")
        elif want is None:
            out.append(f"standardizers.{name}: contract pins no sha256")
        else:
            check(f"standardizers.{name}.sha256", have[name], want)
    return out


def mac_factory(contract: Any, artifact_dir: Path, device: str) -> torch.nn.Module:
    """nutron-cli `mac_policy_server.py --patch-backend rmind --rmind-factory
    rmind.scripts.nero_export:mac_factory`: the TRAINED model in torch.

    The artifact carries no torch weights (only the ONNX), so the checkpoint comes
    from `$NERO_POLICY_CKPT` (with the tokenizer/stats paths its hparams name).
    `camera_cond` is taken from the contract, as the ONNX has it baked in.

    REFUSES a checkpoint that is not the one the artifact describes: it must
    hash to the sha256 the export recorded in `export_report.json` (which ships
    with the contract; an artifact without one is refused unless
    `$NERO_ALLOW_UNPINNED_CKPT=1`, logged), and `contract_mismatches` must be
    empty.

    Raises:
        RuntimeError: when `$NERO_POLICY_CKPT` is not set, or the checkpoint does
            not match the artifact.
    """
    ckpt = os.environ.get("NERO_POLICY_CKPT")
    if not ckpt:
        msg = "set NERO_POLICY_CKPT to the Lightning checkpoint the artifact was exported from"
        raise RuntimeError(msg)
    report_path = Path(artifact_dir) / "export_report.json"
    pinned = None
    if report_path.is_file():
        pinned = (json.loads(report_path.read_text()).get("checkpoint") or {}).get(
            "sha256"
        )
    if pinned is None:
        # the structural check below cannot tell two epochs of one run apart
        reason = (
            f"no checkpoint sha256 pinned in {report_path} (export with --ckpt and "
            "ship export_report.json with the contract): weights identity unverified"
        )
        if os.environ.get(ALLOW_UNPINNED_ENV) != "1":
            msg = f"{reason}; set {ALLOW_UNPINNED_ENV}=1 to serve anyway"
            raise RuntimeError(msg)
        logger.warning(reason, override=f"{ALLOW_UNPINNED_ENV}=1")
    elif sha256(Path(ckpt)) != pinned:
        msg = (
            f"NERO_POLICY_CKPT {ckpt} is not the checkpoint {report_path} was "
            f"exported from (sha256 {pinned})"
        )
        raise RuntimeError(msg)
    policy = load_policy(Path(ckpt)).eval()
    problems = contract_mismatches(policy, contract)
    if problems:
        msg = f"NERO_POLICY_CKPT {ckpt} does not match the contract:\n  " + "\n  ".join(
            problems
        )
        raise RuntimeError(msg)
    logger.info(
        "mac_factory: checkpoint matches the contract",
        ckpt=ckpt,
        sha_pinned=pinned is not None,
    )
    policy.sample_codes = False
    cond = torch.tensor(contract["camera_cond"]["values"], dtype=torch.float32)
    return (
        RoleDecoderStep(NeroPatchPolicyDecoderStep(policy=policy, camera_cond=cond))
        .to(device)
        .eval()
    )


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=Path)
    source.add_argument("--experiment")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--stats", type=Path, required=True, help="nero_fit_stats output dir"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--camera-cond", type=Path, help="JSON (3, 13); default: zeros placeholder"
    )
    parser.add_argument("--n-next-actions", type=int, default=6)
    parser.add_argument("--min-start-index", type=int, default=3)
    parser.add_argument("--gate-frames", type=int, default=40)
    parser.add_argument("--ort-frames", type=int, default=3)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--vit-gflops", type=float, default=4.6, help="ViT-S/14 at 140x224"
    )
    parser.add_argument(
        "--nutron-cli",
        type=Path,
        default=Path(os.environ["NUTRON_CLI_ROOT"])
        if os.environ.get("NUTRON_CLI_ROOT")
        else None,
        help="nutron-cli checkout for gate 3 (default: $NUTRON_CLI_ROOT); without "
        "one gate 3 is reported SKIPPED, never passed",
    )
    parser.add_argument(
        "--require-nutron-cli",
        action="store_true",
        help="a skipped gate 3 (no nutron-cli checkout) is a failure",
    )
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument(
        "--gate-device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument(
        "--weights",
        type=Path,
        help="with --experiment: a {'state_dict': ...} file to load (e.g. a smoke run)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    policy = (
        load_policy(args.ckpt)
        if args.ckpt
        else build_from_experiment(args.experiment, args.override)
    )
    if args.weights is not None:
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        policy.load_state_dict(state.get("state_dict", state))
    if args.ckpt is not None:
        args.checkpoint = args.ckpt
    report = run(args, policy, hw=image_hw(policy))
    if report["failures"]:
        raise SystemExit(1)


def run(  # noqa: C901, PLR0912, PLR0914, PLR0915
    args: argparse.Namespace, policy: NeroPatchPolicy, *, hw: tuple[int, int]
) -> dict[str, Any]:
    """Gates + export + manifest for an instantiated policy (the CLI's body).

    Raises:
        RuntimeError: when the shipped hand standardizer does not round-trip.
            Or when a sided hand token (`hand_sides`) meets stats whose
            side_valid sides differ (contract v3: hand_sides == valid sides).
    """
    policy = policy.cpu().eval()
    policy.sample_codes = False

    cond = (
        torch.tensor(json.loads(args.camera_cond.read_text()), dtype=torch.float32)
        if args.camera_cond
        else torch.zeros(len(policy.cameras), 13)
    )
    step = NeroPatchPolicyDecoderStep(policy=policy, camera_cond=cond).eval()
    window = trunk_window(step.trunk)

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "window": window,
        "tokens_per_frame": step.tokens_per_frame,
    }
    warnings: list[str] = []
    nutron_cli = getattr(args, "nutron_cli", None)
    if nutron_cli is not None and not Path(nutron_cli).exists():
        warnings.append(f"--nutron-cli {nutron_cli} does not exist")
        nutron_cli = None
    # the hand-token builder BEFORE the expensive gates: a missing one must not
    # crash after the ONNX is written
    hf = hand_features_module(nutron_cli) if policy.use_hand else None
    # the stats' side_valid decides the manifest's sides: refuse a sided hand
    # token against single-arm stats BEFORE the expensive gates
    stats = json.loads((args.stats / "absolute_action_stats.json").read_text())
    if policy.hand_sides:
        stats_sides = [
            s
            for s, v in zip(
                ("left", "right"), stats.get("side_valid", ()), strict=False
            )
            if v
        ]
        if stats_sides != list(policy.hand_sides):
            # contract v3: hand_sides == the side_valid-true sides
            msg = (
                f"hand_sides {list(policy.hand_sides)} but {args.stats} has "
                f"side_valid {stats.get('side_valid')}: export with the stats the "
                "policy trained on (the bimanual fit)"
            )
            raise RuntimeError(msg)
    # the checkpoint the artifact is exported from: mac_factory refuses any other
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        report["checkpoint"] = {
            "path": Path(checkpoint).as_posix(),
            "sha256": sha256(Path(checkpoint)),
        }

    # --- gate 1: streaming == windowed (the trained window over a longer episode).
    # On the GPU when there is one (a 40-frame windowed forward is ~19k tokens),
    # in TRUE fp32: TF32 matmuls would make the comparison measure TF32 noise.
    # the gate batch carries the SAME side_valid the manifest ships (the
    # stats'): a bimanual artifact -- with or without a hand token -- is gated
    # on a BOTH-VALID batch, so every side's token, codes and chunk are exercised
    stats_side_valid = [bool(v) for v in stats.get("side_valid", [True, False])]
    if stats_side_valid not in ([True, False], [True, True]):
        msg = (
            f"{args.stats} has side_valid {stats.get('side_valid')}: the export "
            "gates know single-arm [True, False] and bimanual [True, True] only"
        )
        raise RuntimeError(msg)
    batch = nero_robot_batch(
        batch_size=1,
        num_frames=args.gate_frames,
        image_hw=hw,
        seed=7,
        bimanual=all(stats_side_valid),
    )
    gate_side_valid = [bool(v) for v in batch["side_valid"][0].tolist()]
    if gate_side_valid != stats_side_valid:
        msg = (
            f"gate batch side_valid {gate_side_valid} != the manifest's "
            f"{stats_side_valid}: the gates would not check the served sides"
        )
        raise RuntimeError(msg)
    batch["camera_cond"] = cond.reshape(1, *cond.shape)
    gate_device = torch.device(args.gate_device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    step.to(gate_device)
    gate = streaming_gate(
        policy,
        step,
        {k: v.to(gate_device) for k, v in batch.items()},
        frames=args.gate_frames,
    )
    step.cpu()
    report["streaming_vs_windowed"] = gate | {"device": str(gate_device)}
    logger.info("gate: streaming == windowed", **gate)

    # --- export
    onnx_path = args.out / "policy.onnx"
    past = step.empty_cache()
    example = ort_inputs(step, step.frame_inputs(batch, 0), past, 0)
    export_onnx(step, example, onnx_path)
    report["onnx_mb"] = round(onnx_path.stat().st_size / 1e6, 2)

    # --- gate 2: ONNX Runtime CPU == eager
    ort = ort_gate(step, onnx_path, batch, frames=args.ort_frames)
    report["ort_vs_eager"] = ort
    logger.info("gate: ORT == eager", **ort)

    # --- artifacts
    files = {
        "model": "policy.onnx",
        "tokenizer": "tokenizer.pt",
        "state_standardizer": "state_standardizer.json",
        "action_standardizer": "action_standardizer.json",
    }
    save_tokenizer(policy, args.out / files["tokenizer"])
    policy.tokenizer.standardizer.save(args.out / files["action_standardizer"])
    state_std = cast(
        "AxisStandardizer", policy.state_standardizer or AxisStandardizer()
    )
    state_std.save(args.out / files["state_standardizer"])
    if policy.hand_standardizer is not None:
        files["hand_standardizer"] = "hand_standardizer.json"
        digest = policy.hand_standardizer.save(args.out / files["hand_standardizer"])
        # the shipped file must BE the in-graph affine: reload it and compare
        reloaded = HandTokenStandardizer.load(
            args.out / files["hand_standardizer"], groups=policy.hand_groups
        )
        if not (
            torch.equal(reloaded.full_mean, policy.hand_standardizer.full_mean.cpu())
            and torch.equal(reloaded.full_std, policy.hand_standardizer.full_std.cpu())
        ):
            msg = "hand_standardizer.json does not round-trip the in-graph affine"
            raise RuntimeError(msg)
        report["hand_standardizer"] = {
            "sha256": digest,
            "source": policy.hand_standardizer.source,
        }
    elif policy.use_hand:
        logger.warning("hand token WITHOUT a hand standardizer (legacy checkpoint)")
    hand_spec = None
    if hf is not None:
        hand_spec = hf.token_spec(policy.hand_groups)
    payload = manifest(
        policy=policy,
        step=step,
        hw=hw,
        patch_size=args.patch_size,
        camera_cond=cond,
        camera_cond_placeholder=args.camera_cond is None,
        stats=stats,
        hand_spec=hand_spec,
        n_next_actions=args.n_next_actions,
        min_start_index=args.min_start_index,
        onnx_io=onnx_io(onnx_path),
        files=files,
    )
    manifest_path = args.out / "policy_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    report["files"] = {name: sha256(args.out / f) for name, f in files.items()}

    # --- gate 3: nutron-cli's own validation (read-only import). A refusal is a
    # recorded FAILURE (the report is still written); no checkout is SKIPPED.
    pc_path = (
        Path(nutron_cli) / "runtime" / "jetson" / "policy_contract.py"
        if nutron_cli is not None
        else None
    )
    if pc_path is None or not pc_path.exists():
        report["nutron_cli"] = {
            "status": "skipped",
            "reason": "no nutron-cli checkout (--nutron-cli / $NUTRON_CLI_ROOT)"
            if pc_path is None
            else f"{pc_path} not found",
        }
        warnings.append("gate 3 (nutron-cli contract + bindings) SKIPPED")
        logger.warning("gate 3 SKIPPED: no nutron-cli", **report["nutron_cli"])
    else:
        report["nutron_cli"] = nutron_gate(pc_path, manifest_path, args.out, onnx_path)
        logger.info("nutron-cli contract", **report["nutron_cli"])

    flops = flops_per_step(policy, cache_frames=window - 1, vit_gflops=args.vit_gflops)
    report["flops_per_step"] = flops
    if not args.no_latency:
        report["latency"] = latency(step, example, onnx_path=onnx_path)
        logger.info("latency (local GPU, NOT Orin)", **report["latency"])

    failures = []
    if (
        gate["max_abs"] > args.tol
        or gate["code_agreement"] < 1.0
        or gate["bound_code_agreement"] < 1.0
    ):
        failures.append("streaming != windowed")
    if max(ort["max_abs"].values()) > args.tol or not ort["codes_equal"]:
        failures.append("ONNX != eager")
    gate3 = report["nutron_cli"]
    if gate3["status"] == "refused":
        failures.append("nutron-cli refused the contract")
    elif gate3["status"] == "binding_problems":
        failures.append("nutron-cli binding problems")
    elif gate3["status"] == "skipped" and getattr(args, "require_nutron_cli", False):
        failures.append("nutron-cli gate skipped (--require-nutron-cli)")
    report["failures"] = failures
    report["warnings"] = warnings
    (args.out / "export_report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n"
    )
    if warnings:
        logger.warning("export warnings", warnings=warnings)
    if failures:
        logger.error("export gates FAILED", failures=failures)
    else:
        logger.info("export ok", out=args.out.as_posix())
    return report


if __name__ == "__main__":
    main()
