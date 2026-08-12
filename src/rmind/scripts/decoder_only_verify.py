"""Correctness + precision gates for a TRAINED decoder-only PatchPolicy export.

`decoder_only_export.py` produces the ONNX; this script is everything that must
pass *before* a TRT engine built from it may be trusted, plus the trial-set
generation that the on-Orin parity ladder consumes.

Random weights could establish latency and memory (docs/decoder_only_kv_cache.md
§9, §6) but not any of this: every gate here is a statement about weight
*statistics*, so it needs a real checkpoint.

Subcommands, in the order they should be run:

    # 1. the architectural gate, now with trained weights.
    #    Streaming against a ring of `window - 1` frames == one full forward
    #    under frame_block_causal_mask(window). float64 pins it as exact;
    #    float32 shows the accumulation-order residual. Both carry a negative
    #    control, so a pass is falsifiable.
    python -m rmind.scripts.decoder_only_verify gates \
        --artifact yaak/rmind/model-do8m9ot8:v0

    # 2. ONNX-vs-eager. Report ABSOLUTE error against the tensor's own scale --
    #    torch.onnx.export(verify=True) reports per-element RELATIVE error and
    #    screams about near-zero K/V entries (§5, a documented false alarm).
    python -m rmind.scripts.decoder_only_verify onnx-vs-eager \
        --artifact yaak/rmind/model-do8m9ot8:v0 --onnx decoder_n16.onnx

    # 3. the margin screen (trt-export skill §4a): how close is each in-graph
    #    ArgMax to flipping, in fp16 ULPs? ORT-only, no GPU. min ULP < 1 means
    #    low precision WILL flip decisions.
    python -m rmind.scripts.decoder_only_verify margins --onnx decoder_n16.onnx

    # 4. the n=200 trial set + its ORT fp32 reference, and raw .dat files for
    #    trtexec --loadInputs on the Orin (there is no python tensorrt there).
    python -m rmind.scripts.decoder_only_verify trials \
        --onnx decoder_n16.onnx --trials 200 --out trialset/

    # 5. score engine outputs collected on the Orin against that reference
    python -m rmind.scripts.decoder_only_verify score \
        --trialset trialset/ --outputs fp16/ --label fp16

Why the trial set holds `--histories` distinct caches rather than 200
--------------------------------------------------------------------
A trial's `past_k`/`past_v` are 126 MiB at window 16, so 200 independent caches
would be 25 GiB to generate, store and copy to the Orin. Instead the set holds
`--histories` genuinely *streamed* caches -- each one built by running the eager
model tick by tick over a real frame sequence, exactly as drivr would -- and
varies the current frame, speed and waypoints within each. The cache is the
slowest-moving part of the real input (it is driving history), so this is the
cheap axis to share; it is still a limitation and is reported as one.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from structlog import get_logger
from torch import Tensor

from rmind.components.transformer.causal_frame import (
    CausalFrameTransformer,
    frame_rope_cos_sin,
)
from rmind.models.patch_policy_decoder import PatchPolicyDecoderStep
from rmind.scripts.decoder_only_export import load_trained_policy

logger = get_logger(__name__)

# `policy.joint_actions` channels. The tokenizer's continuous normalizer is
# Identity, so gas/brake/steering are already in their `*_normalized` units.
# `turn_signal` has a far larger dynamic range and dominates any single scalar
# summary -- report per channel (trt-export skill §4).
CHANNELS = ("gas_pedal", "brake_pedal", "steering_angle", "turn_signal")
DEFAULT_FRAMES = "/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg"  # noqa: S108
# |diff| above this counts as a changed DECISION rather than float noise: float
# agreement is ~1e-4 and a flipped code moves an action by ~0.1.
CODE_TOL = 0.02


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #


def _trained_step(artifact: str | None, ckpt: str | None) -> PatchPolicyDecoderStep:
    policy = load_trained_policy(artifact=artifact, ckpt=ckpt)
    if not isinstance(policy.encoder, CausalFrameTransformer):
        msg = f"encoder is {type(policy.encoder).__name__}, not CausalFrameTransformer"
        raise TypeError(msg)
    return PatchPolicyDecoderStep(policy=policy).eval()


def _scale_rel(got: Tensor, ref: Tensor) -> tuple[float, float]:
    """`(max abs diff, max abs diff / max abs ref)`.

    Scale-relative, never per-element relative: `new_k` has entries at ~1e-7
    beside entries at ~5, and dividing elementwise turns fp32 round-off on the
    former into a "relative error of 2.7" (§5).
    """
    diff = (got - ref).abs().max().item()
    scale = ref.abs().max().item()
    return diff, (diff / scale if scale else float("nan"))


def _real_frames(spec: str, size: int) -> list[np.ndarray]:
    """Center-cropped, resized `[0,1]` RGB CHW frames from real camera images.

    Real frames are not a nicety: synthetic noise both understated one measured
    failure (4/10 vs 8/10) and manufactured a false one elsewhere, so the
    harness refuses to run without at least one readable image. PIL rather than
    cv2 because the rmind venv has no cv2 and must not be re-synced (ptars has
    no wheel, so `uv sync` destroys it).

    Raises:
        SystemExit: if not one frame in `spec` could be read.
    """
    from PIL import Image  # noqa: PLC0415

    out: list[np.ndarray] = []
    for name in (s.strip() for s in spec.split(",")):
        if not name:
            continue
        try:
            img = Image.open(name).convert("RGB")
        except OSError:
            logger.warning("unreadable frame, skipping", path=name)
            continue
        w, h = img.size
        side = min(h, w)
        img = img.crop((
            (w - side) // 2,
            (h - side) // 2,
            (w - side) // 2 + side,
            (h - side) // 2 + side,
        )).resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        out.append(arr.astype(np.float32))
    if not out:
        msg = f"no readable frames in {spec!r}: the harness REQUIRES real camera frames"
        raise SystemExit(msg)
    logger.info("real frames", count=len(out))
    return out


def _image_input_names(meta: Mapping[str, Any]) -> list[str]:
    """The graph's `inputs_image_<camera>` names, any camera count.

    One flat binding per camera (`PatchPolicyDecoderStep`'s runtime contract),
    so this is just a name filter -- no positional/order assumption, since
    every camera is bound by name.

    Raises:
        ValueError: if no `inputs_image_<camera>` binding is present.
    """
    names = sorted(name for name in meta if name.startswith("inputs_image_"))
    if not names:
        msg = f"no inputs_image_<camera> binding found among {sorted(meta)}"
        raise ValueError(msg)
    return names


def _small_inputs(
    rng: np.random.Generator,
    camera_frames: Mapping[str, np.ndarray],
    num_waypoints: int,
) -> dict[str, np.ndarray]:
    """The per-tick inputs: one real frame per camera plus synthetic speed and
    waypoints.

    `camera_frames` keys are already the full ONNX input names
    (`inputs_image_<camera>`, from `_image_input_names`) mapped to a real frame
    for that camera this tick. Synthetic speed/waypoints is what makes this
    harness ~7x harsher than the road (trt-export skill); the image must
    nonetheless be real, because noise both understated and manufactured
    failures when it was not. The SAME jitter is applied to every camera this
    tick (independent per-camera jitter would need independent real footage per
    camera, which the harness does not have).
    """
    jitter = np.float32(rng.uniform(0.92, 1.08))
    out = {
        name: np.clip(frame * jitter, 0, 1)[None, None].astype(np.float32)
        for name, frame in camera_frames.items()
    }
    out["inputs_speed"] = np.full((1, 1, 1), rng.uniform(0, 40), np.float32)
    out["inputs_waypoints"] = np.stack(
        [
            rng.normal(0, 0.02, num_waypoints).astype(np.float32),
            (np.arange(num_waypoints, dtype=np.float32) + 1)
            * np.float32(rng.uniform(2, 25) / 100.0),
        ],
        axis=1,
    )[None, None].astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# 1. the streaming / full-window equivalence gate, with trained weights
# --------------------------------------------------------------------------- #


def _readouts(flat: Tensor, num_frames: int) -> Tensor:
    b, s, d = flat.shape
    return flat.reshape(b, num_frames, s // num_frames, d)[:, :, -1]


def _stream(
    trunk: CausalFrameTransformer,
    tokens: Tensor,
    *,
    cache_frames: int,
    frozen_rope: bool = False,
) -> Tensor:
    """One frame per tick against a host-side ring, i.e. what drivr does.

    `frozen_rope` is the NEGATIVE CONTROL: it feeds every tick the rotation of
    frame 0, which is what a runtime that forgot to advance the frame counter
    would do (§7 failure mode 3). It must miss by orders of magnitude, otherwise
    the gate is not measuring the positional scheme at all.
    """
    b, num_frames, k, _ = tokens.shape
    past_k, past_v, bias = trunk.empty_cache(
        batch_size=b, cache_frames=cache_frames, dtype=tokens.dtype
    )
    outs: list[Tensor] = []
    for t in range(num_frames):
        cos, sin = frame_rope_cos_sin(
            torch.tensor(0 if frozen_rope else t),
            head_dim=trunk.head_dim,
            base=trunk.rope_base,
        )
        out, new_k, new_v = trunk.step(
            tokens[:, t],
            past_k=past_k,
            past_v=past_v,
            cos=cos.to(tokens.dtype),
            sin=sin.to(tokens.dtype),
            cache_bias=bias,
        )
        outs.append(out[:, -1])
        if not cache_frames:
            continue
        # ring slot write: touch ONE frame block, move nothing (§7). Valid
        # because attention is permutation-invariant over keys and every key
        # carries its own rotation and its own cache_bias entry.
        slot = slice((t % cache_frames) * k, (t % cache_frames + 1) * k)
        past_k, past_v, bias = past_k.clone(), past_v.clone(), bias.clone()
        past_k[..., slot, :] = new_k
        past_v[..., slot, :] = new_v
        bias[..., slot] = 0.0
    return torch.stack(outs, dim=1)


def cmd_gates(args: argparse.Namespace) -> int:  # noqa: PLR0914
    step = _trained_step(args.artifact, args.ckpt)
    trunk = step.trunk
    window = args.window or trunk.window
    if window is None:
        msg = "--window required: the trunk has window=None"
        raise SystemExit(msg)
    # The full forward needs the DENSE mask path: FlexAttention has no CPU
    # backward, is torch.compile-only to be block-sparse, and is numerically
    # equivalent to sdpa anyway (§11.2). `step` is always sdpa regardless.
    trunk.attention_impl = "sdpa"
    for layer in trunk.layers:
        layer.attn.attention_impl = "sdpa"

    num_frames = window + 1  # one frame past a full window: the sliding case
    k = trunk.tokens_per_frame
    g = torch.Generator().manual_seed(1337)
    tokens32 = torch.randn(1, num_frames, k, trunk.dim_model, generator=g)

    logger.info(
        "gate geometry",
        window=window,
        num_frames=num_frames,
        tokens_per_frame=k,
        cached_keys=(window - 1) * k,
        seq_len=num_frames * k,
    )

    results: dict[str, Any] = {"window": window, "num_frames": num_frames}
    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        t = trunk.to(dtype)
        tokens = tokens32.to(dtype)
        flat = tokens.reshape(1, num_frames * k, trunk.dim_model)
        with torch.inference_mode():
            recompute = _readouts(t(flat, num_frames=num_frames), num_frames)
            streamed = _stream(t, tokens, cache_frames=window - 1)
            control = _stream(t, tokens, cache_frames=window - 1, frozen_rope=True)
        abs_d, rel_d = _scale_rel(streamed, recompute)
        c_abs, c_rel = _scale_rel(control, recompute)
        results[dtype_name] = {
            "abs": abs_d,
            "rel": rel_d,
            "control_abs": c_abs,
            "control_rel": c_rel,
        }
        logger.info(
            "streaming == full windowed forward",
            dtype=dtype_name,
            abs_diff=f"{abs_d:.3e}",
            scale_rel=f"{rel_d:.3e}",
            negative_control_rel=f"{c_rel:.3e}",
        )
    trunk.to(torch.float32)

    # float64 must be at machine epsilon (the equivalence is EXACT, so any
    # larger residual is a real defect), float32 only at accumulation noise.
    ok = (
        results["float64"]["rel"] < 1e-12  # noqa: PLR2004
        and results["float32"]["rel"] < 1e-4  # noqa: PLR2004
        and results["float64"]["control_rel"] > 1e-3  # noqa: PLR2004
    )
    logger.info("GATE", passed=ok)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 2. ONNX vs eager
# --------------------------------------------------------------------------- #


def _eager_feed(
    step: PatchPolicyDecoderStep, cache_frames: int, seed: int
) -> dict[str, Tensor]:
    """A warm-cache input set. `past_k`/`past_v` are randn, which is the right
    SCALE for real keys but not real keys -- `trials` streams the real thing."""
    g = torch.Generator().manual_seed(seed)
    past_k, past_v, bias = step.empty_cache(cache_frames=cache_frames)
    cos, sin = step.rope(cache_frames)
    return {
        **{
            f"image_{camera}": torch.rand(1, 1, 3, 224, 224, generator=g)
            for camera in step.policy.cameras
        },
        "speed": torch.rand(1, 1, 1, generator=g) * 130,
        "waypoints": torch.rand(1, 1, 10, 2, generator=g) * 2 - 1,
        "past_k": torch.randn(past_k.shape, generator=g),
        "past_v": torch.randn(past_v.shape, generator=g),
        "cache_bias": torch.zeros_like(bias),
        "rope_cos": cos,
        "rope_sin": sin,
    }


def cmd_onnx_vs_eager(args: argparse.Namespace) -> int:
    import onnxruntime as ort  # noqa: PLC0415

    step = _trained_step(args.artifact, args.ckpt)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    cache_frames = shapes["inputs_past_k"][3] // step.trunk.tokens_per_frame

    worst = 0.0
    results: dict[str, Any] = {"cache_frames": cache_frames, "trials": {}}
    for trial in range(args.trials):
        feed = _eager_feed(step, cache_frames, seed=1000 + trial)
        with torch.inference_mode():
            eager = step(feed)
        got = sess.run(None, {f"inputs_{k}": v.numpy() for k, v in feed.items()})
        names = [o.name for o in sess.get_outputs()]
        ref = {
            "policy.joint_actions": eager["policy", "joint_actions"],
            "new_k": eager["new_k"],
            "new_v": eager["new_v"],
        }
        per_trial = {}
        for name, value in zip(names, got, strict=True):
            abs_d, rel_d = _scale_rel(torch.from_numpy(value), ref[name])
            per_trial[name] = {"abs": abs_d, "scale_rel": rel_d}
            worst = max(worst, rel_d)
        results["trials"][trial] = per_trial
        logger.info(
            "onnx vs eager",
            trial=trial,
            **{
                k: f"abs {v['abs']:.2e} rel {v['scale_rel']:.2e}"
                for k, v in per_trial.items()
            },
        )

    ok = worst < args.tol
    results["worst_scale_rel"] = worst
    logger.info("ONNX-vs-EAGER GATE", worst_scale_rel=f"{worst:.3e}", passed=ok)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 3. margin screen (trt-export skill §4a), on the decoder step graph
# --------------------------------------------------------------------------- #


def cmd_margins(args: argparse.Namespace) -> int:  # noqa: PLR0914, PLR0915
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    model = onnx.load(str(args.onnx))
    graph = model.graph
    exposed = {o.name for o in graph.output}
    probes: list[tuple[str, str]] = []
    for node in graph.node:
        if node.op_type != "ArgMax":
            continue
        src = node.input[0]
        if src not in exposed:
            graph.output.append(
                helper.make_tensor_value_info(src, TensorProto.FLOAT, None)
            )
            exposed.add(src)
        probes.append((node.name, src))
    if not probes:
        logger.warning("no ArgMax in the graph -- nothing to screen")
        return 0
    logger.info("ArgMax probes", count=len(probes))

    probe_path = Path(args.workdir) / "margin_probe.onnx"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        model,
        str(probe_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=probe_path.name + ".w",
        size_threshold=1024,
    )
    sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
    meta = {i.name: (i.shape, i.type) for i in sess.get_inputs()}
    cache_tokens = meta["inputs_past_k"][0][3]
    num_waypoints = meta["inputs_waypoints"][0][2]
    image_names = _image_input_names(meta)
    frames = _real_frames(args.frames, meta[image_names[0]][0][-1])

    rng = np.random.default_rng(args.seed)
    per_probe: dict[str, list[float]] = {name: [] for name, _ in probes}
    names = [o.name for o in sess.get_outputs()]
    for trial in range(args.trials):
        # every camera draws a real frame from the same rotating pool, offset by
        # its index -- no per-camera footage, but never a repeat within one tick
        camera_frames = {
            name: frames[(trial + i) % len(frames)]
            for i, name in enumerate(image_names)
        }
        feed = _small_inputs(rng, camera_frames, num_waypoints)
        # a WARM cache: unfilled slots would mask the cache away entirely and
        # screen a context the served model never sees
        feed["inputs_past_k"] = rng.standard_normal(
            meta["inputs_past_k"][0], dtype=np.float32
        )
        feed["inputs_past_v"] = rng.standard_normal(
            meta["inputs_past_v"][0], dtype=np.float32
        )
        feed["inputs_cache_bias"] = np.zeros((1, 1, 1, cache_tokens), np.float32)
        cos, sin = frame_rope_cos_sin(
            torch.tensor(int(rng.integers(0, 4096))),
            head_dim=meta["inputs_rope_cos"][0][-1],
            base=1000.0,
        )
        feed["inputs_rope_cos"] = cos.reshape(1, -1).numpy().astype(np.float32)
        feed["inputs_rope_sin"] = sin.reshape(1, -1).numpy().astype(np.float32)

        got = dict(zip(names, sess.run(None, feed), strict=True))
        for probe_name, tensor_name in probes:
            logits = np.asarray(got[tensor_name], dtype=np.float64)
            flat = logits.reshape(-1, logits.shape[-1])
            top2 = np.sort(flat, axis=-1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            # fp16 spacing at the top1 magnitude: below one of these the two
            # codes are not distinguishable in fp16 at all
            ulp = np.spacing(np.abs(top2[:, 1]).astype(np.float16)).astype(np.float64)
            per_probe[probe_name].extend((margin / np.maximum(ulp, 1e-12)).tolist())

    summary: dict[str, Any] = {}
    overall_min = float("inf")
    for probe_name, ulps in per_probe.items():
        arr = np.asarray(ulps)
        summary[probe_name] = {
            "n": int(arr.size),
            "min_ulp": float(arr.min()),
            "p1_ulp": float(np.percentile(arr, 1)),
            "median_ulp": float(np.median(arr)),
            "frac_under_1_ulp": float((arr < 1).mean()),
            "frac_under_4_ulp": float((arr < 4).mean()),  # noqa: PLR2004
        }
        overall_min = min(overall_min, summary[probe_name]["min_ulp"])
        logger.info("margin", probe=probe_name, **summary[probe_name])

    verdict = (
        "fp16 WILL flip decisions -- serve fp32"
        if overall_min < 1
        else "marginal: verify parity at n>=200 before serving low precision"
        if overall_min < 4  # noqa: PLR2004
        else "fp16 has always been clean at this margin"
    )
    logger.info("MARGIN SCREEN", min_ulp=f"{overall_min:.3f}", verdict=verdict)
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"min_ulp": overall_min, "verdict": verdict, "probes": summary},
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


# --------------------------------------------------------------------------- #
# 4. trial set + ORT fp32 reference + raw .dat for trtexec
# --------------------------------------------------------------------------- #


def cmd_trials(args: argparse.Namespace) -> int:  # noqa: PLR0914, PLR0915
    import onnxruntime as ort  # noqa: PLC0415

    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "caches").mkdir(parents=True, exist_ok=True)

    step = _trained_step(args.artifact, args.ckpt)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    meta = {i.name: i.shape for i in sess.get_inputs()}
    k = step.trunk.tokens_per_frame
    cache_frames = meta["inputs_past_k"][3] // k
    num_waypoints = meta["inputs_waypoints"][2]
    image_names = _image_input_names(meta)
    frames = _real_frames(args.frames, meta[image_names[0]][-1])
    rng = np.random.default_rng(args.seed)

    per_history = args.trials // args.histories
    if per_history * args.histories != args.trials:
        msg = (
            f"--trials {args.trials} must be divisible by --histories {args.histories}"
        )
        raise SystemExit(msg)

    manifest: dict[str, Any] = {
        "onnx": str(args.onnx),
        "trials": args.trials,
        "histories": args.histories,
        "cache_frames": cache_frames,
        "code_tol": CODE_TOL,
        "channels": list(CHANNELS),
        "frames": args.frames,
        "seed": args.seed,
        "items": [],
    }
    reference = np.zeros(
        (args.trials, *tuple(sess.get_outputs()[0].shape[1:])), np.float32
    )

    trial = 0
    for history in range(args.histories):
        # --- build this history by genuinely STREAMING the eager model, tick by
        # tick, exactly as drivr would: cold cache, ring slot writes, monotone
        # frame counter. Not randn -- these are real keys of real frames.
        past_k, past_v, bias = step.empty_cache(cache_frames=cache_frames)
        for tick in range(cache_frames):
            camera_frames = {
                name: frames[(history + tick + i) % len(frames)]
                for i, name in enumerate(image_names)
            }
            small = _small_inputs(rng, camera_frames, num_waypoints)
            cos, sin = step.rope(tick)
            feed = {
                **{
                    f"image_{camera}": torch.from_numpy(small[f"inputs_image_{camera}"])
                    for camera in step.policy.cameras
                },
                "speed": torch.from_numpy(small["inputs_speed"]),
                "waypoints": torch.from_numpy(small["inputs_waypoints"]),
                "past_k": past_k,
                "past_v": past_v,
                "cache_bias": bias,
                "rope_cos": cos,
                "rope_sin": sin,
            }
            with torch.inference_mode():
                res = step(feed)
            slot = slice((tick % cache_frames) * k, (tick % cache_frames + 1) * k)
            past_k, past_v, bias = past_k.clone(), past_v.clone(), bias.clone()
            past_k[..., slot, :] = res["new_k"]
            past_v[..., slot, :] = res["new_v"]
            bias[..., slot] = 0.0
        logger.info("history streamed", history=history, ticks=cache_frames)

        past_k_np = past_k.numpy().astype(np.float32)
        past_v_np = past_v.numpy().astype(np.float32)
        bias_np = bias.numpy().astype(np.float32)
        for name, value in (
            ("past_k", past_k_np),
            ("past_v", past_v_np),
            ("cache_bias", bias_np),
        ):
            value.tofile(out / "caches" / f"h{history}_{name}.dat")

        for _ in range(per_history):
            camera_frames = {
                name: frames[(trial + i) % len(frames)]
                for i, name in enumerate(image_names)
            }
            small = _small_inputs(rng, camera_frames, num_waypoints)
            frame_index = cache_frames + int(rng.integers(0, 512))
            cos, sin = step.rope(frame_index)
            feed = {
                **small,
                "inputs_past_k": past_k_np,
                "inputs_past_v": past_v_np,
                "inputs_cache_bias": bias_np,
                "inputs_rope_cos": cos.numpy().astype(np.float32),
                "inputs_rope_sin": sin.numpy().astype(np.float32),
            }
            reference[trial] = sess.run(["policy.joint_actions"], feed)[0][0]
            for name in (
                *image_names,
                "inputs_speed",
                "inputs_waypoints",
                "inputs_rope_cos",
                "inputs_rope_sin",
            ):
                feed[name].tofile(out / "inputs" / f"t{trial:03d}_{name}.dat")
            manifest["items"].append({"trial": trial, "history": history})
            if trial % 20 == 0:
                logger.info("trial", trial=trial, action0=reference[trial][0].tolist())
            trial += 1

    np.save(out / "reference_ort_fp32.npy", reference)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "trial set written",
        path=str(out),
        trials=args.trials,
        histories=args.histories,
        reference_shape=list(reference.shape),
    )
    return 0


# --------------------------------------------------------------------------- #
# 5. score collected engine outputs against the reference
# --------------------------------------------------------------------------- #


def cmd_score(args: argparse.Namespace) -> int:  # noqa: PLR0914
    trialset = Path(args.trialset)
    manifest = json.loads((trialset / "manifest.json").read_text())
    reference = np.load(trialset / "reference_ort_fp32.npy")
    tol = args.code_tol

    got = []
    missing = []
    for trial in range(manifest["trials"]):
        path = Path(args.outputs) / f"t{trial:03d}_joint_actions.dat"
        if not path.exists():
            missing.append(trial)
            continue
        got.append((trial, np.fromfile(path, np.float32).reshape(reference.shape[1:])))
    if missing:
        logger.warning("missing engine outputs", count=len(missing), first=missing[:5])
    if not got:
        msg = f"no engine outputs found in {args.outputs}"
        raise SystemExit(msg)

    changed: list[dict[str, Any]] = []
    per_channel = {c: {"max": 0.0, "sum": 0.0, "n": 0} for c in CHANNELS}
    worst = 0.0
    for trial, value in got:
        diff = np.abs(value - reference[trial])
        worst = max(worst, float(diff.max()))
        for i, channel in enumerate(CHANNELS):
            column = diff[..., i]
            per_channel[channel]["max"] = max(
                per_channel[channel]["max"], float(column.max())
            )
            per_channel[channel]["sum"] += float(column.sum())
            per_channel[channel]["n"] += int(column.size)
        if diff.max() > tol:
            flipped = [
                CHANNELS[i] for i in range(len(CHANNELS)) if diff[..., i].max() > tol
            ]
            changed.append({
                "trial": trial,
                "max_abs": float(diff.max()),
                "channels": flipped,
            })

    control_only = [
        c
        for c in changed
        if set(c["channels"]) <= {"gas_pedal", "brake_pedal", "steering_angle"}
    ]
    summary = {
        "label": args.label,
        "scored": len(got),
        "decision_changes": len(changed),
        "decision_changes_control_only": len(control_only),
        "worst_max_abs": worst,
        "code_tol": tol,
        "per_channel": {
            c: {"max_abs": v["max"], "mean_abs": v["sum"] / max(v["n"], 1)}
            for c, v in per_channel.items()
        },
        "changed": changed,
    }
    logger.info(
        "PARITY",
        label=args.label,
        decisions=f"{len(changed)}/{len(got)}",
        control_only=f"{len(control_only)}/{len(got)}",
        worst_max_abs=f"{worst:.6f}",
    )
    for channel, value in summary["per_channel"].items():
        logger.info(
            "per channel",
            channel=channel,
            max_abs=f"{value['max_abs']:.4e}",
            mean_abs=f"{value['mean_abs']:.4e}",
        )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def weights(p: argparse.ArgumentParser) -> None:
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--artifact", help="e.g. yaak/rmind/model-do8m9ot8:v0")
        g.add_argument("--ckpt", help="local checkpoint path")

    p_gates = sub.add_parser("gates", help="streaming == full windowed forward")
    weights(p_gates)
    p_gates.add_argument("--window", type=int, default=None)
    p_gates.add_argument("--out", default=None)
    p_gates.set_defaults(fn=cmd_gates)

    p_ove = sub.add_parser("onnx-vs-eager", help="ORT CPU fp32 vs PyTorch")
    weights(p_ove)
    p_ove.add_argument("--onnx", type=Path, required=True)
    p_ove.add_argument("--trials", type=int, default=3)
    p_ove.add_argument("--tol", type=float, default=1e-5, help="scale-relative")
    p_ove.add_argument("--out", default=None)
    p_ove.set_defaults(fn=cmd_onnx_vs_eager)

    p_m = sub.add_parser("margins", help="ArgMax top1-top2 in fp16 ULPs")
    p_m.add_argument("--onnx", type=Path, required=True)
    p_m.add_argument("--trials", type=int, default=25)
    p_m.add_argument("--frames", default=DEFAULT_FRAMES)
    p_m.add_argument("--workdir", default="/tmp/decoder_margin")  # noqa: S108
    p_m.add_argument("--seed", type=int, default=1337)
    p_m.add_argument("--out", default=None)
    p_m.set_defaults(fn=cmd_margins)

    p_t = sub.add_parser("trials", help="trial set + ORT fp32 reference + .dat files")
    weights(p_t)
    p_t.add_argument("--onnx", type=Path, required=True)
    p_t.add_argument("--trials", type=int, default=200)
    p_t.add_argument("--histories", type=int, default=10)
    p_t.add_argument("--frames", default=DEFAULT_FRAMES)
    p_t.add_argument("--seed", type=int, default=1337)
    p_t.add_argument("--out", required=True)
    p_t.set_defaults(fn=cmd_trials)

    p_s = sub.add_parser("score", help="score engine outputs vs the reference")
    p_s.add_argument("--trialset", required=True)
    p_s.add_argument("--outputs", required=True)
    p_s.add_argument("--label", default="engine")
    p_s.add_argument("--code-tol", type=float, default=CODE_TOL)
    p_s.add_argument("--out", default=None)
    p_s.set_defaults(fn=cmd_score)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
