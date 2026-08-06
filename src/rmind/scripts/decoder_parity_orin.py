#!/usr/bin/env python
# This file is deliberately exempt from the repo's lint profile, and `T201` is the
# reason that matters: `ruff` runs here with `fix = true, unsafe-fixes = true`, and
# left to itself it DELETES every `print` in this script -- which is the script's
# entire product. Its output IS the measurement. The same trap cost a benchmark run
# once already (docs/decoder_only_kv_cache.md §11.6.2, where PLR6104 rewrote a
# `mask_mod` into an in-place op and killed the FlexAttention path).
#
# The rest is the ordinary shape of a standalone operational script rather than
# library code: one long `main` that reads as a procedure, lazy `tensorrt`/`torch`
# imports so the ORT-only paths work on a host with no CUDA, and messages passed
# straight to `SystemExit`.
# ruff: noqa: T201, TRY003, EM101, EM102, ANN001, ANN201, ANN204, PLC0415, C901,
# ruff: noqa: PLR0912, PLR0914, PLR0915, RUF015, RUF069, B905, C416, EXE001,
# ruff: noqa: FURB101, PTH123, PLR6201
"""Decision-parity ladder for the KV-cached decoder step, run entirely on the Orin.

Self-contained on purpose: the KV cache is 126 MiB per trial at window 16, so
shipping 200 pre-built caches from the NAS would be 25 GiB over a ZeroTier relay.
Instead the histories are streamed HERE, using ONNX Runtime on the very graph the
engines were built from -- which is exactly the streaming the runtime does, and
ORT-vs-eager is separately gated to ~1e-5 on sisyphos.

    decoder_parity_orin.py --onnx M.onnx --engines a.trt,b.trt --labels a,b \
        --trials 200 --histories 10 --frames f1.jpg,f2.jpg,f3.jpg

What is scored is DECISION CHANGES -- trials where any action channel moves more
than --code-tol -- not float error. The graph ends in ArgMax over code logits, so
a tiny numeric error flips a discrete code and the action moves by a step, not a
nudge. Float noise is ~1e-4 and a code flip is ~0.1, so the threshold is not
delicate.

Rules this obeys, each learned by someone getting it wrong (trt-export skill §5):
  * real camera frames, several of them -- synthetic noise both understated a
    failure (4/10 vs 8/10) and manufactured a false one;
  * >=50 trials to detect, >=200 to decide (2/50 and 4/50 are indistinguishable);
  * ALWAYS an fp32 control, so precision loss can be told apart from ArgMax
    boundary ties -- an input sitting ON a boundary flips under any perturbation;
  * per-channel magnitudes, because turn_signal's dynamic range dominates any
    single headline number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("gas_pedal", "brake_pedal", "steering_angle", "turn_signal")
CONTROL = ("gas_pedal", "brake_pedal", "steering_angle")


def rope_cos_sin(frame_index: int, head_dim: int, base: float = 1000.0):
    """Frame-RoPE `(1, head_dim)` cos/sin, float64 internally then cast to fp32.

    Mirrors rmind.components.transformer.causal_frame.frame_rope_cos_sin: the
    half-split (GPT-NeoX/Llama) pairing, computed in float64 so a long-episode
    absolute frame counter stays exact. This is the serving contract -- drivr
    computes these host-side, which is why the graph has zero Sin/Cos nodes.
    """
    inv_freq = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angle = float(frame_index) * inv_freq
    cos = np.concatenate([np.cos(angle), np.cos(angle)])
    sin = np.concatenate([np.sin(angle), np.sin(angle)])
    return (
        cos.reshape(1, -1).astype(np.float32),
        sin.reshape(1, -1).astype(np.float32),
    )


def real_frames(spec: str, size: int) -> list[np.ndarray]:
    import cv2

    out = []
    for name in (s.strip() for s in spec.split(",") if s.strip()):
        img = cv2.imread(name)
        if img is None:
            print(f"  !! unreadable, skipping: {name}")
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        out.append((img[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32))
    if not out:
        raise SystemExit(f"no readable frames in {spec!r} -- real frames are required")
    print(f"  real frames: {len(out)}")
    return out


def small_inputs(rng, frame, num_waypoints):
    jitter = np.float32(rng.uniform(0.92, 1.08))
    return {
        "inputs_image": np.clip(frame * jitter, 0, 1)[None, None].astype(np.float32),
        "inputs_speed": np.full((1, 1, 1), rng.uniform(0, 40), np.float32),
        "inputs_waypoints": np.stack(
            [
                rng.normal(0, 0.02, num_waypoints).astype(np.float32),
                (np.arange(num_waypoints, dtype=np.float32) + 1)
                * np.float32(rng.uniform(2, 25) / 100.0),
            ],
            axis=1,
        )[None, None].astype(np.float32),
    }


class TRTRunner:
    """Static-shape engine runner. Validates EVERY binding against the engine.

    `set_tensor_address` takes a raw pointer with no size validation, so a cache
    allocated for a different cache_frames / layer count / head count / dtype is
    not an error -- TRT reinterprets the buffer and the model merely looks weak
    (docs/decoder_only_kv_cache.md §3). Hence the explicit check.
    """

    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"failed to deserialize {path} (TRT/arch mismatch?)")
        self.ctx = self.engine.create_execution_context()
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.io.append((
                name,
                self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                tuple(self.engine.get_tensor_shape(name)),
                trt.nptype(self.engine.get_tensor_dtype(name)),
            ))

    def describe(self):
        return [
            {
                "name": n,
                "input": bool(i),
                "shape": list(s),
                "dtype": np.dtype(d).name,
                "bytes": int(np.prod(s)) * np.dtype(d).itemsize,
            }
            for n, i, s, d in self.io
        ]

    def __call__(self, feed):
        torch = self.torch
        # `keep` holds a reference to every device buffer until after
        # synchronize(): set_tensor_address stores a raw pointer, so letting an
        # input tensor be garbage-collected before execution frees memory the
        # engine is about to read, with no error and arbitrary results.
        keep, out = [], {}
        for name, is_in, shape, npdt in self.io:
            if is_in:
                if name not in feed:
                    raise SystemExit(f"no feed for engine input {name}")
                src = feed[name]
                if tuple(src.shape) != shape:
                    raise SystemExit(
                        f"{name}: engine wants {shape}, got {tuple(src.shape)} "
                        "-- set_tensor_address would NOT catch this"
                    )
                t = torch.from_numpy(np.ascontiguousarray(src.astype(npdt))).cuda()
            else:
                t = torch.empty(shape, dtype=getattr(torch, np.dtype(npdt).name)).cuda()
                out[name] = t
            keep.append(t)
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        result = {k: v.cpu().numpy() for k, v in out.items()}
        del keep
        return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engines", required=True, help="comma-separated .trt paths")
    ap.add_argument("--labels", required=True, help="comma-separated labels")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--histories", type=int, default=10)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--code-tol", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import onnxruntime as ort

    engines = [Path(p) for p in a.engines.split(",")]
    labels = a.labels.split(",")
    if len(engines) != len(labels):
        raise SystemExit("--engines and --labels must have the same length")

    print("== ORT fp32 reference session (CPU) ==")
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    meta = {i.name: i.shape for i in sess.get_inputs()}
    tokens_per_frame = [o.shape for o in sess.get_outputs() if o.name == "new_k"][0][3]
    cache_tokens = meta["inputs_past_k"][3]
    cache_frames = cache_tokens // tokens_per_frame
    head_dim = meta["inputs_rope_cos"][-1]
    num_waypoints = meta["inputs_waypoints"][2]
    image_hw = meta["inputs_image"][-1]
    print(
        f"  cache_frames={cache_frames} tokens_per_frame={tokens_per_frame} "
        f"cache_tokens={cache_tokens} head_dim={head_dim} image={image_hw}"
    )

    runners = {}
    for label, path in zip(labels, engines):
        print(f"== loading engine {label}: {path.name} ==")
        runners[label] = TRTRunner(path)

    # KV-cache memory, read off the ENGINE rather than re-derived from a formula
    cache_report = {}
    for label, runner in runners.items():
        rows = [b for b in runner.describe() if b["name"] in
                ("inputs_past_k", "inputs_past_v", "inputs_cache_bias",
                 "new_k", "new_v")]
        total = sum(b["bytes"] for b in rows if b["name"].startswith("inputs_past"))
        cache_report[label] = {
            "bindings": rows,
            "kv_cache_bytes": total,
            "kv_cache_mib": total / 1024 / 1024,
        }
        print(f"  {label}: KV cache = {total / 1024 / 1024:.2f} MiB "
              f"({[b['dtype'] for b in rows if b['name'] == 'inputs_past_k'][0]})")

    frames = real_frames(a.frames, image_hw)
    rng = np.random.default_rng(a.seed)
    per_history = a.trials // a.histories
    if per_history * a.histories != a.trials:
        raise SystemExit("--trials must be divisible by --histories")

    reference = np.zeros((a.trials, *tuple(sess.get_outputs()[0].shape[1:])), np.float32)
    got = {label: np.zeros_like(reference) for label in labels}

    trial = 0
    for history in range(a.histories):
        # --- stream this history tick by tick through ORT: cold cache, ring slot
        # writes, monotone frame counter. Real keys of real frames, not randn.
        past_k = np.zeros(meta["inputs_past_k"], np.float32)
        past_v = np.zeros(meta["inputs_past_v"], np.float32)
        bias = np.full((1, 1, 1, cache_tokens), -1e4, np.float32)
        for tick in range(cache_frames):
            cos, sin = rope_cos_sin(tick, head_dim)
            feed = small_inputs(
                rng, frames[(history + tick) % len(frames)], num_waypoints
            )
            feed |= {
                "inputs_past_k": past_k,
                "inputs_past_v": past_v,
                "inputs_cache_bias": bias,
                "inputs_rope_cos": cos,
                "inputs_rope_sin": sin,
            }
            _, new_k, new_v = sess.run(
                ["policy.joint_actions", "new_k", "new_v"], feed
            )
            lo = (tick % cache_frames) * tokens_per_frame
            hi = lo + tokens_per_frame
            past_k[:, :, :, lo:hi, :] = new_k
            past_v[:, :, :, lo:hi, :] = new_v
            bias[:, :, :, lo:hi] = 0.0
        print(f"  history {history}: streamed {cache_frames} ticks")

        for _ in range(per_history):
            cos, sin = rope_cos_sin(
                cache_frames + int(rng.integers(0, 512)), head_dim
            )
            feed = small_inputs(rng, frames[trial % len(frames)], num_waypoints)
            feed |= {
                "inputs_past_k": past_k,
                "inputs_past_v": past_v,
                "inputs_cache_bias": bias,
                "inputs_rope_cos": cos,
                "inputs_rope_sin": sin,
            }
            reference[trial] = sess.run(["policy.joint_actions"], feed)[0][0]
            for label, runner in runners.items():
                got[label][trial] = runner({k: v for k, v in feed.items()})[
                    "policy.joint_actions"
                ][0]
            if trial % 10 == 0:
                print(f"    trial {trial}/{a.trials}", flush=True)
            trial += 1

    summary = {
        "onnx": str(a.onnx),
        "trials": a.trials,
        "histories": a.histories,
        "code_tol": a.code_tol,
        "frames": a.frames,
        "seed": a.seed,
        "cache": cache_report,
        "engines": {},
    }
    for label in labels:
        diff = np.abs(got[label] - reference)
        flat = diff.reshape(a.trials, -1, len(CHANNELS))
        changed, control_only = [], 0
        for t in range(a.trials):
            if flat[t].max() > a.code_tol:
                chans = [
                    CHANNELS[c]
                    for c in range(len(CHANNELS))
                    if flat[t, :, c].max() > a.code_tol
                ]
                changed.append({
                    "trial": t,
                    "max_abs": float(flat[t].max()),
                    "channels": chans,
                })
                if set(chans) <= set(CONTROL):
                    control_only += 1
        summary["engines"][label] = {
            "decision_changes": len(changed),
            "decision_changes_control_only": control_only,
            "worst_max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "bit_exact": bool(diff.max() == 0.0),
            "per_channel": {
                CHANNELS[c]: {
                    "max_abs": float(flat[:, :, c].max()),
                    "mean_abs": float(flat[:, :, c].mean()),
                }
                for c in range(len(CHANNELS))
            },
            "changed": changed,
        }
        e = summary["engines"][label]
        print(
            f"\n== {label}: decisions {len(changed)}/{a.trials} "
            f"(control-only {control_only}) worst |d| {diff.max():.6f} "
            f"mean {diff.mean():.3e} bit_exact={e['bit_exact']}"
        )
        for c, v in e["per_channel"].items():
            print(f"     {c:16s} max {v['max_abs']:.4e}  mean {v['mean_abs']:.4e}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
