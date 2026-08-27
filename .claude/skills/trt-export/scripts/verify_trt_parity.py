#!/usr/bin/env python
"""Verify a TRT engine against its ONNX source: is the engine still the model?

Compares `policy.joint_actions` from TensorRT (GPU, built precision) against
ONNX Runtime (CPU, fp32 reference) over N randomized-but-realistic inputs.

For codebook policies the float error is the wrong headline: the graph ends in
ArgMax over code logits, so a small numeric error flips a discrete code and the
action changes by a step, not a nudge. This reports both:

  * float agreement  — max/mean |TRT - ORT| per channel
  * DECISION agreement — fraction of samples where any channel moves more than
    `--code-tol`, i.e. large enough that a code almost certainly flipped

A pure-fp16 build of a VQ policy typically shows tiny mean error and a few
percent decision disagreement, which is the failure that matters.

Usage:
  verify_trt_parity.py --onnx model.onnx --engine model.fp16strict.trt [--trials 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("gas", "brake", "steering", "turn_signal")


def build_inputs(sess, rng, frame01=None):
    """Realistic input set: [0,1] image, km/h speed, /100 ego waypoints."""
    feed = {}
    for i in sess.get_inputs():
        shape = [1 if not isinstance(d, int) else d for d in i.shape]
        name = i.name.lower()
        if "cam" in name:
            h, w = shape[-2], shape[-1]
            if frame01 is not None:
                import cv2
                img = cv2.resize(frame01, (w, h), interpolation=cv2.INTER_AREA)
                chw = img.transpose(2, 0, 1)
                # per-frame jitter so the 6 timesteps aren't identical
                seq = np.stack([np.clip(chw * (0.94 + 0.02 * k), 0, 1)
                                for k in range(shape[1])], axis=0)
                feed[i.name] = seq[None].astype(np.float32)
            else:
                feed[i.name] = rng.uniform(0, 1, shape).astype(np.float32)
        elif "speed" in name:
            feed[i.name] = np.full(shape, rng.uniform(0, 40), np.float32)
        elif "waypoint" in name:
            spacing = rng.uniform(2, 25) / 100.0
            y = (np.arange(shape[2], dtype=np.float32) + 1) * spacing
            x = rng.normal(0, 0.02, shape[2]).astype(np.float32)
            wp = np.stack([x, y], axis=1)
            feed[i.name] = np.tile(wp, (shape[0], shape[1], 1, 1)).astype(np.float32)
        else:
            feed[i.name] = rng.uniform(0, 1, shape).astype(np.float32)
    return feed


class TRTRunner:
    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"failed to deserialize {path} "
                             "(TRT version / GPU arch mismatch?)")
        self.ctx = self.engine.create_execution_context()
        self.io = []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.io.append((n, self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT,
                            tuple(self.engine.get_tensor_shape(n)),
                            trt.nptype(self.engine.get_tensor_dtype(n))))

    def __call__(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        bufs, out = {}, {}
        for name, is_in, shape, npdt in self.io:
            if is_in:
                src = None
                for k, v in feed.items():
                    if k == name or k.lower().replace("batch_", "") in name.lower() \
                            or name.lower() in k.lower():
                        src = v
                        break
                if src is None:
                    raise SystemExit(f"no feed for engine input {name}")
                t = torch.from_numpy(np.ascontiguousarray(src.astype(npdt))).cuda()
            else:
                t = torch.empty(tuple(shape),
                                dtype=getattr(torch, np.dtype(npdt).name)).cuda()
                out[name] = t
            bufs[name] = t
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {k: v.cpu().numpy() for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--frames", default="/tmp/native_raw.png",
                    help="comma-separated real camera frames; trials cycle through "
                         "them so activations vary as they do while driving")
    ap.add_argument("--code-tol", type=float, default=0.02,
                    help="|diff| above this counts as a changed decision, not float noise")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    trt_run = TRTRunner(args.engine)
    out_name = sess.get_outputs()[0].name

    import cv2
    frames = []
    for fp in [Path(x) for x in args.frames.split(",") if x.strip()]:
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, _ = img.shape[:2]
        side = min(h, 320)
        top = max((h - side) // 2, 0)
        frames.append(cv2.cvtColor(img[top:top + side, :], cv2.COLOR_BGR2RGB)
                      .astype(np.float32) / 255.0)
    print(f"images from {len(frames)} real frame(s)" if frames else "images from noise")

    rng = np.random.default_rng(1234)
    diffs, decision_changes, rows = [], 0, []
    for t in range(args.trials):
        feed = build_inputs(sess, rng, frames[t % len(frames)] if frames else None)
        ref = sess.run([out_name], feed)[0]
        got = trt_run(feed)
        eng = next(iter(got.values())) if out_name not in got else got[out_name]
        d = np.abs(eng.astype(np.float64) - ref.astype(np.float64))
        diffs.append(d)
        changed = bool((d > args.code_tol).any())
        decision_changes += int(changed)
        rows.append(dict(trial=t, max_abs=float(d.max()),
                         step0=[round(float(x), 4) for x in ref[0, 0]],
                         step0_trt=[round(float(x), 4) for x in eng[0, 0]],
                         decision_changed=changed))
        if changed:
            print(f"  trial {t}: DECISION CHANGED  max|d|={d.max():.4f}  "
                  f"ref step0={np.round(ref[0,0],3)}  trt step0={np.round(eng[0,0],3)}")

    D = np.stack(diffs)
    print(f"\n=== {args.engine.name} vs {args.onnx.name} ({args.trials} trials) ===")
    print(f"  max |diff| overall : {D.max():.6f}")
    print(f"  mean |diff|        : {D.mean():.6f}")
    for c, ch in enumerate(CHANNELS):
        print(f"    {ch:12s} max {D[:, :, :, c].max():.6f}  mean {D[:, :, :, c].mean():.6f}")
    frac = decision_changes / args.trials
    print(f"  decision changes   : {decision_changes}/{args.trials} ({100*frac:.1f} %) "
          f"at tol {args.code_tol}")
    verdict = ("PASS — engine matches the ONNX reference"
               if decision_changes == 0 and D.max() < 0.05 else
               "FAIL — engine disagrees with its own ONNX; do not serve this build")
    print(f"  VERDICT: {verdict}")
    if args.out:
        args.out.write_text(json.dumps(
            dict(engine=str(args.engine), onnx=str(args.onnx), trials=args.trials,
                 max_abs=float(D.max()), mean_abs=float(D.mean()),
                 decision_changes=decision_changes, rows=rows), indent=2))
    return 0 if decision_changes == 0 and D.max() < 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
