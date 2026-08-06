#!/usr/bin/env python
"""TRT builder with per-layer-name precision pinning, int8 PTQ, and a dry-run
layer dump.  Superset of scripts/build_trt_engine.py.

Modes:
  --list-layers                  parse ONNX -> print every network layer (no build)
  --precision fp32|fp16|fp16-strict|mixed|int8|int8-mixed
  --fp32-regex RE                for 'mixed'/'int8-mixed': layers whose name matches
                                 are pinned to FP32 (with OBEY_PRECISION_CONSTRAINTS)
  --calib-frames ...             int8 calibration frames (real camera images)
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_FP16_ONLY_LAYERS = frozenset({
    trt.LayerType.CONSTANT, trt.LayerType.SHUFFLE, trt.LayerType.CAST,
    trt.LayerType.SLICE, trt.LayerType.GATHER, trt.LayerType.CONCATENATION,
    trt.LayerType.TOPK, trt.LayerType.SELECT,
})



_FLOAT_DT = None


def _float_dts():
    global _FLOAT_DT
    if _FLOAT_DT is None:
        dts = [trt.DataType.FLOAT, trt.DataType.HALF]
        for extra in ("BF16",):
            if hasattr(trt.DataType, extra):
                dts.append(getattr(trt.DataType, extra))
        _FLOAT_DT = tuple(dts)
    return _FLOAT_DT


def pin_fp32(L) -> bool:
    """Force a layer to FP32 compute, skipping layers that produce no float tensor.

    Integer producers (index Constants, Cast-to-int, TopK's indices output, Gather
    on indices) cannot be given precision Float -- TRT raises API Usage Error 3.
    """
    outs = []
    for j in range(L.num_outputs):
        try:
            outs.append((j, L.get_output(j)))
        except Exception:
            pass
    if not outs or not any(o.dtype in _float_dts() for _, o in outs):
        return False
    try:
        L.precision = trt.float32
    except Exception:
        return False
    for j, o in outs:
        if o.dtype in _float_dts():
            try:
                L.set_output_type(j, trt.float32)
            except Exception:
                pass
    return True


def make_network(onnx_path: Path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
        raise SystemExit("failed to parse ONNX")
    return builder, network


def build_feed_shapes(network):
    out = {}
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        out[inp.name] = (tuple(int(d) for d in inp.shape), trt.nptype(inp.dtype))
    return out


class FrameCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds realistic batches built from real camera frames."""

    def __init__(self, shapes, frames, n_batches, cache):
        super().__init__()
        import torch
        self.torch = torch
        self.shapes, self.frames, self.n = shapes, frames, n_batches
        self.cache = Path(cache)
        self.idx = 0
        self.rng = np.random.default_rng(7)
        self.keep = {}

    def get_batch_size(self):
        return 1

    def _make(self):
        import cv2
        feed = {}
        for nm, (sh, dt) in self.shapes.items():
            low = nm.lower()
            if "cam" in low:
                h, w = sh[-2], sh[-1]
                img = self.frames[self.idx % len(self.frames)]
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                chw = img.transpose(2, 0, 1)
                seq = np.stack([np.clip(chw * (0.90 + 0.04 * k), 0, 1) for k in range(sh[1])], 0)
                feed[nm] = seq[None].astype(np.float32)
            elif "speed" in low:
                feed[nm] = np.full(sh, self.rng.uniform(0, 40), np.float32)
            elif "waypoint" in low:
                spacing = self.rng.uniform(2, 25) / 100.0
                y = (np.arange(sh[2], dtype=np.float32) + 1) * spacing
                x = self.rng.normal(0, 0.02, sh[2]).astype(np.float32)
                wp = np.stack([x, y], 1)
                feed[nm] = np.tile(wp, (sh[0], sh[1], 1, 1)).astype(np.float32)
            else:
                feed[nm] = self.rng.uniform(0, 1, sh).astype(np.float32)
        return feed

    def get_batch(self, names):
        if self.idx >= self.n:
            return None
        feed = self._make()
        ptrs = []
        for nm in names:
            t = self.torch.from_numpy(np.ascontiguousarray(feed[nm].ravel())).cuda()
            self.keep[nm] = t
            ptrs.append(int(t.data_ptr()))
        self.idx += 1
        print(f"  calibration batch {self.idx}/{self.n}", flush=True)
        return ptrs

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache):
        self.cache.write_bytes(cache)


def load_frames(spec):
    import cv2
    frames = []
    for p in [x for x in spec.split(",") if x.strip()]:
        img = cv2.imread(p)
        if img is None:
            continue
        h = img.shape[0]
        side = min(h, 320)
        top = max((h - side) // 2, 0)
        frames.append(cv2.cvtColor(img[top:top+side, :], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    if not frames:
        raise SystemExit("no calibration frames loaded")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=None)
    ap.add_argument("--precision", default="fp16",
                    choices=["fp32", "fp16", "fp16-strict", "mixed", "int8", "int8-mixed",
                               "bf16", "bf16-mixed", "tf32"])
    ap.add_argument("--fp32-regex", default=None)
    ap.add_argument("--bf16-index-ranges", default=None,
                    help="layer indices to pin to BF16 (per-layer, needs a mixed precision)")
    ap.add_argument("--fp32-index-ranges", default=None,
                    help="e.g. '1737-1875,2853-2961' -- pin these network layer indices to FP32")
    ap.add_argument("--workspace-gb", type=int, default=6)
    ap.add_argument("--list-layers", action="store_true")
    ap.add_argument("--calib-frames",
                    default="/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg")
    ap.add_argument("--calib-batches", type=int, default=24)
    ap.add_argument("--calib-cache", default=None)
    ap.add_argument("--strict-exempt", default=None,
                    help="fp16-strict: comma list of LayerType names left in FP16. "
                         "Default reproduces scripts/build_trt_engine.py: "
                         "CONSTANT,SHUFFLE,CAST,SLICE,GATHER,CONCATENATION,TOPK,SELECT")
    a = ap.parse_args()

    builder, network = make_network(a.onnx)
    print(f"network: {network.num_layers} layers, {network.num_inputs} inputs, "
          f"{network.num_outputs} outputs", flush=True)

    if a.list_layers:
        for i in range(network.num_layers):
            L = network.get_layer(i)
            outs = []
            for j in range(L.num_outputs):
                try:
                    outs.append(f"{tuple(L.get_output(j).shape)}")
                except Exception:
                    pass
            print(f"{i:5d} {str(L.type).replace('LayerType.',''):16s} "
                  f"{L.name[:100]:100s} {';'.join(outs)[:60]}")
        return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_gb * (1 << 30))
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        sh = tuple(int(d) for d in inp.shape)
        profile.set_shape(inp.name, sh, sh, sh)
    config.add_optimization_profile(profile)

    p = a.precision
    if p == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif p == "tf32":
        # set NOTHING: TF32 is on by default, so this is the tensor-core fp32 path --
        # 10-bit mantissa matmul inputs, fp32 accumulate, fp32 storage between ops.
        # 226 ms at 1/200 on dinowm_big => dominated by encfp32-fp16trunk (98 ms, 1/200).
        # Diagnostic only; the REFERENCE must keep TF32 cleared to stay 0/200.
        print("tf32: TRT defaults (TF32 tensor cores on, no other precision flags)")
    elif p == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif p == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
        print("bf16: BF16 flag set, TRT picks per layer")
    elif p == "fp16-strict":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        exempt = _FP16_ONLY_LAYERS
        if a.strict_exempt is not None:
            exempt = frozenset(getattr(trt.LayerType, t.strip().upper())
                               for t in a.strict_exempt.split(",") if t.strip())
        print(f"fp16-strict exempt set: {sorted(str(e).replace('LayerType.','') for e in exempt)}")
        n32 = n16 = 0
        for i in range(network.num_layers):
            L = network.get_layer(i)
            if L.type in exempt:
                n16 += 1
                continue
            if pin_fp32(L):
                n32 += 1
            else:
                n16 += 1
        print(f"fp16-strict: fp32-pinned={n32} fp16-kept={n16}")
    elif p in ("mixed", "int8-mixed", "bf16-mixed"):
        if not (a.fp32_regex or a.fp32_index_ranges):
            raise SystemExit("--fp32-regex or --fp32-index-ranges required for mixed")
        rx = re.compile(a.fp32_regex) if a.fp32_regex else None
        idxset = set()
        for part in (a.fp32_index_ranges or "").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-")
                idxset.update(range(int(lo), int(hi) + 1))
            else:
                idxset.add(int(part))
        if p == "bf16-mixed":
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            if a.bf16_index_ranges:
                config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        hits, skipped = [], []
        for i in range(network.num_layers):
            L = network.get_layer(i)
            if (rx is not None and rx.search(L.name)) or (i in idxset):
                if pin_fp32(L):
                    hits.append(f"{i}:{str(L.type).replace('LayerType.','')}:{L.name[:70]}")
                else:
                    skipped.append(f"{i}:{str(L.type).replace('LayerType.','')}:{L.name[:60]}")
        print(f"mixed: skipped {len(skipped)} non-float (integer/index) layers")
        print(f"mixed: pinned {len(hits)} layers to FP32 "
              f"(regex={a.fp32_regex!r} ranges={a.fp32_index_ranges!r})")
        if a.bf16_index_ranges:
            bset = set()
            for part in a.bf16_index_ranges.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    lo, hi = part.split("-")
                    bset.update(range(int(lo), int(hi) + 1))
                else:
                    bset.add(int(part))
            bset -= idxset          # FP32 pins win
            nb = 0
            for i in sorted(bset):
                L = network.get_layer(i)
                try:
                    if L.get_output(0).dtype in (trt.DataType.INT32, trt.DataType.INT64,
                                                 trt.DataType.BOOL):
                        continue
                    L.precision = trt.DataType.BF16
                    nb += 1
                except Exception:
                    pass
            print(f"mixed: pinned {nb} layers to BF16 (ranges={a.bf16_index_ranges!r})")
            if nb == 0:
                raise SystemExit("bf16 ranges matched no float layers -- refusing to build")
        for h in hits:
            print("   ", h)
        if not hits:
            raise SystemExit("regex matched no layers -- refusing to build a mislabelled engine")
    if p in ("int8", "int8-mixed"):
        config.set_flag(trt.BuilderFlag.INT8)
        cache = a.calib_cache or str(a.onnx.with_suffix(".calib"))
        frames = load_frames(a.calib_frames)
        config.int8_calibrator = FrameCalibrator(build_feed_shapes(network), frames,
                                                 a.calib_batches, cache)
        print(f"int8: entropy-calibrator2, {a.calib_batches} batches from "
              f"{len(frames)} real frames, cache={cache}", flush=True)

    suffix = {"fp32": ".trt", "tf32": ".tf32.trt", "fp16": ".fp16.trt", "fp16-strict": ".fp16strict.trt",
              "bf16": ".bf16.trt", "bf16-mixed": ".bf16mixed.trt",
              "mixed": ".mixed.trt", "int8": ".int8.trt", "int8-mixed": ".int8mixed.trt"}[p]
    engine_path = a.engine or a.onnx.with_name(a.onnx.stem + suffix)
    print(f"building -> {engine_path}", flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("BUILD FAILED")
    engine_path.write_bytes(bytes(ser))
    print(f"built in {time.time()-t0:.0f}s, {engine_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
