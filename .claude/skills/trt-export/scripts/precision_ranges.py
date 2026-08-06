#!/usr/bin/env python
"""Derive the FP32 layer-index ranges for the serving build, per model.

The serving standard is a PAIR of engines:
  1. fp32                                  -- the reference, bit-exact, always build it
  2. fp32 encoder + fp16 trunk + fp32 decode -- ~4.3x faster at the same measured parity

Range (2) needs three boundaries, and they are ARCHITECTURE-SPECIFIC -- do not reuse numbers
from another model. Measured on dinov2_dinowm_big they were 0-1945 / 1946-2850 / 2851-2961,
but a dinov3 arm at 256x256 has a different layer count entirely.

Why these boundaries: relative error is created in the image encoder (9e-4 -> 3.2e-3) and is
flat through the temporal trunk (2.46e-3 -> 2.63e-3). Pinning the tail does nothing -- engines
with TopK/ArgMax/decode pinned came out bit-identical to unpinned ones. The encoder is only
~18% of runtime, so protecting it is cheap. The decode head is pinned because the code_head
ArgMax margins live there.

Usage:
    precision_ranges.py MODEL.onnx [--drivr-dir /home/max/Code/drivr]
Prints the ranges and a ready-to-run build command.
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path


def build_network(onnx_path: Path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  onnx parse error: {parser.get_error(i)}", file=sys.stderr)
            raise SystemExit("failed to parse ONNX")
    return network


# The temporal trunk is called `encoder` in rmind's PatchPolicy (confusingly -- the IMAGE
# encoder is `image_encoder`), so its position embedding / first block marks where the
# per-frame image encoder ends.
TRUNK_START = re.compile(r"(encoder\.position_embedding|encoder\.layers\.0\.|speed_embedding)")
DECODE_START = re.compile(r"(code_head\.|offset_head\.)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--drivr-dir", default="/home/max/Code/drivr")
    a = ap.parse_args()

    net = build_network(a.onnx)
    n = net.num_layers
    names = [net.get_layer(i).name for i in range(n)]

    trunk = next((i for i, nm in enumerate(names) if TRUNK_START.search(nm)), None)
    decode = next((i for i, nm in enumerate(names) if DECODE_START.search(nm)), None)

    print(f"layers            : {n}")
    if trunk is None or decode is None:
        print("  !! could not locate boundaries by name.")
        print("     Inspect with: build_mixed.py --onnx M.onnx --list-layers")
        print("     Look for the first `encoder.position_embedding`/`encoder.layers.0.*`")
        print("     constant (trunk start) and the first `code_head.*` constant (decode start).")
        return 2
    if not (0 < trunk < decode < n):
        print(f"  !! implausible ordering: trunk={trunk} decode={decode} n={n}")
        return 2

    enc_hi, dec_lo, dec_hi = trunk - 1, decode, n - 1
    enc_pct = 100.0 * trunk / n
    print(f"image encoder     : 0-{enc_hi}          ({enc_pct:.0f}% of layers)")
    print(f"temporal trunk    : {trunk}-{dec_lo - 1}")
    print(f"decode head       : {dec_lo}-{dec_hi}")
    print(f"\nfp32 ranges       : 0-{enc_hi},{dec_lo}-{dec_hi}")
    print(f"\n# 1/2  reference engine (always build this)")
    print(f"{a.drivr_dir}/.venv/bin/python build_mixed.py \\")
    print(f"    --onnx {a.onnx} --precision fp32 --workspace-gb 6 \\")
    print(f"    --engine {a.onnx.with_suffix('')}.trt")
    print(f"\n# 2/2  serving engine: fp32 encoder + fp16 trunk + fp32 decode")
    print(f"{a.drivr_dir}/.venv/bin/python build_mixed.py \\")
    print(f"    --onnx {a.onnx} --precision mixed --workspace-gb 6 \\")
    print(f"    --fp32-index-ranges 0-{enc_hi},{dec_lo}-{dec_hi} \\")
    print(f"    --engine {a.onnx.with_suffix('')}.encfp32-fp16trunk.trt")
    print("\n# then: parity_matrix.py --trials 200 across BOTH, fp32 first as the control")
    return 0


if __name__ == "__main__":
    sys.exit(main())
