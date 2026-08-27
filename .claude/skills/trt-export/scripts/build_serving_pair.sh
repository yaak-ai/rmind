#!/bin/bash
# Build the SERVING PAIR for one ONNX, and verify it at n=200.
#
#   1. fp32                                     -- the reference. Bit-exact. Always build it.
#   2. fp32 encoder + fp16 trunk + fp32 decode  -- ~4.3x faster at the same measured parity.
#
# These two are the standard. Do NOT build fp16-strict: it is dominated (232 ms at 1/200 vs
# 98 ms at 1/200 for the pair's second engine) because it pins the GEMMs to FP32 and leaves
# data movement in fp16 -- the inverse of how these models train. Do not build int8 either:
# no speed gain over fp16 (both bandwidth-bound) and worse parity. Do not build bf16: 12-19x
# more flips than fp16 at identical latency, despite being the training precision.
#
# Layer ranges are derived per model by precision_ranges.py -- never hardcode them, they are
# architecture-specific (a dinov2 224x224 arm and a dinov3 256x256 arm differ entirely).
#
# Usage: build_serving_pair.sh MODEL.onnx [DRIVR_DIR] [TRIALS]
set -u
ONNX="${1:?usage: build_serving_pair.sh MODEL.onnx [DRIVR_DIR] [TRIALS]}"
D="${2:-/home/max/Code/drivr}"
TRIALS="${3:-200}"
PY="$D/.venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
FRAMES="${FRAMES:-/tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg}"

base="${ONNX%.onnx}"
FP32="$base.trt"
MIX="$base.encfp32-fp16trunk.trt"

echo "######## 0. graph inspection"
"$PY" "$HERE/inspect_onnx.py" "$ONNX" 2>&1 | grep -vE "Warning|pkg_resources"

echo ""
echo "######## 1. margin pre-flight (40 s, no GPU) — min ULP < 1 predicts fp16 flips"
"$PY" "$HERE/margin_screen.py" --onnx "$ONNX" --trials 25 --frames "$FRAMES" 2>&1 \
    | grep -vE "Warning|pkg_resources|^  [0-9]+/" | sed 's/^/  /'

echo ""
echo "######## 2. derive the fp32 layer ranges for this architecture"
RANGES=$("$PY" "$HERE/precision_ranges.py" "$ONNX" --drivr-dir "$D" 2>&1 | tee /dev/stderr \
         | awk '/^fp32 ranges/{print $NF}')
[ -n "$RANGES" ] || { echo "FATAL: could not derive ranges — inspect with --list-layers"; exit 1; }
echo "  using: $RANGES"

echo ""
echo "######## 3. build both engines"
for spec in "fp32|$FP32|" "mixed|$MIX|$RANGES"; do
    IFS='|' read -r prec eng rng <<<"$spec"
    if [ -s "$eng" ]; then echo "  BUILT (cached): $(basename "$eng")"; continue; fi
    start=$(date +%s)
    if [ -n "$rng" ]; then
        "$PY" "$HERE/build_mixed.py" --onnx "$ONNX" --precision "$prec" \
            --fp32-index-ranges "$rng" --engine "$eng" --workspace-gb 6 \
            > "/tmp/bsp_$(basename "$eng").log" 2>&1
    else
        "$PY" "$HERE/build_mixed.py" --onnx "$ONNX" --precision "$prec" \
            --engine "$eng" --workspace-gb 6 > "/tmp/bsp_$(basename "$eng").log" 2>&1
    fi
    [ -s "$eng" ] || { echo "  BUILD FAILED: $prec"; tail -8 "/tmp/bsp_$(basename "$eng").log"; exit 1; }
    echo "  BUILT: $(basename "$eng") ($(du -h "$eng"|cut -f1), $(( $(date +%s) - start ))s)"
done

echo ""
echo "######## 4. latency"
for eng in "$FP32" "$MIX"; do
    "$TRTEXEC" --loadEngine="$eng" --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000 \
        > "/tmp/bspb_$(basename "$eng").log" 2>&1
    ms=$(grep -m1 "GPU Compute Time" "/tmp/bspb_$(basename "$eng").log" \
         | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
    echo "  $(basename "$eng"): ${ms:-?} ms"
done

echo ""
echo "######## 5. parity, $TRIALS trials, fp32 first as the control"
"$PY" "$HERE/parity_matrix.py" --onnx "$ONNX" --engines "$FP32,$MIX" --trials "$TRIALS" \
    --frames "$FRAMES" --out "$base.parity.json" 2>&1 \
    | grep -vE "Warning|pkg_resources|onnxruntime|^\[|ORT [0-9]|\.\.\. [0-9]" | sed 's/^/  /'

echo ""
echo "fp32 MUST be 0/$TRIALS. If it is not, the harness or the reference is broken — stop."
echo "The mixed engine is expected ~1/200 on this input distribution (which is ~7x harsher"
echo "than real driving states: synthetic speed/waypoints tighten margins). Quarantine by"
echo "renaming to QUARANTINE-FAILED-PARITY.* if it is materially worse."
echo "SERVING PAIR DONE"
