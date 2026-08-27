#!/bin/bash
# Build TRT engines for every *.onnx in a directory, at one or more precisions,
# and benchmark each with trtexec.
#
# Waits for the host to be genuinely idle first: TRT picks kernels by TIMING them,
# so building under load bakes in slower tactics. If the load never settles this
# exits WITHOUT building rather than producing a mistimed engine.
#
# Usage:
#   build_and_bench.sh ONNX_DIR ["fp32 fp16-strict"] [DRIVR_DIR]
# Example:
#   build_and_bench.sh ~/onnx_exports/patch_policy "fp32 fp16-strict fp16"
set -u
PP="${1:?usage: build_and_bench.sh ONNX_DIR [PRECISIONS] [DRIVR_DIR]}"
PRECS="${2:-fp32 fp16-strict}"
D="${3:-/home/max/Code/drivr}"
PY="$D/.venv/bin/python"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

suffix_for () {  # precision -> engine filename suffix
    case "$1" in
        fp32)        echo ".trt" ;;
        fp16)        echo ".fp16.trt" ;;
        fp16-strict) echo ".fp16strict.trt" ;;
        *)           echo ".$1.trt" ;;
    esac
}

echo "=== waiting for sustained low load (3 x <3.0, 45s apart, up to 2h) ==="
settled=0
for i in $(seq 1 160); do
    L=$(awk '{print $1}' /proc/loadavg)
    if [ "$(echo "$L < 3.0" | bc -l)" = "1" ]; then
        settled=$((settled + 1)); echo "  load $L — quiet ($settled/3)"
        [ $settled -ge 3 ] && break
    else
        [ $settled -gt 0 ] && echo "  load $L — busy again, resetting"
        settled=0; [ $((i % 8)) -eq 0 ] && echo "  load $L, still waiting ..."
    fi
    sleep 45
done
[ $settled -ge 3 ] || { echo "GAVE UP: load never settled — not building"; exit 3; }
echo "proceeding — load settled"; free -g | head -2

for f in "$PP"/*.onnx; do
    [ -e "$f" ] || { echo "no *.onnx in $PP"; exit 2; }
    name=$(basename "$f" .onnx)
    for prec in $PRECS; do
        eng="$PP/${name}$(suffix_for "$prec")"
        echo ""; echo "=== $name [$prec]"
        if [ -s "$eng" ]; then
            echo "BUILT (cached): $(basename "$eng")"
        else
            start=$(date +%s)
            "$PY" "$D/scripts/build_trt_engine.py" --onnx "$f" --precision "$prec" \
                --workspace-gb 6 > "/tmp/build_${name}_${prec}.log" 2>&1 || {
                    echo "BUILD FAILED: $name $prec"; tail -8 "/tmp/build_${name}_${prec}.log"
                    continue; }
            [ -s "$eng" ] || { echo "BUILD ODD: exit 0 but $eng missing"; continue; }
            echo "BUILT: $(basename "$eng") ($(du -h "$eng" | cut -f1), $(( $(date +%s) - start ))s)"
        fi
        "$TRTEXEC" --loadEngine="$eng" --iterations=60 --avgRuns=20 --useSpinWait \
            --warmUp=1000 > "/tmp/bench_$(basename "$eng").log" 2>&1
        gpu=$(grep -m1 "GPU Compute Time" "/tmp/bench_$(basename "$eng").log" \
              | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
        e2e=$(grep -m1 "\] \[I\] Latency:" "/tmp/bench_$(basename "$eng").log" \
              | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
        echo "LATENCY $(basename "$eng"): compute_median=${gpu:-?}ms end_to_end=${e2e:-?}ms"
    done
done

echo ""; echo "=== engines ==="; ls -la "$PP"/*.trt 2>/dev/null
echo "ALL DONE — now verify parity (scripts/verify_trt_parity.py), including an fp32 control"
