#!/usr/bin/env bash
# Build + benchmark + profile one decoder-step engine on delta-dev1 (AGX Orin).
#
#   dev1_measure.sh <onnx> <fp32|fp16|tf32> <outdir>
#
# Flag rationale (/nasa/max/skills/trt-export/SKILL.md):
#  * --noTF32 for fp32. trtexec's DEFAULT "fp32" is the TF32 tensor-core path,
#    which rounds matmul INPUTS to a 10-bit mantissa. That is a different numeric
#    configuration and it scores 1/200, not 0/200; only the TF32-CLEARED build has
#    ever been decision-exact. Omitting this flag would both inflate the fp32
#    baseline's speed and silently break the parity control.
#    (Equivalent to drivr build_trt_engine.py --precision fp32, which calls
#     config.clear_flag(trt.BuilderFlag.TF32).)
#  * --profilingVerbosity=detailed at BUILD time. Kernel/tactic names are baked
#    into the engine; profiling an engine built at default verbosity returns
#    stripped names, and absence-of-name would read as absence-of-kernel. That
#    matters here because the _gemm_mha_v2 fused-MHA check is the critical one.
#  * separate profile run: --dumpProfile inflates totals (46.2 vs 41.8 ms at N=6
#    was measured) and inflates large-tensor kernels most, so latency and profile
#    must not come from the same run.
#  * clock sampled THROUGH the run: nvhost_podgov drops to 306 MHz between
#    inferences. dev1 is pinned (governor=performance, min=max=918 MHz) but an
#    earlier version of the skill got this backwards by sampling cur_freq only
#    after each inference, so it is verified every time rather than assumed.
set -uo pipefail

ONNX="${1:?usage: dev1_measure.sh <onnx> <precision> <outdir>}"
PREC="${2:?}"
OUTDIR="${3:?}"

TRTEXEC=/usr/src/tensorrt/bin/trtexec
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
BASE="$(basename "${ONNX%.onnx}")"
TAG="$BASE.$PREC"
mkdir -p "$OUTDIR"
ENGINE="$OUTDIR/$TAG.trt"

case "$PREC" in
  fp32) FLAGS=(--noTF32) ;;
  tf32) FLAGS=() ;;
  fp16) FLAGS=(--fp16) ;;
  *) echo "unknown precision $PREC" >&2; exit 2 ;;
esac

echo "===== $TAG ====="
echo "-- host state before build (a mistimed engine is worse than no engine) --"
uptime
echo "clock: $(cat $DEVFREQ/cur_freq) governor=$(cat $DEVFREQ/governor)"

if [ -s "$ENGINE" ]; then
  echo "-- engine exists, skipping build --"
else
  echo "-- build --"
  T0=$(date +%s)
  "$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" \
    "${FLAGS[@]}" \
    --memPoolSize=workspace:"${WORKSPACE_MB:-4096}" \
    --profilingVerbosity=detailed \
    --skipInference --verbose \
    > "$OUTDIR/$TAG.build.log" 2>&1
  echo "build rc=$? in $(( $(date +%s) - T0 ))s"
  if [ ! -s "$ENGINE" ]; then
    echo "!! BUILD PRODUCED NO ENGINE:" >&2
    tail -30 "$OUTDIR/$TAG.build.log" >&2
    exit 1
  fi
fi
echo "engine: $(du -m "$ENGINE" | cut -f1) MiB"

# Confirm the precision flags actually took rather than trusting they did.
echo "-- builder precision state --"
grep -aiE "TF32|FP16|BF16|INT8" "$OUTDIR/$TAG.build.log" \
  | grep -aiE "disabl|enabl|precision" | sort -u | head -8

echo "-- benchmark (clock sampled through the run) --"
( while :; do cat $DEVFREQ/cur_freq; sleep 0.02; done ) > "$OUTDIR/$TAG.clock.txt" 2>/dev/null &
CLOCK_PID=$!
"$TRTEXEC" --loadEngine="$ENGINE" \
  --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000 \
  > "$OUTDIR/$TAG.bench.log" 2>&1
kill $CLOCK_PID 2>/dev/null; wait $CLOCK_PID 2>/dev/null

echo "-- clock distribution during benchmark --"
awk '{printf "%d\n", $1/1000000}' "$OUTDIR/$TAG.clock.txt" | sort -n | uniq -c \
  | awk '{printf "   %s MHz: %s samples\n", $2, $1}'

echo "-- GPU compute time --"
grep -aE "GPU Compute Time:|Latency:|Throughput:" "$OUTDIR/$TAG.bench.log"

echo "-- profile (separate run) --"
"$TRTEXEC" --loadEngine="$ENGINE" \
  --iterations=30 --avgRuns=10 --useSpinWait --warmUp=500 \
  --dumpProfile --separateProfileRun \
  --exportProfile="$OUTDIR/$TAG.profile.json" \
  --dumpLayerInfo --exportLayerInfo="$OUTDIR/$TAG.layers.json" \
  > "$OUTDIR/$TAG.profile.log" 2>&1

echo "-- CRITICAL CHECK: fused MHA kernel _gemm_mha_v2 --"
for f in "$OUTDIR/$TAG.build.log" "$OUTDIR/$TAG.layers.json" \
         "$OUTDIR/$TAG.profile.json" "$OUTDIR/$TAG.profile.log"; do
  [ -f "$f" ] && printf '   %-24s %s hits\n' "$(basename "$f")" \
    "$(grep -ao "_gemm_mha_v2" "$f" 2>/dev/null | wc -l)"
done
echo "   -- all *mha* tactic names present --"
grep -ahoE "[A-Za-z0-9_]*mha[A-Za-z0-9_]*" \
  "$OUTDIR/$TAG.build.log" "$OUTDIR/$TAG.layers.json" 2>/dev/null \
  | sort | uniq -c | sort -rn | head -12
echo "   (empty above = NO fused MHA at all)"

echo "DONE $TAG"
