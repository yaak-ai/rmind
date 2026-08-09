#!/usr/bin/env bash
# Chained after the fp32/fp16 sweep: the fp16 KV-cache-I/O engine, the mixed
# fp32-encoder/fp16-trunk engine, the engine-level fused-MHA interrogation, the
# cache-binding memory report, and the n=200 parity ladder.
#
# Chained rather than concurrent throughout: TRT picks kernels by TIMING them, so
# anything competing for the GPU while a build runs bakes in slower tactics.
set -u
for i in $(seq 1 400); do
  grep -aq ALL_BUILDS_DONE ~/build_v2.log 2>/dev/null && break
  sleep 15
done
grep -aq ALL_BUILDS_DONE ~/build_v2.log || { echo "BUILDS DID NOT FINISH"; exit 1; }

echo "=== wait for the host to settle (TRT times kernels; load poisons them) ==="
for i in $(seq 1 40); do
  L=$(awk '{print $1}' /proc/loadavg); echo "  load $L"
  awk -v l="$L" 'BEGIN{exit !(l < 2.0)}' && break
  sleep 30
done
uptime

DRIVR=/home/max/Code/drivr
D=/home/max/onnx_exports/decoder_causal_v2
O=$D/engines
ONNX=$D/decoder_do8m9ot8v2_n16.onnx
TRTEXEC=/usr/src/tensorrt/bin/trtexec
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu

# ---------------------------------------------------------------- fp16 cache
# A plain --fp16 build does NOT halve the KV cache: TRT preserves the dtypes the
# ONNX declares for network I/O and casts internally, so past_k/past_v come back
# float32 and the engine pays ~2.5 ms/tick of reformat copies. The halving must be
# asked for per tensor. Input order verified against this graph:
#   0 image  1 speed  2 waypoints  3 past_k  4 past_v  5 cache_bias  6 cos  7 sin
# Outputs: 0 policy.joint_actions  1 new_k  2 new_v
E=$O/decoder_do8m9ot8v2_n16.fp16cache.trt
echo; echo "=== build fp16 + fp16 KV-cache I/O ==="
if [ -s "$E" ]; then echo "-- exists, skip"; else
T0=$(date +%s)
$TRTEXEC --onnx=$ONNX --saveEngine=$E --fp16 \
  --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp16:chw,fp16:chw,fp16:chw,fp32:chw,fp32:chw \
  --outputIOFormats=fp32:chw,fp16:chw,fp16:chw \
  --memPoolSize=workspace:4096 --profilingVerbosity=detailed --skipInference --verbose \
  > $O/fp16cache.build.log 2>&1
echo "rc=$? in $(( $(date +%s) - T0 ))s engine=$(du -m $E 2>/dev/null|cut -f1) MiB"
fi
if [ -s "$E" ]; then
  ( while :; do cat $DEVFREQ/cur_freq; sleep 0.02; done ) > $O/fp16cache.clock.txt 2>/dev/null &
  CP=$!
  $TRTEXEC --loadEngine=$E --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000 \
    > $O/fp16cache.bench.log 2>&1
  kill $CP 2>/dev/null; wait $CP 2>/dev/null
  grep -aE "GPU Compute Time: min" $O/fp16cache.bench.log
  awk '{printf "%d\n",$1/1000000}' $O/fp16cache.clock.txt | sort -n | uniq -c \
    | awk '{printf "   clock %s MHz: %s\n",$2,$1}'
  # profile it too: §12.7's finding was that the two most expensive kernels in the
  # PLAIN fp16 engine are cache reformat copies, which this engine should not have
  $TRTEXEC --loadEngine=$E --iterations=30 --avgRuns=10 --useSpinWait --warmUp=500 \
    --dumpProfile --separateProfileRun --exportProfile=$O/fp16cache.profile.json \
    --dumpLayerInfo --exportLayerInfo=$O/fp16cache.layers.json \
    > $O/fp16cache.profile.log 2>&1
else
  echo "!! fp16-cache build FAILED"; grep -aiE "error|not supported" $O/fp16cache.build.log | sort -u | head -8
fi

# ------------------------------------------------------------------- mixed
echo; echo "=== derive fp32 ranges for THIS graph (never hardcode them) ==="
$DRIVR/.venv/bin/python $DRIVR/scripts/precision_ranges.py $ONNX 2>&1 | tee ~/ranges_v2.txt
RANGES=$(grep -aoE "^fp32 ranges *: *[0-9,-]+" ~/ranges_v2.txt | sed 's/.*: *//')
echo "RANGES=[$RANGES]"
MIXED=$O/decoder_do8m9ot8v2_n16.encfp32-fp16trunk.trt
if [ -n "$RANGES" ]; then
  echo "=== build mixed: fp32 encoder + fp16 trunk + fp32 decode ==="
  $DRIVR/.venv/bin/python $DRIVR/scripts/build_mixed.py \
    --onnx $ONNX --precision mixed --workspace-gb 4 \
    --fp32-index-ranges "$RANGES" --engine $MIXED > $O/mixed.build.log 2>&1
  echo "rc=$? engine=$(du -m $MIXED 2>/dev/null|cut -f1) MiB"
  tail -4 $O/mixed.build.log
  if [ -s "$MIXED" ]; then
    $TRTEXEC --loadEngine=$MIXED --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000 \
      > $O/mixed.bench.log 2>&1
    grep -aE "GPU Compute Time: min" $O/mixed.bench.log
    # build_mixed.py does NOT set profiling_verbosity, so its tactic names are
    # partly stripped and its BUILD LOG reports 0 _gemm_mha_v2 hits even when the
    # fusion is present (§12.8's false alarm). Interrogate the ENGINE instead.
    $TRTEXEC --loadEngine=$MIXED --iterations=30 --avgRuns=10 --useSpinWait --warmUp=500 \
      --dumpProfile --separateProfileRun --exportProfile=$O/mixed.profile.json \
      --dumpLayerInfo --exportLayerInfo=$O/mixed.layers.json > $O/mixed.profile.log 2>&1
  fi
else
  echo "!! could not derive ranges -- skipping the mixed arm"
fi

echo POSTBUILD_V2_DONE
