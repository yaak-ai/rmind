#!/usr/bin/env bash
# Re-bench the v0 engines that are still on dev1 from 6 Aug, BEFORE building v2.
#
# This is the discriminator for the task's "if v2 differs from v0 by more than
# ~1%, investigate" clause. Without it, a v2/v0 disagreement is ambiguous between
# (a) host state drifted in three days and (b) something about v2 is different.
# Re-benching v0 today, on the same host at the same pinned clock, separates
# them: if v0 no longer reproduces 59.82/16.97 the BASELINE moved, not v2.
#
# No rebuild -- these are the byte-identical engines from the §12 run.
set -u
TRTEXEC=/usr/src/tensorrt/bin/trtexec
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
E=/home/max/onnx_exports/decoder_trained/engines
O=/home/max/onnx_exports/decoder_causal_v2/rebench_v0
mkdir -p "$O"

echo "=== host state ==="; uptime
echo "clock: $(cat $DEVFREQ/cur_freq) governor=$(cat $DEVFREQ/governor)"

for TAG in n16.fp32 n16.fp16 n6.fp32 n6.fp16 n32.fp32 n32.fp16; do
  ENG=$E/decoder_do8m9ot8_$TAG.trt
  [ -s "$ENG" ] || ENG=$E/LATENCY-ONLY-WRONG-WINDOW.decoder_do8m9ot8_$TAG.trt
  if [ ! -s "$ENG" ]; then echo "-- MISSING $TAG"; continue; fi
  echo "===== v0 $TAG ($(du -m "$ENG"|cut -f1) MiB) ====="
  ( while :; do cat $DEVFREQ/cur_freq; sleep 0.02; done ) > "$O/$TAG.clock.txt" 2>/dev/null &
  CP=$!
  "$TRTEXEC" --loadEngine="$ENG" --iterations=60 --avgRuns=20 --useSpinWait \
    --warmUp=1000 > "$O/$TAG.bench.log" 2>&1
  kill $CP 2>/dev/null; wait $CP 2>/dev/null
  grep -aE "GPU Compute Time: min" "$O/$TAG.bench.log"
  awk '{printf "%d\n",$1/1000000}' "$O/$TAG.clock.txt" | sort -n | uniq -c \
    | awk '{printf "   clock %s MHz: %s samples\n",$2,$1}'
done
echo REBENCH_DONE
