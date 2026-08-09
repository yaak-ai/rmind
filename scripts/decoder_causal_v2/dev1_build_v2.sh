#!/usr/bin/env bash
# Build + bench + profile the v2 decoder engines on delta-dev1.
#
# Serial, never concurrent: TRT selects kernels by TIMING them, so a second build
# (or a parity run) competing for the GPU bakes slower tactics into whichever
# engine is still building. A mistimed engine is worse than no engine.
#
# Window 16 is the SERVED context. 6 and 32 are latency-curve points only -- they
# were built from a window-16 checkpoint, which `step` runs against any cache
# length while silently extrapolating (docs §10.3). They get renamed
# LATENCY-ONLY-WRONG-WINDOW.* at the end so nobody serves one.
set -u
M=/home/max/Code/rmind-causal-v2/src/rmind/scripts/decoder_only_trt_measure.sh
D=/home/max/onnx_exports/decoder_causal_v2
O=$D/engines
mkdir -p "$O"

echo "###### host state at start ######"
uptime
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/governor

for N in 16 6 32; do
  for P in fp32 fp16; do
    echo "############################################################"
    "$M" "$D/decoder_do8m9ot8v2_n$N.onnx" "$P" "$O"
  done
done
echo ALL_BUILDS_DONE
