#!/usr/bin/env bash
# The n=200 decision-parity ladder for v2, chained after every build.
#
# Run entirely on the Orin so the ORT reference and the engines see byte-identical
# inputs, and because a 126 MiB cache per trial cannot be shipped from the NAS.
# Histories are genuinely STREAMED here -- cold start, ring slot writes, monotone
# frame counter -- which is what drivr does; a cache assembled any other way would
# not exercise the thing being tested.
#
# fp32 first and non-negotiable: if the fp32 control is not 0/200 the HARNESS is
# broken and no other row means anything.
set -u
for i in $(seq 1 400); do
  grep -aq POSTBUILD_V2_DONE ~/post_v2.log 2>/dev/null && break
  sleep 15
done
grep -aq POSTBUILD_V2_DONE ~/post_v2.log || { echo "POSTBUILD DID NOT FINISH"; exit 1; }

echo "=== settle before scoring (engines are timed here too) ==="
for i in $(seq 1 40); do
  L=$(awk '{print $1}' /proc/loadavg); echo "  load $L"
  awk -v l="$L" 'BEGIN{exit !(l < 2.0)}' && break
  sleep 30
done
uptime

D=/home/max/onnx_exports/decoder_causal_v2
O=$D/engines
ONNX=$D/decoder_do8m9ot8v2_n16.onnx
B=decoder_do8m9ot8v2_n16

ENG="$O/$B.fp32.trt,$O/$B.fp16.trt"
LAB="fp32,fp16"
[ -s "$O/$B.encfp32-fp16trunk.trt" ] && ENG="$ENG,$O/$B.encfp32-fp16trunk.trt" && LAB="$LAB,encfp32-fp16trunk"
[ -s "$O/$B.fp16cache.trt" ]        && ENG="$ENG,$O/$B.fp16cache.trt"        && LAB="$LAB,fp16-fp16cache"
echo "engines: $LAB"

/home/max/Code/drivr/.venv/bin/python ~/decoder_parity_orin.py \
  --onnx "$ONNX" --engines "$ENG" --labels "$LAB" \
  --trials 200 --histories 10 \
  --frames /tmp/native_raw.png,/tmp/frame_video1.jpg,/tmp/frame_video0_.jpg \
  --out ~/parity_v2_n16_full.json
echo "PARITY_V2_DONE rc=$?"
