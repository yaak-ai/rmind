#!/usr/bin/env bash
# Build + benchmark + profile one decoder-step engine on delta-dev1 (AGX Orin).
#
#   dev1_measure.sh <onnx> <fp32|fp16|fp16cache|tf32> <outdir>
#
# Flag rationale (/nasa/max/skills/trt-export/SKILL.md):
#  * fp16cache hands past_k/past_v/cache_bias and new_k/new_v over as fp16
#    (runbook §12.7). A plain --fp16 build does NOT halve the cache: TRT
#    preserves the dtypes the ONNX declares for network I/O, so the cache
#    bindings come back float32 and several ms/tick are burned reformatting
#    them. Measured on decoder_c1n9agobv1_n16: 120.47 -> 60.23 MiB, reformat
#    nodes 12 -> 7. The IOFormats strings are POSITIONAL, which is why the
#    binding order is asserted below.
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
PY="${PY:-python3}"
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
BASE="$(basename "${ONNX%.onnx}")"
TAG="$BASE.$PREC"
mkdir -p "$OUTDIR"
ENGINE="$OUTDIR/$TAG.trt"

# Positional, and correct ONLY for this binding order (asserted post-build):
#   in  0 image 1 speed 2 waypoints 3 past_k 4 past_v 5 cache_bias
#       6 rope_cos 7 rope_sin
#   out 0 policy.joint_actions 1 new_k 2 new_v
CACHE_IN=fp32:chw,fp32:chw,fp32:chw,fp16:chw,fp16:chw,fp16:chw,fp32:chw,fp32:chw
CACHE_OUT=fp32:chw,fp16:chw,fp16:chw
EXPECT_BINDINGS="inputs_image inputs_speed inputs_waypoints inputs_past_k inputs_past_v inputs_cache_bias inputs_rope_cos inputs_rope_sin policy.joint_actions new_k new_v"

case "$PREC" in
  fp32) FLAGS=(--noTF32) ;;
  tf32) FLAGS=() ;;
  fp16) FLAGS=(--fp16) ;;
  fp16cache) FLAGS=(--fp16 --inputIOFormats="$CACHE_IN" --outputIOFormats="$CACHE_OUT") ;;
  *) echo "unknown precision $PREC" >&2; exit 2 ;;
esac
echo "flags: ${FLAGS[*]}"

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
  RC=$?   # capture BEFORE the arithmetic below; $? in the echo would be fragile
  echo "build rc=$RC in $(( $(date +%s) - T0 ))s"
  if [ ! -s "$ENGINE" ]; then
    echo "!! BUILD PRODUCED NO ENGINE:" >&2
    tail -30 "$OUTDIR/$TAG.build.log" >&2
    exit 1
  fi
fi
echo "engine: $(du -m "$ENGINE" | cut -f1) MiB"

# NOTE: there used to be a "builder precision state" grep over the build log
# here. It was REMOVED, not fixed: it matched trtexec's unconditional
# "TF32 is enabled by default" hint and printed even under --noTF32, so it was a
# check that always passed. Precision is now read per-layer from
# --exportLayerInfo after the profile run below.

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

echo "-- PER-LAYER PRECISION (replaces the always-passing build-log grep) --"
# DETECTOR, NOT A GUARD: layers.json only exists after the engine is built, so a
# reordered ONNX yields a silently-wrong fp16cache engine that this then flags
# after the fact. There is no name-based --inputIOFormats in trtexec, so the
# positional string cannot be validated up front. Treat a FAIL here as "throw
# the engine away", not as "the build was prevented".
"$PY" - "$OUTDIR/$TAG.layers.json" "$PREC" "$EXPECT_BINDINGS" <<'PYEOF'
import collections, json, sys

layer_info, prec, expect = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3].split()
layers, bindings = layer_info["Layers"], layer_info.get("Bindings", [])
print("   layers: %d" % len(layers))


def out_fmts(layer):
    return [t.get("Format/Datatype", "?") for t in (layer.get("Outputs") or [])]


def is_fp16(fmt):
    # TRT spells the same dtype two ways: bare "Half" for internal tensors,
    # "Row major linear FP16 format" for bindings. Match both or the check lies.
    return "Half" in fmt or "FP16" in fmt


hist = collections.Counter(
    f for layer in layers if isinstance(layer, dict) for f in out_fmts(layer)
)
print("   output Format/Datatype histogram:")
for fmt, n in hist.most_common():
    print("      %-38s %d" % (fmt, n))
half = sum(n for fmt, n in hist.items() if is_fp16(fmt))
print("   => HALF/FP16 layer outputs: %d of %d" % (half, sum(hist.values())))
if prec == "fp32" and half:
    print("   !! fp32 build has %d FP16 layer outputs -- NOT a clean fp32 engine" % half)

# An fp32 output BINDING does not mean fp32 logits. On the fp16cache engine the
# final Reformat emits `policy.joint_actions` as "Row major linear FP32" while
# CONSUMING "Row major linear FP16" -- the fp32 is a cast of an fp16 result.
print("   TAIL 8 layers (fp32 binding != fp32 logits -- read the INPUT dtype):")
for layer in layers[-8:]:
    if isinstance(layer, dict):
        ins = [t.get("Format/Datatype", "?") for t in (layer.get("Inputs") or [])]
        print("      %-46s %-12s in=%s out=%s"
              % (str(layer.get("Name"))[:46], layer.get("LayerType"), ins, out_fmts(layer)))

if bindings != expect:
    print("   !! BINDING ORDER MISMATCH -- the positional IOFormats string is WRONG")
    print("      expected: %s" % expect)
    print("      actual:   %s" % bindings)
    sys.exit(1)
print("   bindings match the IOFormats assumption (%d)" % len(bindings))

if prec == "fp16cache":
    bad = [
        t.get("Name")
        for layer in layers if isinstance(layer, dict)
        for t in (layer.get("Outputs") or [])
        if t.get("Name") in ("new_k", "new_v") and not is_fp16(t.get("Format/Datatype", ""))
    ]
    if bad:
        print("   !! fp16cache did NOT halve the cache; still fp32: %s" % bad)
        sys.exit(1)
    print("   fp16cache confirmed: new_k/new_v are FP16")
PYEOF
echo "   layer-precision audit rc=$?"

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
