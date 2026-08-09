#!/usr/bin/env bash
# Quarantine the wrong-window engines and leave the measured numbers beside them.
#
# The n6/n32 engines were built from a WINDOW-16 checkpoint. `step` reads the
# cache length off past_k's shape and has no intrinsic maximum, so they run
# happily against a 6- or 32-frame cache while silently extrapolating (docs
# §10.3). They are latency-curve points and nothing else. A wrong-window engine
# under a plausible name will eventually be served by someone, so the name has to
# say what it is.
set -u
O=/home/max/onnx_exports/decoder_causal_v2/engines
cd "$O" || exit 1

for N in 6 32; do
  for P in fp32 fp16; do
    F=decoder_do8m9ot8v2_n$N.$P.trt
    [ -s "$F" ] && mv -v "$F" "LATENCY-ONLY-WRONG-WINDOW.$F"
  done
done

cat > "$O/PARITY-NOTES.md" <<'EOF'
# decoder_do8m9ot8 **v2** (epoch 2, step 242392) — measured on delta-dev1

TRT 10.7.0.23, AGX Orin, GPU clock pinned 918 MHz (`governor=performance`,
verified by sampling `cur_freq` THROUGH every benchmark), host idle.
ONNX exported from `yaak/rmind/model-do8m9ot8:v2`, producer `pytorch 2.12.1+cu130`.

## Servable — window 16 (the trained window, the only valid one)

| engine | latency (median GPU compute) | KV cache | decisions @ n=200 | worst \|d\| |
| --- | --- | --- | --- | --- |
| `decoder_do8m9ot8v2_n16.fp32.trt` (`--noTF32`) | 59.67 ms | 120.47 MiB fp32 | **0/200** | 9.4e-07 |
| `decoder_do8m9ot8v2_n16.fp16cache.trt` | 13.66 ms | 60.23 MiB fp16 | 2/200 | 0.0428 |
| `decoder_do8m9ot8v2_n16.fp16.trt` | 17.05 ms | 120.47 MiB fp32 | 3/200 | 0.0579 |
| `decoder_do8m9ot8v2_n16.encfp32-fp16trunk.trt` | 21.80 ms | 120.47 MiB fp32 | 3/200 | 0.0547 |

**Serve fp32.** It is the only decision-exact configuration and there is no
latency pressure at this window to trade anything for it. Every flip in every
low-precision arm is a CONTROL channel (steering/throttle/brake); `turn_signal`
never crosses tolerance, so there is no "only the indicator moved" reading.

⚠️ The `fp16cache` parity number is OPTIMISTIC and must not be shipped on: the
harness streams each history through ORT in fp32 and casts to fp16 once per
trial, whereas a real fp16 ring accumulates the cache in fp16 across every tick.
Re-run the ladder with a genuinely fp16 ring before serving that engine.

## Quarantined — `LATENCY-ONLY-WRONG-WINDOW.*`

Built from the window-16 checkpoint at contexts 6 and 32 for the latency curve.
`step` will run them against any cache size while silently EXTRAPOLATING the
frame-RoPE beyond anything trained. **Do not serve. Not parity-tested.**

| context | fp32 | fp16 |
| --- | --- | --- |
| 6 | 41.80 ms | 10.57 ms |
| 32 | 92.22 ms | 27.17 ms |
EOF
echo "-- wrote PARITY-NOTES.md --"
ls -la "$O"/*.trt
echo FINALIZE_DONE
