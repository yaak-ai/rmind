#!/usr/bin/env bash
# Export the v2 decoder step-graph. Run on the LOCAL box, not delta-dev1: dev1's
# only surviving interpreter is Python 3.10 (the nix 3.12 the rmind venv pointed
# at has been garbage-collected) and the branch needs `typing.override`, i.e.
# 3.12. This is also the environment-matched choice rather than a fallback --
# v0's ONNX carries producer `pytorch 2.12.1+cu130`, which is exactly this box's
# torch, so the v2-vs-v0 latency comparison is not confounded by an exporter
# change.
#
# CUDA_VISIBLE_DEVICES=-1 keeps this off the local GPU (the empty string does NOT
# hide a device under torch 2.12). Export is CPU-side anyway.
set -u
S=/tmp/claude-30035/-home-max-Code-rmind-rqv/f1d292a7-fe84-4328-b3fe-bb3e53fa3f6c/scratchpad
WT=$S/wt-causal
D=$S/onnx_v2
PY=/home/max/Code/rmind-rqv/.venv/bin/python
mkdir -p "$D"
cd "$WT" || exit 1

for N in 16 6 32; do
  OUT=$D/decoder_do8m9ot8v2_n$N.onnx
  if [ -s "$OUT" ]; then echo "-- exists, skip: $OUT"; continue; fi
  echo "===== export context $N ====="
  T0=$(date +%s)
  PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES=-1 "$PY" -m rmind.scripts.decoder_only_export \
    --mode decoder --artifact yaak/rmind/model-do8m9ot8:v2 \
    --context "$N" --out "$OUT" > "$D/export_n$N.log" 2>&1
  echo "rc=$? in $(( $(date +%s) - T0 ))s"
  ls -la "$OUT" 2>/dev/null || { echo "!! NO ONNX"; tail -30 "$D/export_n$N.log"; }
done
echo "-- sizes (v0 was 212368893 B at every context) --"
ls -la "$D"/*.onnx 2>/dev/null
echo EXPORT_DONE
