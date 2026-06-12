#!/usr/bin/env bash
# Run the policy waypoint-probe conditions for one checkpoint, then analyze.
# Usage (inside `nix develop`):  bash scripts/wp_probe/run_conditions.sh <wandb_artifact> <tag>
# e.g. bash scripts/wp_probe/run_conditions.sh yaak/rmind/model-<runid>:v<best> mean
set -uo pipefail
export HOSTNAME=$(hostname)
ART="$1"; TAG="$2"
NB="${NB:-100}"
ROOT="${ROOT:-/tmp/wp_screen}"
cd /home/max/Code/rmind-wpts-ft

for cond in "baseline none" "waypoints none" "baseline lateral"; do
  set -- $cond
  echo ">>> $TAG  perm=$1 refl=$2"
  uv run --with contextily rmind-predict \
    --config-path "$PWD/config" --config-name predict.yaml \
    inference=yaak/control_transformer/policy_wp_probe \
    model.artifact="$ART" permutation="$1" reflection="$2" \
    datamodule.predict.shuffle=true \
    "++datamodule.predict.generator._target_=torch.Generator" \
    paths.rbyte.cache=/nasa/max/rbyte_cache \
    +trainer.limit_predict_batches="$NB" \
    hydra.run.dir="$ROOT/$TAG/$1-$2" 2>&1 | tail -1
done
echo "=== analyze $TAG ==="
uv run python scripts/wp_probe/analyze.py "$ROOT" --objective policy --models "$TAG"
echo "DONE $TAG"
