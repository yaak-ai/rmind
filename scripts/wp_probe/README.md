# Waypoint probes (permutation + mirror)

Causal, predict-time probes for *how* a checkpoint uses waypoints for steering —
without retraining. Two metrics:

- **Permutation importance** — shuffle `data/waypoints/xy_normalized` across the
  batch and measure the rise in steering L1. ΔL1 > 0 ⇒ the model uses waypoint
  information. (`FeaturePermutator`)
- **Mirror test** — negate the lateral waypoint coord (axis 0 of `(x, y)`); if
  steering is driven by the signed path direction, predicted steering flips ⇒
  `corr(baseline, mirror) < 0`. Negating the longitudinal coord (axis 1) is a
  control that should barely move steering. (`FeatureReflector`)

Waypoint tensor is `[b, t, 10 waypoints, 2 (x, y)]`; **axis 0 = lateral**,
axis 1 = longitudinal (verified: axis 0 is symmetric ±, axis 1 mostly positive).

## Running

All conditions must see **identical frames** for the paired mirror test — set
`shuffle=true` with a seeded `torch.Generator` (deterministic order across runs):

```bash
# policy (fine-tuned) head:  inference=...policy_wp_probe, models via model.artifact
# pretrain inverse-dynamics: inference=...pretrain_wp_probe
for cond in "baseline none" "waypoints none" "baseline lateral" "baseline longitudinal"; do
  set -- $cond
  rmind-predict --config-name predict.yaml \
    inference=yaak/control_transformer/policy_wp_probe \
    model.artifact=yaak/rmind/<MODEL_ARTIFACT> \
    permutation=$1 reflection=$2 \
    datamodule.predict.shuffle=true \
    ++datamodule.predict.generator._target_=torch.Generator \
    paths.rbyte.cache=/tmp/rbyte_cache \
    +trainer.limit_predict_batches=100 \
    hydra.run.dir=/tmp/wp_matrix/<MODEL_TAG>/$1-$2
done
```

## Analyzing

```bash
python scripts/wp_probe/analyze.py /tmp/wp_matrix --objective policy --models mean concat
python scripts/wp_probe/analyze.py /tmp/wp_matrix --objective inverse_dynamics --models base pe
```

Reports row-alignment (sanity), permutation ΔL1, and mirror corr/flip — overall
and on turning frames (`|gt steer| >= 0.05`). The acceptance target for a good
waypoint representation is mirror corr → negative (the mean-aggregation policy
baseline reaches ≈ −0.65).
