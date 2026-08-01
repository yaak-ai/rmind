# Counterfactual max-speed override probe

Checkpoint: `/home/max/Code/rmind-traffic-rules/lightning_logs/tlq9u7t4/checkpoints/epoch=0-step=250.ckpt`
Inputs: real val batches from experiment=yaak/patch_policy/dinov2_dinowm_maxspeed (seed=1337, 2 batches); 64 samples/override; argmax decoding; vocab_size=13.

All deltas vs the `None(UNKNOWN)` baseline, last-frame readout. `kl` = mean KL(base || override) per (sample, quantizer); `code_flips` = argmax code change rate; `d*` = decoded-action chunk deltas (denormalized units, mean over samples x horizon).

| override | kl | max_dlogit | code_flips | dgas_abs | dgas_signed | dbrake_abs | dbrake_signed | dsteer_abs | dsteer_signed |
|---|---|---|---|---|---|---|---|---|---|
| 5(WALK) | 2.086e-07 | 7.124e-03 | 0.000e+00 | 2.316e-05 | -1.058e-07 | 2.206e-05 | -3.952e-06 | 2.540e-05 | -3.722e-06 |
| 10 | 6.881e-07 | 0.0141 | 3.906e-03 | 1.728e-04 | -4.759e-05 | 4.606e-03 | -4.575e-03 | 1.878e-04 | 7.785e-05 |
| 30 | 2.916e-07 | 0.0120 | 3.906e-03 | 1.693e-04 | -5.847e-05 | 4.598e-03 | -4.578e-03 | 1.792e-04 | 8.105e-05 |
| 50 | 5.001e-07 | 0.0147 | 3.906e-03 | 1.710e-04 | -5.876e-05 | 4.597e-03 | -4.573e-03 | 1.780e-04 | 7.661e-05 |
| 100 | 1.044e-07 | 6.102e-03 | 0.000e+00 | 2.770e-05 | -5.459e-06 | 3.476e-05 | -1.082e-05 | 2.312e-05 | -3.835e-07 |
| -1(UNLIMITED) | 1.263e-06 | 0.0226 | 3.906e-03 | 1.719e-04 | -6.322e-05 | 4.602e-03 | -4.577e-03 | 1.836e-04 | 8.318e-05 |

## Pairwise distinctness (min over batches of max |logit delta|)

- 10 vs -1(UNLIMITED): 5.074e-03
- 10 vs 100: 1.387e-02
- 10 vs 30: 2.354e-03
- 10 vs 50: 1.440e-03
- 100 vs -1(UNLIMITED): 1.324e-02
- 30 vs -1(UNLIMITED): 7.209e-03
- 30 vs 100: 6.030e-03
- 30 vs 50: 2.838e-03
- 5(WALK) vs -1(UNLIMITED): 1.796e-02
- 5(WALK) vs 10: 1.367e-02
- 5(WALK) vs 100: 4.884e-03
- 5(WALK) vs 30: 1.079e-02
- 5(WALK) vs 50: 1.436e-02
- 50 vs -1(UNLIMITED): 4.436e-03
- 50 vs 100: 9.958e-03
- None(UNKNOWN) vs -1(UNLIMITED): 1.083e-02
- None(UNKNOWN) vs 10: 1.402e-02
- None(UNKNOWN) vs 100: 1.770e-03
- None(UNKNOWN) vs 30: 3.663e-03
- None(UNKNOWN) vs 5(WALK): 6.174e-03
- None(UNKNOWN) vs 50: 7.571e-03

## None == all-UNKNOWN check: max |logit delta| = 0

## Verdict

PASS: overrides flow through (all override pairs pairwise-distinct in logits) and override=None is bitwise-identical to an all-NaN max_speed input (missing == all-UNKNOWN).
