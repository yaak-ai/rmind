# DOE: trajectory mode count (`num_trajectory_hypotheses`)

Baseline: `yaak/alex-tmp/dfldy2ki` (celestial-sun-425), commit `2b69760f`,
experiment `yaak/patch_policy/dinov2_dinowm_causal_readout_dropout_depth_anything_v2_r1b_r2`,
`K = num_trajectory_hypotheses = 5`.

## Question

How does the number of trajectory modes K change the trajectory we would actually
serve (the mode the mode head picks)? Does it affect the main policy heads that
share the readout feature?

## What the baseline already shows

| metric (val, end of run) | value | reading |
|---|---|---|
| `mode_accuracy` | 0.371 | chance for K=5 is 0.20. If one mode dominates, the majority-class rate is the better reference, and it is not logged |
| `trajectory_best_index_unique_frac` | 0.0039 | ≈ 3 distinct winners per 768-sample batch, so about 2 of the 5 modes look unused |
| `trajectory_loss_xy` (winner) | 0.152 m | oracle error (best of K), not the error of the trajectory we serve |
| `trajectory_loss_heading` (winner) | 0.70 ° | same |

So at K=5 the model may already use fewer modes than it has. That makes the
low end of the sweep (K ≤ 3) as interesting as the high end.

## Prerequisites (done)

### P1. Fix the mode-label bug ✅

`PatchPolicy._readout_metrics` (`src/rmind/models/patch_policy.py:1243`)
recomputes `best_index` with `winner_takes_all_pose_l1_components` and passes
the loss module's `heading_weight` and `reduction`, but **not `xy_weight`**. It
therefore uses the default `_DEFAULT_XY_WEIGHT = 0.01`, while the loss that
trains the trajectory head uses `trajectory_xy_weight = 4.0` (dfldy2ki).

Consequences:
- The mode head learns labels (`best_index`) picked under a heading-dominated
  weighting (xy 0.01 vs heading 0.5). The trajectory head's actual winner is
  picked under an xy-dominated one (4.0 vs 0.5). These two can disagree.
- `mode_accuracy`, `trajectory_best_index_unique_frac`, and the winner
  `trajectory_loss_xy`/`trajectory_loss_heading` are all measured against the
  wrong winner.

The same omission was in `predict_step`'s `best_index`, which also used a
wrong fallback heading weight (0.1).

Fix: `WinnerTakesAllPoseLoss.components()` runs
`winner_takes_all_pose_l1_components` with the module's own weights. Both the
metric/label path and `predict_step` now call it, and `PatchPolicy` requires
`losses["trajectory"]` to be a `WinnerTakesAllPoseLoss`, so the label always
picks the same winner as the optimized loss. A regression test covers the case
where the old and new weightings disagree.

### P2. Add metrics that are fair across K ✅

The oracle metric (best of K) always improves as K grows, so it cannot rank
arms. All names below sit under `{train,val}/policy/metric/`.

Per batch (Lightning averages them over the epoch):

| metric | definition | why |
|---|---|---|
| `trajectory_selected_xy` / `_heading` | error of the served hypothesis: the mode head's argmax, or the only one at K=1 | **primary**: this is what we serve |
| `trajectory_selected_fde_xy` | final-pose xy error of the served hypothesis | long-horizon drift |
| `trajectory_selected_xy_last` / `_fde_xy_last` | the same at the newest frame only | matches the deployed readout |
| `trajectory_loss_xy` / `_heading` (existing names) | oracle winner error (minADE@K) | ceiling; the gap to selected is what mode selection costs |
| `trajectory_oracle_fde_xy` | oracle winner final-pose xy error | the same, for FDE |
| `mode_nll` | unweighted CE of the mode head against `best_index` | calibration; not affected by `mode_weight` |

Per val epoch, computed from counts summed across ranks (val only):

| metric | definition | why |
|---|---|---|
| `mode_win_rate/<k>` | share of samples that hypothesis k wins | the full usage histogram |
| `mode_usage_entropy_norm` | entropy of that histogram / ln K (K>1 only) | 1 = all modes used, → 0 = collapse |
| `mode_dead_count` | modes with win rate < 1 / (10 K) | count of unused modes |
| `mode_majority_rate` | largest win rate | the honest chance level |
| `mode_accuracy_epoch` / `mode_accuracy_lift` | accuracy, and accuracy minus `mode_majority_rate` | skill over always guessing the most common mode, not over 1/K |

The selected-mode metrics are not logged for K>1 without a mode head, because
nothing would be served. The existing `predict/*` pedal/steering L1 and
`val/policy/loss/code_*` remain the guardrails.

## Design

**Factor:** K ∈ {1, 3, 5, 8, 12}

- K=1 is the unimodal control: winner-takes-all becomes plain L1, and the mode
  head is trivial (its CE is 0).
- K=5 re-runs the baseline with P1 and P2 in place. It cannot be compared
  directly with dfldy2ki because the mode labels change.
- Head cost is negligible: the last layer is 1024 × (15·K), i.e. 180 outputs at K=12.

**Held fixed:** everything in dfldy2ki's command line (below), including
`mode_weight=0.2`, `trajectory_weight=1.0`, xy/heading weights, LR schedule and data.

Note: the uniform-prediction CE is ln K (0.0 at K=1, 1.10 at K=3, 1.61 at K=5,
2.08 at K=8, 2.48 at K=12), so at fixed `mode_weight` the mode term's gradient
share grows a little with K. This is accepted as part of the treatment. If the
code losses regress at high K, the first follow-up is `mode_weight = 0.2 · ln5 / lnK`.

### Phase A: screening (5 runs)

One seed (1337) per K, 60k steps (≈ 28 % of the full 215k schedule). Set
`lr_total_steps=60000` so the cosine schedule completes rather than being cut
off mid-decay. Purpose: rule out clearly bad K and see where mode collapse
sets in.

### Phase B: confirmation (≤ 6 runs)

Full 215k-step schedule, matching the baseline, for the best 2 K values from
Phase A plus K=5. Use seeds {1337, 2024}. The second seed gives a noise
estimate. Without it, differences of a few cm in `trajectory_selected_xy` cannot
be called.

Total: 5 short + 6 full runs ≈ 7.4 full-run equivalents.

## Hypotheses

- **H1:** oracle xy decreases monotonically with K. This is a sanity check; if it
  fails, the trajectory head is under-trained.
- **H2:** selected xy is U-shaped in K. Larger K improves coverage, but mode
  selection gets harder and more modes die.
- **H3:** `mode_dead_count` grows with K. Above some K, extra modes add nothing.
- **H4:** the policy heads (`code_*`, `predict/*`) are unaffected by K
  (|Δ| within seed noise).

## Decision rule

Choose the smallest K whose `trajectory_selected_xy` is within one seed-std of
the best arm, subject to:
- `mode_usage_entropy_norm ≥ 0.7`,
- `mode_dead_count ≤ 1`,
- no guardrail regression larger than seed noise on `val/policy/loss/code_0`
  and `predict/policy/score_l1/*`.

If K=1 wins, the mode head is not paying for itself. Drop it, and treat
multimodality as a closed-loop question rather than an offline one.

## Configs

Under `config/experiment/yaak/patch_policy/`:

| file | arm |
|---|---|
| `dinov2_dinowm_causal_readout_dropout_depth_anything_v2_r1b_r2_modes.yaml` | DOE base = K=5 control. It bakes in every hyperparameter override that dfldy2ki passed on its command line. The composed config was checked against dfldy2ki's logged config, and all model/data/trainer keys match. |
| `..._r1b_r2_modes_k1.yaml` | K=1: `num_trajectory_hypotheses: 1`, `model.mode_head: null`. This is the only change from the base: trajectory head output becomes 15 (1 × 5 poses × 3), there's no mode head, and the `mode_head` tag is dropped. |

K ∈ {3, 8, 12} can be launched from the base config with
`++num_trajectory_hypotheses=K`. The heads' widths interpolate from that value.

## Launch

The launch command adds only infrastructure overrides (the same ones dfldy2ki
used). Keep the `paths.rbyte.cache` name: it's the 16-step trajectory cache.

```sh
INFRA=(
  ++datamodule.train.dataset.samples.executor.max_workers=8
  ++datamodule.val.dataset.samples.executor.max_workers=8
  ++datamodule.train.dataset.samples.resume=true
  ++datamodule.val.dataset.samples.resume=true
  datamodule.train.num_workers=2
  wandb.project=alex-tmp
  paths.rbyte.cache=.rbyte_cache_16step_traj
)
EXP=yaak/patch_policy/dinov2_dinowm_causal_readout_dropout_depth_anything_v2_r1b_r2_modes
PHASE_A=(++lr_total_steps=60000 ++trainer.max_steps=60000 seed=1337)

# Phase A, K=1
just train experiment=${EXP}_k1 "${INFRA[@]}" "${PHASE_A[@]}"

# Phase A, K=5 control (needed to compare against K=1 -- dfldy2ki is NOT
# comparable: pre-fix labels, and 215k steps vs 60k)
just train experiment=$EXP "${INFRA[@]}" "${PHASE_A[@]}"

# Phase A, the other arms
for K in 3 8 12; do
  just train experiment=$EXP "${INFRA[@]}" "${PHASE_A[@]}" \
    ++num_trajectory_hypotheses=$K
done
```

Phase B is the same command without `PHASE_A`, which leaves the base's
`lr_total_steps=215000`, with `seed=1337` and `seed=2024`.

Notes:
- `config/trainer/default.yaml` limits run length by `max_epochs: 10`. In
  Phase A, `++trainer.max_steps=60000` caps the run earlier, because Lightning
  stops at whichever limit comes first.
- Every arm is tagged `mode_count_doe` plus `K${num_trajectory_hypotheses}`.
  The tag follows the override, so a K=8 launch is tagged `K8` without extra flags.
- `just train` runs `check-git`, so commit (and push) these changes first. The
  run's `metadata.git.commit` then pins the code that produced it.
