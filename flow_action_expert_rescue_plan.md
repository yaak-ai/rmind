# Flow Action Expert Rescue Plan

## Summary

Run a focused 1-2 day experiment sequence to determine whether the current flow
action expert can be made to converge on the corrected single-drive overfit.
Optimize for saving the flow path first, using open-loop W&B metrics plus
`just fan`/horizon diagnostics as the decision gate.

## Key Changes

- Add a no-EMA flow callback config for overfit diagnostics: same freezer and PE
  logging as `finetune_flow`, but validation must use raw weights because EMA
  lagged in this regime.
- Add or extend fan/eval output to emit a small machine-readable metrics table:
  spike/flat hit@0.05/0.1/0.2, spike/flat mean L1, best-of-32 L1, per-horizon
  L1, and horizon lag/slope ratio.
- Add first-class maneuver metrics only after the fan table is stable; use raw
  action space for all reported metrics, especially for transformed-action
  experiments.
- Do not change conditioning in this cycle. Keep summaries + waypoints and leave
  image tokens out.

## Experiment Sequence

1. **Genuine single-drive overfit, current flow**
   - Run corrected `yaak/overfit` with frozen encoder, RoPE on,
     `chunk_delta_weight=10`, heun/32 validation, no EMA validation.
   - Stretch LR schedule so it does not hit floor early: target roughly one
     cosine cycle over the full 400 epochs.
   - Gate with `just fan` at heun/32 and 32 samples.
   - Pass criteria: `pe_drift` clearly > 0, horizon lag flattens toward 0, spike
     hit@0.05 improves over the old 56% reference, and raw-space
     `sample_l1`/fan L1 continue trending down rather than stalling.

2. **If overfit fits but chunks stay blurred**
   - Implement one architecture escalation: slot-conditioned AdaLN, by
     conditioning modulation on `(flow_time, slot_id)`.
   - Keep everything else fixed. Re-run only the single-drive overfit and
     fan/horizon diagnostics.
   - Pass criteria: lag profile no longer follows `h - 3`, chunk slope ratio
     moves toward 1, with no spike regression.

3. **If flat improves but spikes remain poor**
   - Implement continuous steering-only mu-law action transform inside
     `FlowPolicyObjective`.
   - Train in transformed space, sample/invert back to raw space, and compute all
     metrics in raw space.
   - Do not revive linear per-channel standardization; it already improved bulk
     while hurting spikes.
   - Pass criteria: spike L1 and spike hit improve without losing the flat gains.

4. **If both current flow and mu-law fail**
   - Stop flow rescue for this PR cycle.
   - Mark the PR as instrumentation/diagnostics plus negative result, and open a
     separate plan for categorical/discrete action heads.

## Public Interfaces

- New Hydra callback config: `trainer/callbacks=finetune_flow_no_ema`.
- Optional `FlowPolicyObjective` hparams if transform is needed:
  - `action_transform: null | "steering_mulaw"`
  - `mulaw_mu: 255`
- Optional fan output flag:
  - `+fan.metrics_out=...json` for spike/flat/horizon metrics.

## Test Plan

- Run `just test tests/test_flow_policy.py` after metric/config changes.
- Add tests for mu-law round-trip and raw-space metric computation if the
  transform is implemented.
- Add a config-instantiation test for the no-EMA callback and any new objective
  hparams.
- Verify `just fan` can run from both a live artifact and cached `.npz`.

## Assumptions

- Evaluation is open-loop only for this cycle.
- Base artifact remains `yaak/rmind/model-e61ycirr:v9`.
- Primary overfit drive is `Niro096-HQ/2023-01-11--13-47-36`.
- Overfit/debug runs use isolated rbyte cache paths to avoid concurrent cache
  collisions.
- Flow-space loss is not comparable across action transforms; raw-space fan
  metrics decide.
