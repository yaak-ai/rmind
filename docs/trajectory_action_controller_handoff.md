# Trajectory → action controller: findings handoff

Self-contained brief for continuing this investigation. Written for an agent
with no prior context on this conversation — everything needed to reproduce
or extend the findings is here or in the referenced files.

## Goal and scope

Build, entirely offline from one predict parquet (no Hydra/rbyte/checkpoint
access), a pipeline that:

1. reconstructs per-drive action traces (gas/brake/steering) from overlapping
   predict windows,
2. builds a trajectory ground truth by dead-reckoning heading + speed
   (replacing the flawed GPS-chain `policy/trajectory_gt`, see
   `[[project_frozen_vs_unfrozen_trajectory]]`),
3. trains a model that maps a future trajectory → the action(s) that realize
   it (an inverse-dynamics "controller"),
4. evaluates it as an "MPC-lite" on top of a trajectory a model predicts,
5. sweeps both of its horizons — the *prediction* horizon (how far ahead the
   plan it consumes reaches) and the *control* horizon (how many actions it
   emits per pass) — on one shared anchor set.

Explicitly **not** built: true receding-horizon MPC. The controller shares
MPC's interface — take a plan, emit a control sequence, apply index 0,
re-solve next tick — but not its machinery: no forward model, no cost
function, no optimizer search over action sequences. It is a direct learned
inverse map. No trajectory-predicting checkpoint is targeted yet; evaluation
uses `policy/trajectory_value` already sitting in the parquet as the stand-in.

**If you read only one thing**: the extended horizon is worth taking
(finding #9 — `horizon=30, action_steps=1`, −9.9% brake L1, +3.8pp turn
signal), the near-stop band it was built to fix is already at its
information floor and cannot be fixed this way (#10), and emitting an action
*sequence* makes the applied action worse (#11). Of the things tried since:
the per-drive calibration *vector* is the one clear win (#13 — −12.5% gas L1,
gas head only), and clipping the brake target to dodge the near-stop
ambiguity backfires — clip the output instead (#14).

Full original plan (scope decisions, file layout, verification commands):
`/home/alex/.claude/plans/can-we-1-get-streamed-toucan.md`.

## Data and reproducibility

- Parquet: `outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.parquet`
  (1,971,336 windows, 655 drives, predict-on-train). 11-tick windows: 5
  history + 6 horizon, tick spacing **0.333s** (measured directly), so the 6
  horizon ticks span **1.667s** past `GT_IDX`.
- `FEAT_IDX=4` ("now"), `GT_IDX=5` (first scored horizon tick, matches
  `policy/prediction_value`).
- **Derived**: `.../model-4vqatiom:v4.ticks.parquet` (40 MB, 2,087,541 rows)
  is the same data unpacked to one row per `(drive, tick)` — the input for
  anything horizon-related. Start here, not from the 516 MB source; see
  finding #8.
- `batch/meta/input_id` = `<vehicle>/<drive-timestamp>`, e.g.
  `Niro096-HQ/2023-01-11--13-47-36`. 27 distinct vehicles in this parquet (all
  "Niro*-HQ").
- All Python must run via `nix develop --command uv run python ...` — plain
  `python` inside `nix develop` breaks on `libstdc++.so.6`
  (`[[project_rmind_training_gotchas]]`).
- Standard eval split used throughout below: `load(..., n_drives=250,
  seed=7)`, then `np.random.default_rng(7).shuffle(drives)` on the
  **sorted, deduplicated** drive list, 10% held out as `val_drives`. The
  sort is load-bearing — see bug #2 below.
- "Per-drive" dynamics fits: split each val drive's rows in half by
  `frame_idx` order, fit on the first half, evaluate on the second half
  (same-session calibration, no leakage).

### Cross-table comparability: "the MLP" is not one number

Read every L1 in this document as *within-table* only. The same nominal
`TrajectoryToActionMLP` is reported at wildly different absolute values across
findings, because the anchor set, the split, the batch size, the row subset and
(before bug #5) the weight initialization all differ:

| finding | gas | brake | steering | scored on |
|---|---|---|---|---|
| #1 baseline | 0.0382 | 0.0182 | 0.0188 | `n_drives=250`, 6-tick, single seed, pre-bug-#5 |
| #5 / #7 MLP | 0.060 | 0.035 | 0.022 | `n_drives=250`, 6-tick, **against the per-drive parametric protocol** |
| #9 h6 | 0.0435 | 0.0251 | 0.0212 | full file, 1.88M h30 anchors, `batch_size=4096` |
| #9 h30 | 0.0432 | 0.0226 | 0.0214 | same |
| #13 `base` | 0.0444 | 0.0200 | 0.0191 | 932k **second-half** anchors, `batch_size=1024` |
| #14 baseline | — | 0.02253 | — | full file, h30, raw brake target, 5 seeds |

The #1-vs-#5 gap (1.6–1.9x on the *same* stated split) is the one that is
**not yet explained**. The plausible cause is that #5 scores only the second
halves of val drives — it has to, because the parametric arm it compares
against needs the drive's first half for calibration — while #1 scores all val
rows; #13's `base` at 0.0200 brake on an explicitly second-half anchor set is
consistent with that. But both were ad hoc scripts that were not persisted, so
this is a hypothesis, not a record. **Settle it before quoting #5 or #7's
absolute numbers again**: re-run the #5 comparison from the current script and
report both row subsets. Until then, #5 and #7 support only the *ordering*
(parametric loses to the MLP; near-stop is ~2.3x the aggregate), not their
magnitudes, and neither can be compared against #9's or #14's tables.

## Code added this investigation

- `src/rmind/components/dead_reckoning.py` — `dead_reckon_future_trajectory`,
  `gnss_anchor_drift_m`, ported verbatim from
  `feat/drivor:src/rmind/components/drivor/trajectory_target.py` (commit
  `920030a`). **Convention**: `heading` is a compass bearing (0°=north), so
  forward in the rotated ego frame is local **+y** (`cos(Δheading)`), lateral
  is **+x** (`sin(Δheading)`) — NOT the standard math-angle convention. Easy
  to get backwards; this was fixed once already on `feat/drivor` and
  rediscovered independently as a plotting footgun in
  `scripts/viz_policy_predictions.py:138-143`. Don't rederive, the tests in
  `tests/test_dead_reckoning.py` are the reference.
- `src/rmind/components/tick_trace.py` — `build_tick_table`,
  `horizon_windows`: undo the 11-tick window packaging into a per-drive tick
  trace, and enumerate anchors with a contiguous N-tick future (phase-aware,
  pause- and backwards-clock-gated). See finding #8.
- `src/rmind/components/controller.py` — dynamics models:
  - `TrajectoryToActionMLP`: `build_features(position, heading, speed)` →
    flattened 6-pose trajectory (18-dim) + current speed (1-dim) = 19-dim
    input; outputs `(mean, logvar)` per continuous field (gas/brake/steer)
    + turn-signal logits.
  - `LongitudinalDynamicsParams` / `fit_longitudinal_dynamics` /
    `invert_longitudinal_dynamics`: `dv ≈ gas_gain·gas − brake_gain·brake −
    drag_coeff·speed − offset`, OLS via `torch.linalg.lstsq`, analytically
    invertible for gas/brake given a target `dv`.
  - `LaggedLongitudinalDynamicsParams` / `fit_lagged_longitudinal_dynamics` /
    `invert_lagged_longitudinal_dynamics`: multi-tap FIR extension (gas/brake
    at t, t-1, t-2, ...). **Net negative finding, see below** — kept for
    reference/tests but not used in the winning configuration.
  - `LateralDynamicsParams` / `fit_lateral_dynamics` /
    `invert_lateral_dynamics`: bicycle-model style,
    `dheading ≈ steer_gain·(steering·speed) + camber_bias`.
  - `TrajectoryToActionMLP(action_steps=K)` emits the next K actions in one
    pass — MPC's control horizon, of which a receding-horizon loop applies
    index 0. `K=1` keeps the pre-sequence output *and* head weight shapes
    exactly, so old checkpoints still load. **Net negative for the applied
    action, see finding #11**; kept because the per-step breakdown it
    produces is the finding.
- `src/rmind/scripts/trajectory_action_controller.py` — CLI with `episode`,
  `stitch`, `horizon`, `nearstop`, `calibrate`, `gaincond`, `reckon`,
  `train`, `eval`, `modes` subcommands. `calibrate` measures the split-half
  reliability of every per-drive gain (finding #12) and `gaincond` feeds the
  fitted calibration to the MLP against permuted controls (finding #13);
  both share `drive_half_split()` / `_tick_pairs()` / `calibrate_drives()`,
  which apply the same phase- and clock-gating `horizon_windows` does (bug
  #3 reaches the calibration fits too). `horizon` sweeps the trajectory (prediction) and action
  (control) horizons on one shared anchor set and drive split — see findings
  #9 and #11; `nearstop` computes the near-stop information floor of finding
  #10. `read_ticks()` reads the derived tick parquet directly (start here);
  `drive_split()` is the shared whole-drive hold-out — anchors overlap in
  `horizon` of their `horizon + 1` ticks, so a row-level split leaks
  near-duplicates and reports nothing. `load_ticks()` /
  `stitch_horizon()` build the extended-horizon dataset (finding #8), and
  `count_anchors()` sizes a horizon without materializing its trajectory
  arrays — at horizon 150 those come to ~9GB by arithmetic (1.5M anchors x
  151 ticks x 8B x 4 arrays), for what is ultimately one integer. Both share
  the per-drive walk in `_iter_anchor_index()`. `dead_reckon()` (windowed)
  and `dead_reckon_stitched()` share one `_dead_reckon()` helper, so bug #1's
  float64-timestamp trap is guarded in exactly one place. `load()` supports
  `n_drives`/`seed` for reproducible drive subsampling (see bug #2 for a
  subtlety here).
- `scripts/action_clipping/` — the finding #14 probes. `clip_probe.py` asks
  where each action actually saturates, `steer_probe.py` the same for the
  speed-gated steering case, `floor_clip.py` recomputes #10's information
  floor as a function of clip level, and `clip_target_sweep.py` is the sweep
  proper: 6 target transforms
  (brake and gas clips, plus a low-speed steering clip) x 5 fit seeds on one
  fixed split, reporting raw L1 alongside *matched* metrics that clip
  prediction and truth alike so a clipped-target model and a raw-target model
  are scored on the same scale. Standalone rather than a subcommand because it
  only calls public pieces of `trajectory_action_controller` (`read_ticks`,
  `stitch_horizon`, `dead_reckon_stitched`, `drive_split`, `_fit`); promote it
  if target transforms turn into a real axis.
- Tests: `tests/test_dead_reckoning.py` (6), `tests/test_controller.py` (14),
  `tests/test_tick_trace.py` (16), `tests/test_calibration.py` (9). All 45
  passing at time of writing. The
  load-bearing one is
  `test_dead_reckon_prefix_matches_shorter_horizon`: the horizon sweep's
  entire validity rests on a long roll-out's prefix being bit-for-bit the
  short roll-out.
- Checkpoints on disk (not committed): `/tmp/controller_v3.pt` +
  `.json` sidecar is the older 6-tick MLP checkpoint (`n_drives=250, seed=7`,
  config asserts `n_rows` matches on load — use this one, not `v1`/`v2`,
  which predate bug #2's fix and can't be exactly reproduced anymore). The
  findings-#9/#11 sweeps write `/tmp/horizon_sweep_traj/` (h6/h15/h30 at
  K=1) and `/tmp/horizon_sweep_ctrl/` (h30 at K=1/3/6/15/30), each a
  `horizon_sweep.json` plus one checkpoint per configuration; both are
  full-file, `seed=7`. **`controller_h30k1.pt` is the best controller found
  so far.** Note these are NOT comparable row-for-row with `controller_v3`:
  the sweep uses all 655 drives, a horizon-30 anchor set, and its own split.

## Bugs found and fixed along the way

1. **Float32 timestamp precision**: Unix-epoch microsecond timestamps
   (~1.67e15) cast to float32 before dead-reckoning silently zero every `dt`
   (float32 can't represent sub-second differences at that magnitude) →
   whole dead-reckoned trajectory collapses to zero. Fix: keep
   `time_stamp_s` as float64 through the dead-reckoning call, cast the
   *output* to float32 afterward. Regression test:
   `test_dead_reckon_survives_float32_epoch_timestamps`.
2. **`polars.unique()` non-determinism**: row order isn't stable across
   calls, so `rng.choice()` on an un-sorted `.unique()` result silently broke
   `--seed` reproducibility (two calls with identical args gave only 132/300
   overlapping drives). Fix: `sorted()` the drive list before `rng.choice()`
   in `load()`. Any checkpoint trained before this fix (`controller.pt`,
   `controller_v2.pt`) can't have its exact split reconstructed — use
   `controller_v3.pt`.
3. **Non-increasing tick clock** (data defect, not fixed — gated around).
   11/655 drives contain one backwards timestamp step (worst −3463 ms); the
   dataset's own `max_frame_gap_ms` filter bounds only the *maximum* gap, so
   these reach the parquet and corrupt the `dt` that
   `dead_reckon_future_trajectory` integrates. 102 windowed rows (0.005%) are
   affected. `horizon_windows` requires `0 < dt <= max_gap_ms`; the windowed
   path has no such guard. Full detail and drive names in finding #8.
4. `fit_lagged_longitudinal_dynamics` requires `gas_lags` and `brake_lags` to
   have the same `num_lags` (implicit in how it slices the solved
   coefficient vector) — now raises `ValueError` if they don't, with a
   docstring note. Caught via a test bug (mismatched synthetic lag counts).
5. **`_fit` never seeded model initialization.** The model was constructed
   *before* the only `manual_seed` call, which seeded the minibatch sampler
   alone, so the weights came from the process-global RNG. A fit's result
   therefore depended on how many fits had run before it in the same
   process, and no single run was reproducible. Demonstrated while building
   finding #13: the *same* arm, at the same `seed`, on identical features and
   an identical split, gave gas L1 0.0465 in a four-arm script and 0.0453 in
   a five-arm one — the only difference being how many models had been
   initialized first. Fixed by `torch.manual_seed(seed)` at the top of
   `_fit`. See finding #13 for what this costs findings #9 and #11.

## Confirmed findings, in order

### 1. Vehicle-conditioning ablation hurts, doesn't help

Adding a learned one-hot vehicle embedding to the baseline MLP controller
made every field worse on held-out drives:

| field | baseline | +vehicle | Δ |
|---|---|---|---|
| gas_pedal | 0.0382 | 0.0412 | +8.0% (worse) |
| brake_pedal | 0.0182 | 0.0210 | +15.9% (worse) |
| steering_angle | 0.0188 | 0.0236 | +25.5% (worse) |
| turn_signal acc | 0.8146 | 0.8191 | +0.6% (better) |

Naive statistical patches (learned embedding, post-hoc additive bias
correction at vehicle or drive level) do **not** successfully exploit
whatever real per-vehicle/per-drive effect exists, on held-out data.

**Two caveats added in hindsight, neither of which overturns the finding.**
(a) This was a single fit per arm and predates bug #5, so the two arms did not
share an initialization — the deltas are unpaired. They are large enough
(+8.0% / +15.9% / +25.5% against a floor of gas 1.0% / brake 0.9% / steering
2.0%, #14) that the direction is safe, but the magnitudes are not. (b) The
absolute values here are the lowest anywhere in this document and are not
comparable to #9's or #13's — see the comparability note above. Finding #12
supplies the mechanism this finding could only observe: 69% of `gas_gain`'s
spread is between drives of the *same* vehicle, so a static per-vehicle
one-hot can address at most a third of the effect while paying the full
variance cost.

### 2. The "hidden speed modes" hypothesis is real — and it's drive-level, not just vehicle-level

Confirmed via three independent methods (matched-condition ANOVA on raw
pedal-vs-outcome, tightened outcome-matching showing *increasing* effect size
under tighter matching, and per-vehicle breakdown showing a broad ~20x spread
not driven by outliers). Nested variance decomposition shows drive-level
variance comparable in magnitude to vehicle-level variance — the effect
tracks the drive/session, not just which physical vehicle it is (consistent
with a driver-selectable or session-persistent "mode": regen calibration,
drive mode, etc.). See `cmd_modes` in the script and
`[[project_fd_permutation_importance]]`-adjacent reasoning. This motivated
finding #1 above (why a *global*, static vehicle embedding can't help — the
effect isn't a fixed per-vehicle property).

### 3. A genuine per-session parametric forward-dynamics model dramatically improves longitudinal control

Fitting `LongitudinalDynamicsParams` on the first half of a held-out drive
and evaluating (via analytic inversion) on the second half of the *same*
drive:

- Gas L1: 0.414 → **0.073** (naive/pooled baseline → per-drive calibrated),
  an 82% reduction.
- This validates the user's MPC-theory framing directly: same inputs
  (current state + target), same "car setup" (drive-specific calibration),
  same optimization (analytic OLS inversion) → consistent actions for
  consistent trajectories, unlike a pooled black-box model that implicitly
  averages over hidden per-drive modes.

### 4. The opposite pattern holds for steering: global (pooled) wins

`steer_gain` is essentially vehicle-invariant (0.0174–0.0194 for 22/24
vehicles) — personalizing only adds noise from smaller per-drive sample
sizes. Best steering config: **global pooled** dynamics fit, L1 = 0.031.
Two apparent outlier vehicles (Niro101-HQ, Niro131-HQ) were diagnosed as
heading-sensor corruption at speed (data-quality artifact, not real steering
difference), consistent with a previously-documented corruption pattern —
see `[[project_vehiclemotion_field_availability]]`-adjacent notes.

**Takeaway**: personalization level is field-specific. Longitudinal dynamics
(gas/brake) carry a real per-session "mode" worth calibrating for; lateral
dynamics (steering) don't — the mechanical steering ratio is shared across
the fleet. **Half-superseded by finding #12**: the per-session mode is real
for *gas* only. `brake_gain` turns out not to be estimable per drive at all,
so it belongs with steering in the do-not-personalize bucket — for the
opposite reason (steering is genuinely constant, brake is genuinely
unmeasurable here).

### 5. Even the best per-field parametric combo still loses to the plain MLP

Combining the *winning* variant per field (per-drive for gas/brake, global
for steering) against the plain `TrajectoryToActionMLP`:

| field | parametric (best variant) | MLP | Δ |
|---|---|---|---|
| gas_pedal | 0.073 | 0.060 | parametric +22% worse |
| brake_pedal | 0.081 | 0.035 | parametric +131% worse |
| steering_angle | 0.031 | 0.022 | parametric +41% worse |

Reason: the parametric models collapse the target to a single scalar
(`target_dv` or `target_dheading`), discarding the full 6-step trajectory
*shape* the MLP consumes directly. This gap turned out to be central to
finding #7 below (it's not a fixed "MLP wins" story — it's specifically a
missing-information story).

⚠️ **The MLP column here is not comparable to #9, #13 or #14** — it is ~1.7x
worse than finding #1's baseline on the same stated split, most likely because
this comparison scores only the second halves of val drives (the parametric arm
needs the first half to calibrate). See the comparability note under "Data and
reproducibility". Take the *ordering* from this table, not the values.

### 6. Adding pedal *history* (past lag taps) makes the dynamics model worse, not better — twice-confirmed, understood mechanism

User's hypothesis: gas/brake/speed changes lag behind pedal input by more
than one tick (inertia), so a richer per-drive dynamics model should
condition on `gas[t], gas[t-1], gas[t-2], ...`. Tested two ways, both
negative:

- **Multi-tap linear regression** (`LaggedLongitudinalDynamicsParams`,
  num_lags 1→2→3): gas L1 0.074→0.089→0.099 (monotonically worse); brake L1
  0.081→0.079→0.086 (flat/worse).
- **Single-time-constant EMA smoothing** (`effective_gas[t] = α·gas[t] +
  (1−α)·effective_gas[t-1]`, sidesteps multicollinearity entirely — one
  scalar knob, not N correlated coefficients): α=1.0 (no smoothing) → 0.074;
  α=0.7 → 0.075; α=0.5 → 0.077; α=0.3 → 0.081; α=0.15 → 0.087. Monotonically
  worse as smoothing increases, at *every* tested value.

**Mechanism**: `gas[t]`, `gas[t-1]`, `gas[t-2]` are highly autocorrelated in
real driving (r=0.94–0.98 for lags 1–2). More importantly, the target here is
Δspeed over a single ~0.333s tick — at that timescale `gas[t]` combined with
`speed[t]` (which already reflects the accumulated effect of prior pedal
history, since it's a Markov state) already captures essentially all the
predictive signal; adding `gas[t-1]` reintroduces a noisier, less relevant
estimate of the same already-encoded state rather than new information. This
does **not** contradict the physical reality of inertia — a separate,
earlier sustained-throttle test in this investigation showed the "speed
mode" effect keeps building over the full ~2s horizon (η² 0.156→0.143,
slowly decaying, not gone) — inertia shows up as **multi-tick drift**, not as
a same-tick lag between input and output. A 1-tick-ahead `dv` model is
already close to as fast as this system's response gets.

**Implication for future work**: don't add past-pedal lag taps to
single-tick dynamics models — it's a net loss. If inertia matters for a
task, model it via the *horizon length of the outcome being tracked*, not
via extra lagged inputs on the action side.

### 7. Near-stop / "idling" is not a trivial slice — it's the single hardest regime, and it's a genuine identifiability problem, not a modeling-power problem

Stratifying by speed band (`SPEED_BANDS = ((0,5),(5,10),(10,20),(20,35),
(35,60),(60,130))`) on the same val split:

- **0–5 km/h is ~15% of val rows** (12,878 / 88,326) — not negligible.
- It's the **worst band for brake** in both formulations tested:

  | model | brake L1, 0–5 km/h | brake L1, all bands | ratio |
  |---|---|---|---|
  | per-drive parametric (scalar `target_dv`) | 0.193 | 0.081 | 2.4x |
  | trajectory MLP (full 6-tick/1.667s horizon) | 0.080 | 0.035 | 2.3x |

**Root cause, directly measured**: for rows with `speed_now < 5 km/h` and
`brake_now > 0.3`, 99.3% of the time the car is already within 0.3 km/h of
zero speed on the *next* tick, and `dv` there correlates 0.955 with
`-speed_now` — **not** with brake pressure at all. Once nearly stopped, `dv`
is clamped near zero regardless of how hard the brake is pressed, so
`target_dv → brake` inversion from a scalar target is fundamentally
unidentifiable in this regime: brake=0.2 and brake=0.8 both produce "stayed
stopped," and no amount of dynamics-model sophistication (more lags, more
per-drive calibration) can resolve that from `(state, scalar target_dv)`
alone — it's a missing-information problem.

**This is why the MLP does so much better in this band**: it consumes the
full 6-tick future trajectory shape (whether the car stays stopped vs. is
about to creep/launch), which is exactly the information that disambiguates
otherwise-identical brake pressures at `v≈0`. This is direct evidence that
*more future horizon* (not more past lag — see #6, the opposite direction)
helps this specific failure mode.

**Caveat / open question**: even the MLP retains the same *relative*
penalty in this band (2.3x vs. aggregate) as the parametric model (2.4x),
meaning real ambiguity survives the current 1.667s horizon (e.g. a red light
longer than that gives no signal about when it'll change). ~~Testing a longer
horizon requires new data.~~ **Superseded by finding #8** — it does not.

~~**Prediction**: a longer horizon should help the 0–5 km/h brake band
most.~~ **Falsified by findings #9 and #10.** The horizon extension was run;
0–5 km/h is the one brake band that does *not* improve (+2.3% at 10s), and
#10 shows why — half those rows never move at all over the whole horizon, so
their trajectory is the same zero trajectory at any length, and the trained
controllers already sit on the resulting information floor. The diagnosis
below (missing information) survives; the inference that the missing
information lies further ahead *in the ego trajectory* does not.

### 8. The 11-tick window is packaging, not a horizon limit — longer horizons need no new data

The predict dataset is built with `episode_stride == episode_step == 10`
(`config/_templates/dataset/yaak/train.yaml`, resolved in the run's
`.hydra/config.yaml` to `every: 10i` / `period: 110i` / `gather_every: 10`),
so **consecutive windows advance by exactly one tick** and overlap by 10 of
their 11 ticks. The parquet therefore already tiles every drive at 0.333s
resolution. Since the trajectory GT is *dead-reckoned* from `speed` +
`headings_denoised/heading` — raw per-tick columns, not model outputs — the
horizon is a free parameter once the packaging is undone. No re-run of
predict, no GPU, no checkpoint.

Implemented as `rmind.components.tick_trace` (`build_tick_table`,
`horizon_windows`) plus a `stitch` subcommand; `tests/test_tick_trace.py`
covers it (9 tests).

**The stitch is lossless and exact.** Overlapping windows agree on every
value column at every shared tick (8 drives, 167k tick-observations, 8.7x
redundancy, zero disagreements), and `stitch --verify` reproduces the
windowed `dead_reckon()` at `horizon=6` bit-for-bit over the whole file —
1,971,276/1,971,336 windowed rows matched, max |Δ| position `0.000e+00`,
heading `1.49e-08` (float32 noise).

**The artifact**:
`outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.ticks.parquet`
— 2,087,541 rows × 10 flat columns (`input_id`, `frame_idx`, `time_stamp`,
`speed`, `heading`, `gnss_xy`, the three pedals, `turn_signal`), **40 MB**
vs. the 516 MB source. Built in 3.5s; regenerate with `stitch --out`. Feed it
straight to `stitch_horizon(ticks, horizon=N)` — `load_ticks()` is only
needed to rebuild it from the source parquet.

Anchors surviving with a fully contiguous future (full file, phase-aware,
pause-gated):

| horizon | seconds | anchors | % of 2,087,541 ticks |
|---|---|---|---|
| 6 (today) | 2.0 | 2,036,353 | 97.5% |
| 15 | 5.0 | 1,965,816 | 94.2% |
| 30 | 10.0 | 1,878,874 | 90.0% |
| 60 | 20.0 | 1,753,309 | 84.0% |
| 90 | 30.0 | 1,654,155 | 79.2% |
| 150 | 50.0 | 1,496,961 | 71.7% |

**Two things that will bite anyone reimplementing this:**

1. **Phase grids.** Filtering drops raw frames, which shifts a clip's start
   off the `every: 10i` grid, so one drive can carry several interleaved
   phase grids (1.67 on average, up to 8). A phase-blind sort sees diffs of
   3/7 where the data is perfectly contiguous within its own phase, and
   costs ~70% of the mean run length (74.7 vs 245.2 ticks). `horizon_windows`
   walks each phase separately.
2. **The tick clock can run backwards.** `Niro111-HQ/2023-03-20--10-49-39`
   steps −369 ms at one tick; 11/655 drives have one such step each (worst:
   `Niro101-HQ/2023-01-01--12-01-47`, −3463 ms). 102/1,971,336 windowed rows
   (0.005%) contain such a step somewhere in their 11 ticks; for 60 of them it
   falls inside the scored `FEAT_IDX..FEAT_IDX+6` span, and those 60 are
   exactly the rows `stitch --verify` cannot match. The dataset's own
   `max_frame_gap_ms` gate only bounds the *maximum* gap, so a negative `dt`
   passes it — and `dt` is integrated directly by
   `dead_reckon_future_trajectory`, stepping the trajectory backwards in time.
   `horizon_windows` requires `0 < dt <= max_gap_ms`.

**Where the extended GT stops being trustworthy.** Dead-reckoning error
compounds, so `gnss_anchor_drift_m` matters far more at 10s than at 1.667s.
At `horizon=30`, drift against the raw GNSS anchor versus the distance
actually travelled over those 10s — all 1,878,874 anchors, whole file:

| speed band (km/h) | n | median drift (m) | p90 (m) | median path (m) | drift % | p90 % |
|---|---|---|---|---|---|---|
| 0–5 | 285,348 | 0.50 | 2.14 | 19.3 | 7.2 | 22.7 |
| 5–10 | 76,254 | 1.13 | 3.96 | 25.6 | 6.4 | 18.1 |
| 10–20 | 206,191 | 1.47 | 4.80 | 45.6 | 4.9 | 10.1 |
| 20–35 | 387,498 | 2.31 | 6.00 | 72.6 | 3.5 | 8.4 |
| 35–60 | 506,386 | 3.98 | 8.94 | 122.6 | 3.4 | 7.0 |
| 60–130 | 417,197 | 7.70 | 15.34 | 216.2 | 3.4 | 6.5 |

In absolute meters the drift grows with speed (7.70m median in the 60–130
band), but **relative** drift is a flat 3.4–4.9% from 10 km/h up, so the
10-second target holds across the operating range. The near-stop band — the
one finding #7 says needs the extra horizon — has by far the *smallest*
absolute error (0.50m median, 2.14m p90). Its higher relative figure (7.2%)
is an artifact of dividing by a ~19m path, and 0.50m is well inside the
ambiguity the horizon is meant to resolve (stopped vs. creeping vs.
launching). **So the horizon-extension experiment is viable exactly where it
matters.**

Reproduce this table from the tick parquet alone — no source parquet, ~1 min:

```python
ticks = pl.read_parquet(".../model-4vqatiom:v4.ticks.parquet")
a = stitch_horizon(ticks, horizon=30)
position, _ = dead_reckon_stitched(a)
drift = gnss_anchor_drift_m(
    dead_reckoned_position_normalized=position,
    gnss_xy=torch.from_numpy(a["gnss_xy"]).float(),
    heading_deg=torch.from_numpy(a["heading"]).float(),
    reference_index=0,
).numpy()
dt_s = np.diff(a["time_stamp"].astype(float), axis=1) / 1e6
path_m = (a["speed"][:, :-1] / 3.6 * dt_s).sum(axis=1)   # denominator
```

The one thing stitching cannot extend is `policy/trajectory_value`: it is
`Array(Float32, shape=(6, 2))`, baked into the checkpoint. Asking whether a
*trajectory-predicting* model wants a longer horizon still needs retraining.

### 9. The horizon extension works — but nowhere near where finding #7 predicted

The experiment finding #8 unblocked, run at full scale: `horizon` subcommand,
all 655 drives, 1,878,874 anchors, 589 train / 66 val drives (1,701,395 /
177,479 rows), `action_steps=1`, 20k steps, seed 7. Every horizon trains and
validates on **identical rows** — the anchors are enumerated once at horizon
30 and the shorter trajectories are sliced out of the same dead-reckoned
roll-out, which is exact (`test_dead_reckon_prefix_matches_shorter_horizon`).
Re-enumerating per horizon would instead hand the longer horizons a strictly
more contiguous subset of each drive and confound the result with
survivorship.

Aggregate val L1 on the applied action:

| field | h6 (2.0s) | h15 (5.0s) | h30 (10.0s) | h30 vs h6 |
|---|---|---|---|---|
| gas_pedal | 0.0435 | 0.0450 | 0.0432 | **−0.7%** |
| brake_pedal | 0.0251 | 0.0230 | 0.0226 | **−9.9%** |
| steering_angle | 0.0212 | 0.0213 | 0.0214 | +0.8% |
| turn_signal acc | 0.8118 | 0.8269 | 0.8424 | **+3.8%** |

**Finding #7's prediction is falsified.** It predicted the 0–5 km/h brake
band would improve most. It is the *only* brake band that doesn't improve at
all:

| speed band (km/h) | brake h6 | brake h15 | brake h30 | Δ |
|---|---|---|---|---|
| 0–5 | 0.0677 | 0.0696 | 0.0693 | **+2.3%** |
| 5–10 | 0.0349 | 0.0278 | 0.0265 | −24.3% |
| 10–20 | 0.0293 | 0.0244 | 0.0221 | −24.6% |
| 20–35 | 0.0227 | 0.0189 | 0.0180 | −20.6% |
| 35–60 | 0.0148 | 0.0122 | 0.0124 | −16.3% |
| 60–130 | 0.0105 | 0.0102 | 0.0103 | −2.0% |

The entire gain lives in **5–35 km/h**, the approach to a stop. That is the
regime where "a stop is coming within 10s" is real, actionable, and invisible
at 1.667s — and where the answer is a brake *modulation*, which is what the
extra horizon supplies. Above ~35 km/h the gain fades (10s of trajectory at
100 km/h is 280m of mostly-straight road, which says little about the pedal).
The near-stop penalty ratio does not shrink; it widens (2.7x → 3.1x).

Turn signal is the other clear winner (+3.8pp, monotone in horizon), and for
the obvious reason: signalling is an intent declared seconds before the
manoeuvre, so a 10s path shows the turn that a 1.667s path does not.

Steering is flat at every horizon — consistent with finding #4 (lateral
dynamics are near-memoryless and fleet-invariant; the immediate path curvature
is already sufficient). The +0.8% in the table is inside the 2.0% steering
noise floor #14 measures, so "flat" is the whole of it.

### 10. Near-stop braking is at its information floor — it was never a horizon problem

Finding #7's *diagnosis* (missing information) was right; its *prescription*
(more future horizon) was wrong, and finding #9 shows why empirically. The
`nearstop` subcommand shows the mechanism directly.

Of the 25,085 held-out near-stop rows (14.1% of val), **50.6% (12,685) travel
less than 0.5 m over the entire 10-second horizon.** For those rows the future
trajectory is the zero trajectory — *identical no matter how far ahead it is
drawn*. Lengthening the horizon adds literally zero bits about them.

That makes the achievable L1 computable. Binning those rows by current speed
(0.5 km/h bins) and taking each bin's median — the L1-optimal constant, fit
as an oracle on the evaluation rows themselves — is the best any predictor
conditioning on (zero trajectory, current speed) can do:

| field | oracle floor | h6 | h15 | h30 |
|---|---|---|---|---|
| gas_pedal | 0.0000 | 0.0001 | 0.0000 | 0.0000 |
| brake_pedal | **0.0815** | 0.0804 | 0.0809 | 0.0807 |
| steering_angle | 0.0654 | 0.0703 | 0.0699 | 0.0708 |

**All three controllers sit on the brake floor at every horizon** (the ~1%
they come in under it is not a violation: the oracle is piecewise-constant on
hard 0.5 km/h bins, while the model sees continuous speed and interpolates
across bin edges). Target brake in this group has median 0.2744 and p90
0.4207 — a genuinely wide distribution that nothing in the ego's own future
path distinguishes. A stopped car is stopped whether the driver is holding
0.2 or 0.5.

On the complementary near-stop rows that *do* move, the trajectory earns its
keep — the models beat the same speed-only oracle comfortably (which is
expected there; the oracle is a baseline, not a floor, once the trajectory
carries information):

| field | speed-only baseline | h6 | h15 | h30 |
|---|---|---|---|---|
| gas_pedal | 0.0313 | 0.0167 | 0.0163 | **0.0152** |
| brake_pedal | 0.0882 | **0.0547** | 0.0581 | 0.0577 |
| steering_angle | 0.0999 | **0.0946** | 0.0970 | 0.1002 |

**Conclusion**: roughly half the near-stop band is irreducible from ego
trajectory alone, and that half sets the band's floor. Resolving it requires
information the ego's future path does not contain at any horizon —
exteroception (the camera stream showing the light or the lead vehicle's brake
lights), or the pedal history the model is not given. Do not spend more effort
extending the horizon for this; #6 closed the past-lag direction and #9/#10
close the future-horizon one. **Stop treating near-stop brake L1 as a
tractable modelling target and report it separately from the bands that are.**

### 11. Emitting an action *sequence* (MPC's control horizon) hurts the action you actually apply

**Read the step index carefully — it is offset by one from "now".** Step *k*
is the action at `anchor + 1 + k`, because `stitch_horizon` puts the first
action target at `anchor+1` to match the windowed `GT_IDX`
(`trajectory_action_controller.py:333-336`, and `action_index = index[:,
1:action_steps+1]`). The anchor itself is `FEAT_IDX`, "now", and its own
action is never a target in any configuration here. So **step 0 already is
the one-tick-ahead action**, not the action at the anchor — which is what
makes the setup latency-correct out of the box; see the latency note below.

`TrajectoryToActionMLP` now takes `action_steps`: it emits the next K actions
in one pass — an open-loop control sequence over the plan, of which a
receding-horizon controller applies index 0 and re-solves. Swept at trajectory
horizon 30, same anchors and split, K ∈ {1, 3, 6, 15, 30}, val L1 on **step 0**
(the applied action):

| K (control horizon) | gas | brake | steering | turn acc |
|---|---|---|---|---|
| 1 | **0.0433** | **0.0224** | 0.0207 | 0.8421 |
| 3 | 0.0436 | 0.0229 | 0.0208 | **0.8426** |
| 6 | 0.0447 | 0.0234 | **0.0198** | 0.8420 |
| 15 | 0.0466 | 0.0247 | 0.0198 | 0.8402 |
| 30 | 0.0516 | 0.0273 | 0.0223 | 0.8409 |

Gas and brake degrade monotonically (+19% / +22% at K=30). Steering is the one
exception, gaining ~4% at K=6–15 before degrading — but that ~4% is barely 2x
the 2.0% steering noise floor #14 later measured, from a single seed, so treat
it as unconfirmed. Mechanism: a fixed-capacity
128-wide shared trunk now serves 30x the outputs, and the extra steps carry no
information about step 0 — plain multi-task interference. **Keep `K=1` for the
applied action.**

The per-step breakdown is the interesting part, and it is *not* monotone —
step 0 is the hardest step in the sequence, not the easiest (K=6, trajectory
horizon 30):

| step ahead | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| tick, from "now" | +1 | +2 | +3 | +4 | +5 | +6 |
| gas | 0.0455 | 0.0437 | **0.0434** | 0.0439 | 0.0450 | 0.0464 |
| brake | 0.0239 | 0.0232 | **0.0228** | 0.0230 | 0.0234 | 0.0243 |
| steering | 0.0217 | 0.0200 | 0.0193 | 0.0191 | **0.0191** | 0.0194 |

The trajectory predicts the action ~1 second out better than it predicts the
action one tick out. This is the inverse-dynamics structure the repo already
exploits in `InverseDynamicsPredictionObjective` (actions from *surrounding*
observations): the anchor's trajectory brackets a mid-sequence action with
several poses on each side, so that action is interpolated, whereas step 0 has
exactly one trajectory segment before it and is effectively extrapolated off
the anchor. It also means the immediate pedal carries driver jitter that a
smooth path cannot resolve.

**Consequence for a deployed controller**: the action a receding-horizon loop
must apply is structurally the one this model predicts worst, and no amount of
control horizon fixes that — it is a property of where the action sits
relative to the plan.

#### Latency note: the applied action is step 0, and that is already correct

A controller that spends one tick (0.333s) computing cannot apply its output
at the anchor — by the time the action exists the car is at `anchor+1`. It is
therefore worth stating explicitly that **this pipeline is already aligned
for exactly one tick of end-to-end latency**: the features are the trajectory
and speed at the anchor, and the `K=1` target is the action at `anchor+1`.
The alignment is inherited from the windowed `FEAT_IDX`/`GT_IDX` convention
rather than deliberately chosen, but it is the right one, and
`controller_h30k1.pt` needs no retargeting. Recommendation #1 stands as
written.

This means the pessimistic paragraph above *survives* the latency argument
rather than being excused by it: step 0 is both the hardest step in the
sequence and the step a 1-tick-latency controller applies. The
one-sided-context mechanism is why — `build_features` starts the trajectory
at the anchor, so step 0's action has visible path on one side only, whereas
step 2's is bracketed before and after.

**Where the latency argument does bite: end-to-end latency above one tick.**
Add actuation delay, or run the loop slower than 3 Hz, and the applied index
moves to step 1 or beyond — onto the favourable side of the non-monotonicity:

| field | step 0 (+1 tick) | step 1 (+2 ticks) | Δ |
|---|---|---|---|
| gas | 0.0455 | 0.0437 | **−4.0%** |
| brake | 0.0239 | 0.0232 | **−2.9%** |
| steering | 0.0217 | 0.0200 | **−7.8%** |

So the per-step table doubles as a **latency-compensation curve**: measure
the real end-to-end delay in ticks and it gives both the index to apply and
what that index costs. **Within a fixed `K=6` head**, two ticks of latency is
cheaper to serve than one — latency buys a better-posed target. That is a
within-model comparison and does not survive as stated across the two axes:
the crossing table below has `K=6`/step 1 at gas 0.0437 and brake 0.0232
against `K=1`/step 0's 0.0433 and 0.0224, i.e. step 1's bracketing gain does
not pay back `K=6`'s capacity cost. The gain is only bankable if step 1 is
served by a *retargeted `K=1`* head, which nothing here measures.

Serving step 1 needs `K >= 2`, which reopens the capacity tradeoff above, and
the sweep only crossed the two axes at `K=6`. That crossing is a wash:

| | `K=1`, step 0 | `K=6`, step 1 |
|---|---|---|
| gas | **0.0433** | 0.0437 |
| brake | **0.0224** | 0.0232 |
| steering | 0.0207 | **0.0198** |
| turn acc | **0.8421** | 0.8420 |

`K=6`'s multi-task interference and step 1's bracketing gain are the same
order of magnitude and cancel — gas and brake land slightly *worse* than
`K=1` at step 0, steering slightly better. So a 2-tick-latency controller
should not just take `K=6` and apply index 1. The configuration to run is
`K=1` with the target shifted to `anchor+2` — a one-line change to
`action_index`'s slice start — which takes the bracketing gain without paying
the interference cost. Untested: nothing in either sweep measures a
single-step head at a non-unit offset. **Only worth running if the measured
latency actually exceeds one tick**; at one tick the current baseline is
already the right model.

### 12. Only `gas_gain` is estimable per drive — `brake_gain` and `steer_gain` are not

Finding #3 showed a per-drive `LongitudinalDynamicsParams` fit transforms the
*parametric* controller, and finding #2 located the effect at the
drive/session level rather than the vehicle. The obvious next move is to
encode each drive by its fitted gains and hand that to a model. Before asking
whether a model can *use* such a code, the `calibrate` subcommand asks
whether the code is **there**: fit each drive's first and second half
independently and compare the between-drive spread against the half-to-half
noise. This bounds every conditioning scheme at once — no architecture can
exploit a number that does not survive re-estimation.

All 644 drives with two usable halves, whole file. `reliability` is
`(var_between − var_within) / var_between` with `var_within = var(a − b) / 2`
(the per-half noise variance, assuming both halves are equally noisy), and
`eta2(vehicle)` is the share of the surviving spread that is between vehicles
rather than between drives of the same vehicle:

| estimator | n | p5 / p50 / p95 | half-half r | reliability | η²(vehicle) |
|---|---|---|---|---|---|
| **gas_gain** / all rows | 619 | 3.28 / 5.81 / 8.19 | **+0.600** | **0.500** | 0.345 |
| brake_gain / all rows | 620 | −0.11 / 2.15 / 6.95 | +0.194 | 0.000 | 0.170 |
| brake_gain / v>5 | 602 | 3.57 / 6.05 / 8.15 | +0.216 | 0.000 | 0.219 |
| brake_gain / braking only | 182 | 5.75 / 8.02 / 10.48 | +0.248 | 0.000 | 0.219 |
| brake_gain / hard braking | 85 | 6.10 / 9.27 / 12.24 | +0.176 | 0.000 | 0.163 |
| steer_gain / all rows | 620 | 0.0150 / 0.0186 / 0.0200 | +0.018 | 0.000 | 0.052 |
| steer_gain / v>5 | 604 | 0.0156 / 0.0186 / 0.0200 | +0.047 | 0.000 | 0.041 |
| steer_gain / v>5, turning | 293 | 0.0166 / 0.0188 / 0.0198 | +0.207 | 0.000 | 0.122 |

**`gas_gain` is real**: r = +0.600, and roughly half its drive-to-drive
spread survives re-estimation. It is also the direct explanation for finding
#1 — η²(vehicle) = 0.345 means **69% of the spread is between drives of the
same vehicle**, so a static per-vehicle one-hot can address at most a third
of the effect while paying the full variance cost of the extra parameters.

**`brake_gain` is not estimable, because the braking happens where `dv` is
clamped.** The median drive spends 18.6% of its calibration-half ticks with
`brake > 0.05`, but only **6.8%** with `brake > 0.05` *and* `v > 5 km/h` —
most braking in this data sits in exactly the near-stop regime where finding
#10 showed `dv` carries no information about pedal pressure. Tightening the
gate to isolate real braking makes the estimate scarcer without making it
sharper: 620 → 182 → 85 fittable drives, `r` stuck at +0.18–0.25,
reliability 0.000 throughout. This is finding #10's identifiability wall
seen from the *calibration* side rather than the control side. (The last two
rows use `_brake_only_gain`, which drops the gas column: gating on
`brake > 0.05` drives that column to nearly all zeros, making the joint
4-column design rank-deficient, and `lstsq`'s least-norm solution then
returns a `brake_gain` that is not one.)

**`steer_gain` is estimable but constant**, so there is nothing per-drive to
encode: p5–p95 is 0.0150–0.0200 (±13% around the median) and
η²(vehicle) = 0.052 — 95% of even that tiny spread is within-vehicle noise.
This is the quantitative form of finding #4: the mechanical steering ratio is
shared across the fleet, so a per-drive code can only inject noise. The one
gate that lifts `r` (v>5 and actually turning, +0.207) also halves `n` and
still has reliability 0.

**Consequence**: of the four `LongitudinalDynamicsParams` coefficients plus
`steer_gain`, exactly one is worth conditioning on. `drag_coeff` (r = +0.316)
and `offset` (r = +0.360) are likewise at reliability 0.00–0.06 — their
apparent correlation is shared drive-level structure, not a reproducible
per-drive value. Finding #13 tests what the one good scalar actually buys —
and finds that the scalar buys almost nothing while the full four-coefficient
vector, individually unreliable coefficients included, buys a lot.

### 13. Per-drive calibration as an MLP input: the `gas_gain` scalar is a wash, the full 4-parameter vector is a −12.5% gas win

The natural follow-up to #12: hand the per-drive calibration to the MLP as
extra input features and see whether it does for the learned controller what
finding #3 showed it does for the parametric one. `gaincond` subcommand,
whole file, horizon 30, `--steps 20000 --fit-seeds 7 11 13`, `batch_size=1024`.

Protocol — **calibrate on each drive's first half, train and score only on
anchors in its second half**, whole drives held out. This is causally valid
(past → future within a session) and leak-free. 644/655 drives calibrate
(`gas_gain` p5/p50/p95 = 3.182 / 5.807 / 8.349); 932,007 of 1,878,874 anchors
survive the second-half + calibrated-drive gate; 580 train / 64 val drives
(839,249 / 92,758 rows). Every arm sees **identical rows** — only the feature
vector differs. The parameters are robust-z standardized across drives.

**The shuffled arms are the load-bearing control**: the same scalars,
permuted across drives. An arm that beats `base` by no more than its own
shuffled twin has measured capacity or noise, not calibration. Each arm is
refit under all three seeds because a single fit's spread is the same order
as the effect (bug #5).

| arm | gas | brake | steering | turn acc |
|---|---|---|---|---|
| base | 0.0444±0.0005 | 0.0200±0.0013 | 0.0191±0.0001 | 0.8428±0.0021 |
| `gasgain` (1-vec) | 0.0438±0.0003 (−1.4%) | 0.0215±0.0014 (+7.3%) | +1.8% | −0.04pp |
| `gasgain_shuf` | 0.0452±0.0008 (+1.8%) | 0.0201±0.0005 | +3.8% | −0.04pp |
| **`allparams` (4-vec)** | **0.0389±0.0007 (−12.5%)** | 0.0262±0.0100 (+30.9%) | +4.5% | −0.10pp |
| `allparams_shuf` | 0.0449±0.0003 (+1.1%) | 0.0210±0.0018 (+5.1%) | +4.6% | −0.67pp |

**`gas_gain` alone is not worth a feature.** −1.4% against `base`, ~3% against
its own shuffled twin, both the size of the seed sd. That is exactly what
#12's reliability of 0.500 buys: real, but too thin to bank.

**The full 4-vector** (`gas_gain, brake_gain, drag_coeff, offset`) is a
different result: per-seed gas range 0.0380–0.0396 versus base 0.0438–0.0450,
**non-overlapping**, and its shuffled control lands at +1.1%, so the gain is
the drive-specific *values* and not the four extra input dimensions. The gain
is concentrated above 10 km/h — i.e. where gas actually matters (the 0–5
band's gas L1 is 0.007 to begin with, so its −3.4% is nothing in absolute
terms):

| gas L1 by speed band (km/h) | 0–5 | 5–10 | 10–20 | 20–35 | 35–60 | 60–130 |
|---|---|---|---|---|---|---|
| base | 0.0070 | 0.0299 | 0.0334 | 0.0371 | 0.0489 | 0.0673 |
| `allparams` | 0.0068 | 0.0298 | 0.0311 | 0.0338 | 0.0415 | 0.0582 |
| Δ | −3.4% | −0.2% | −6.8% | −9.0% | **−15.1%** | **−13.5%** |
| `allparams_shuf` | 0.0074 | 0.0320 | 0.0355 | 0.0390 | 0.0496 | 0.0660 |

**Gas head only — the calibration hurts brake wherever it is added.** Every
conditioned arm is worse on brake than `base`, and `allparams`' brake sd is
0.0100: seed 7 blew up to 0.0403 while s11/s13 sat at ~0.019. That one
arm-seed is also what makes its 60–130 brake cell read 0.0368 against a base
of 0.0123. Nothing about brake is supportable from this run; if the vector is
adopted, gate it into the gas head (or re-run brake with more seeds first).
Steering is uniformly slightly worse (+1.8 to +4.6%), consistent with #12's
`steer_gain` being constant — and note `*_shuf` is worse than the
corresponding real arm on steering too, so part of that is just capacity.

**This sharpens #12 rather than contradicting it, but the mechanism is now an
open question.** #12 measured each coefficient's split-half reliability in
isolation and found only `gas_gain` survives (`drag_coeff` r = +0.316,
`offset` r = +0.360, both at reliability 0.00–0.06). Yet the *vector* is what
works and the scalar barely is. So the vector is carrying drive-level
signature that no single coefficient survives re-estimation as. **The obvious
alternative explanation is not yet excluded**: the fitted vector may be
largely a proxy for the drive's operating regime (speed distribution, road
type) rather than for car setup. If so, plain drive-level speed statistics
would buy the same thing without a dynamics fit at all. Cheap to settle —
correlate the standardized vector against per-drive speed/accel summary stats
straight off the tick parquet (~90s), and re-run one arm with those stats
substituted for the gains.

**Deployment caveat for anything MPC-shaped.** The protocol calibrates on the
drive's first half, so the controller is *uncalibrated for the first half of
every session*. Making this deployable means recursive estimation — RLS on
the same 4-parameter model, seeded with a fleet prior — plus a measurement of
how fast the estimate converges as a function of elapsed driving. Neither is
tested here.

Artifacts: `/tmp/gasgain/gaincond.json` (per-seed, per-band), `gaincond.log`.
**Not comparable to finding #9's table** — a different anchor set (932k
second-half anchors vs 1.88M), a different split, and `batch_size=1024` vs
4096. `base` at 0.0444 against #9's h30 at 0.0432 is a different dataset, not
a regression.

### 14. Clipping a *target* is worse than clipping the *output* — the near-stop brake ambiguity cannot be defined away

If half the near-stop band is irreducible because brake=0.2 and brake=0.8 both
mean "stayed stopped" (#10), an obvious move is to stop asking for the
distinction: clip the brake target, on the theory that holding a stopped car
only needs *enough* pedal, so the discarded range is control-irrelevant label
noise. Tested directly, and it is a net loss — including in the band it was
built for.

`scripts/action_clipping/clip_target_sweep.py`, horizon 30, whole file, one
fixed split (seed 7, identical rows for every arm), `batch_size=4096`, 20k
steps, refit
under 5 training seeds (7, 11, 23, 42, 101). Within a seed, the baseline and
every variant share initialization and minibatch order, so each comparison is
**paired** and the seed effect cancels (bug #5).

**The premise checks out on the metric side, which is what makes the negative
result interesting.** Brake genuinely saturates where #10 said: at 0–5 km/h the
brake→Δv slope is ~0 at *every* pedal level, and `brake > 0.3` is 24.8% of
stationary rows against <1% of every moving band. Rescoring #10's near-stop
floor with the target clipped therefore removes a large chunk of it —
**0.0815 → 0.0471 at c=0.3 (−42%)**, and → 0.0747 at c=0.4 (−8.4%). That
part of the floor was never resolvable error; it was label noise being
counted as error. Clipping is a legitimate *metric* fix.

It is not a *training* fix — which is what the sweep below shows.

The comparison only means something on a **matched metric**: clip both the
prediction *and* the truth at `c`, so a raw-target model and a clipped-target
model are scored on the same scale. Scoring a clipped-target model on its own
clipped metric against a baseline on the unclipped one measures the metric,
not the model.

| target transform | matched at own clip level | `brake@0.3` (common scale) | raw brake L1 | paired Δ on `brake@0.3`, per seed |
|---|---|---|---|---|
| **none (baseline)** | — | **0.01917** | **0.02253** | — |
| `brake<=0.4` | 0.02233 (+2.0%) | 0.01960 (+2.2%) | 0.02296 (+1.9%) | +2.1 / +4.7 / +2.4 / −0.4 / +2.6 |
| `brake<=0.3` | 0.02012 (+4.9%) | 0.02012 (+4.9%) | 0.02344 (+4.0%) | +5.2 / +8.1 / +1.0 / +3.2 / +7.2 |
| `brake<=0.2` | 0.01624 (+13.5%) | 0.02417 (+26.1%) | 0.02748 (+22.0%) | +28.7 / +25.3 / +27.9 / +24.7 / +23.7 |

**Training on the clipped target loses to training on the raw target and
clipping at inference — at every clip level, on the matched metric, and even
when scored at the very clip level it was trained for** (the first column,
where the baseline is scored identically: `brake<=0.2` is +13.5% against the
baseline's own 0.01430 at `brake@0.2`). 5/5 seeds worse at `<=0.3` and
`<=0.2`; 4/5 at `<=0.4`. It also bleeds into heads it never touched
(`brake<=0.4`: steering +2.2%; every arm: turn −0.1pp).

**And it is worst exactly where it was supposed to help.** Raw brake L1 in the
0–5 km/h band — the #10 ambiguity band the clip was designed to remove:

| 0–5 km/h brake L1 | baseline | ≤0.4 | ≤0.3 | ≤0.2 |
|---|---|---|---|---|
| | **0.0694** | 0.0698 | 0.0722 | 0.0981 |

**Mechanism**: at c=0.3 the clip piles **38% of stationary rows onto an atom
at exactly c**, and the continuous heads are Gaussian `(mean, logvar)` — a
point mass is the one shape a Gaussian cannot fit, so the head does worse on
the clipped target than it does on the smooth raw target *and then gets
clipped afterwards*. Clipping also flattens the gradient that calibrates the
head's output scale. `brake<=0.2` is catastrophic for a second, simpler
reason: 0.2 sits *below* the near-stop hold median of 0.2744 (#10), so it
clips ordinary driving everywhere rather than just the ambiguous stops.

**So: if brake authority limits are wanted, clamp the model's output at
inference, and report the clipped metric alongside it.** That gets both the
control property and the honest near-stop number for free, with no retraining
and no accuracy cost: the baseline row *is* the raw-target model scored under
clipping (0.01917 at `brake@0.3`), and every clipped-target arm sits above it.
The same holds for `gas<=0.3` (+3.9% gas, +1.3% brake — a target clip on one
field degrading another).

⚠️ **Measurement trap this sweep makes concrete**: baseline raw brake 0.02253
→ `brake@0.3` 0.01917 is a −15% "improvement" that is *entirely* the metric
shrinking. None of these clipped numbers can be compared against #10's 0.0815
near-stop floor, which is on the unclipped target.

**Do not clip gas or steering either, and the steering case is the instructive
one.** `clip_probe.py` finds gas never saturates — a positive Δv slope in every
speed x magnitude bin, including the top ones — so a gas clip is pure signal
destruction, and the sweep agrees (`gas<=0.3`: +3.9% gas). Steering *does*
saturate, but only below 5 km/h (`dheading ∝ steer·speed`), and a speed-gated
`|steer| <= 0.3 @ v<5` clip is the only arm in the sweep that wins its own
matched metric: −1.3% diluted over all rows, −2.0% on raw steering, paired
negative in 4/5 seeds (−2.3 / −0.4 / −1.9 / −3.2 / +1.2), and larger still
when scored on the near-stop rows alone.

**That win should not be banked, for a reason the brake case does not share.**
Wheel angle at a standstill is *pre-positioning* — it determines the launch
direction, so it should be recoverable in principle from the future path, just
not from the path at the resolution and horizon this controller currently sees.
Brake above `c` is unrecoverable at any horizon and from any signal in the ego
trajectory. So the steering clip is deleting information a better model could
use, and it merely happens to flatter the current one; the brake clip is
deleting information nothing can use. Only the second is a defensible metric
change, and neither is a defensible target change.

**"In principle" is doing real work in that argument, and #10's own numbers
push back on it.** In the stationary near-stop group the controllers score
steering 0.0703 / 0.0699 / 0.0708 against a speed-only oracle's 0.0654 — the
one field and regime where conditioning on the trajectory is *worse* than
ignoring it. (That oracle is fit in-sample on the eval rows, so it is
advantaged; the signal is that brake matched it there and steering did not.)
So the asymmetry above is an argument from physics, not from measurement, and
the measurement currently leans the other way. Leaving the steering clip
unbanked is still the right call — a −1.3% diluted win against a 2.0% steering
noise floor is not a result in either direction — but the *reason* should be
read as provisional. If a later model does recover launch direction from the
path, that is what retires this paragraph; until then near-stop steering is an
open weakness rather than a resolved one.

#### The by-product: this setup's run-to-run noise floor

The sweep repeats an identical baseline under 5 `_fit` seeds on a fixed split,
which finally puts a number on the bar every controller delta in this document
has to clear: **brake 0.9%, gas 1.0%, steering 2.0%** (baseline `brake@0.3`
per seed: 0.01912 / 0.01886 / 0.01925 / 0.01943 / 0.01918). Anything smaller
than that in a single-seed sweep is not a result.

Applied retroactively, and this is the uncomfortable part:

- **#9's steering row (+0.8% at h30) is inside the noise band** — read it as
  "flat", which is what that finding concluded anyway on independent grounds.
- **#11's steering exception (−4% at K=6–15) is barely at 2x the floor** and
  came from a single seed. It is the one claim in #11 that should not be
  relied on without a paired re-run; the gas and brake degradations there
  (+19%/+22% at K=30) clear the floor by a wide margin.
- **#9's brake −9.9% and turn-signal +3.8pp survive comfortably**, as does
  **#13's gas −12.5%** (12x the gas floor, with non-overlapping seed ranges).
- **#14's own `brake<=0.4` arm (+2.2%) is only ~2x the brake floor** — hence
  reporting it as "worse in 4/5 seeds" rather than as a point estimate. The
  `<=0.3` and `<=0.2` arms clear it 5x and 29x.

**Always run >=3 paired seeds before believing a controller delta.** Bug #5
was the mechanism; this is the magnitude.

Artifact: `/tmp/clip_seeds.json` (per-seed, per-band, every matched metric).

## Recommended next steps, in priority order

1. **Adopt `horizon=30, action_steps=1` as the controller baseline**
   (`/tmp/horizon_sweep_traj/controller_h30k1.pt`). It is the best
   configuration found: −9.9% brake L1 and +3.8pp turn-signal accuracy over
   the 6-tick formulation, at no cost on gas or steering, and it costs
   nothing to build (finding #8 — no re-predict, no GPU for the data). Its
   action target is already correctly aligned for one tick of controller
   latency (the target sits at `anchor+1`, not at the anchor) — see #11's
   latency note, which also gives the compensation curve if the real
   end-to-end latency turns out to be longer than one tick.
2. **Report stratified metrics as standard practice going forward** for this
   controller (and probably for the policy objective more broadly) — the
   aggregate L1 numbers materially understate how bad near-stop braking is,
   and this was invisible until stratifying. Findings #9–#11 are all
   invisible in aggregate: #9's entire brake gain is one band, and #10's
   floor is half of another. Two additions from #14: **report near-stop brake
   on a clipped metric** (c=0.3 — 42% of the 0.0815 floor is unresolvable
   label noise being counted as error) while still *training* on the raw
   target. Note what 0.0471 is and is not: it is **#10's oracle floor
   recomputed under the clip**, not a controller score. The controller's own
   clipped near-stop brake L1 was never measured — measure it before quoting
   a single "honest number" for the band, and expect it to sit near 0.0471 the
   way the unclipped controllers sat on 0.0815. Second addition: **run >=3
   paired fit seeds** before believing any delta, against a floor of brake
   0.9% / gas 1.0% / steering 2.0%.
3. **Condition the gas head on the per-drive 4-parameter calibration vector**
   (#13): −12.5% gas L1, non-overlapping seed ranges, and the shuffled
   control confirms it is the drive-specific values rather than the extra
   input width. Do **not** use the `gas_gain` scalar alone (−1.4%, inside the
   seed noise), and **gate the vector into the gas head only**. Two things to
   settle before adopting it: whether the vector is really car setup or just a
   proxy for the drive's speed regime, and how it gets estimated online, given
   that the offline protocol needs half a drive to calibrate.

   **The gas-head gate rests on evidence of absence, not absence of
   evidence.** #13's own read is that "nothing about brake is supportable from
   this run": `allparams`' brake sd is 0.0100 because one seed of three blew
   up to 0.0403, and the *shuffled* control is also +5.1%, so what that arm
   measures is capacity and noise rather than calibration. Steering's +1.8 to
   +4.6% straddles the 2.0% steering noise floor (#14), and its shuffled arms
   are worse than the real ones too. So gas-head-only is the safe adoption,
   but brake and steering conditioning are **untested rather than refuted** —
   re-run those two with >=5 paired seeds before ruling them out.
4. **Clamp brake at the model's output and in the metric, never at the
   training target** (#14). Output clipping is free; target clipping costs
   +4.9% brake L1 on the matched metric at `brake<=0.3` and +26% at
   `brake<=0.2`, makes the 0–5 km/h band it was meant to fix worse, and
   degrades the heads it does not touch. Do not clip gas (it never saturates)
   or steering (it saturates only below 5 km/h, and what saturates there is
   launch-direction pre-positioning that a better model could still recover
   from the path — an argument from physics; #14's addendum notes that #10's
   measurement currently leans the other way, so treat the *reason* as
   provisional even though the do-not-clip call is not).
5. **Four directions are now closed; don't reopen them without new evidence.**
   Past pedal lag taps are a net loss *as extra inputs to a single-tick
   forward Δv model*, with an understood mechanism (#6). That is the only form
   of pedal history anything here tested, and it does **not** close pedal
   history as an input to the *inverse* controller — the opposite direction,
   and one rec #6 lists as an open lead. Do not read #6 as closing it.
   Future horizon beyond ~10s cannot help near-stop braking, because half
   that band never moves and its trajectory is horizon-invariant (#9, #10).
   Emitting a longer action sequence degrades the action you apply (#11).
   Clipping a continuous target — brake or gas — is worse than clipping the
   output, with an understood mechanism (#14). The near-stop brake ambiguity
   cannot be modelled away or extended away, and it can be *defined* away only
   in the metric (#14's clipped floor), never in the target; only the
   exteroception lead below actually resolves it.
6. **The open lead is exteroception, not horizon.** Finding #10 bounds what
   ego-trajectory-only can achieve at v≈0 and shows the controller is already
   there. Everything left in that band is in signals this pipeline
   deliberately excludes — the camera stream (traffic light phase, lead
   vehicle brake lights) and the pedal history. Testing that means leaving
   the parquet-only setting, so it is a genuinely bigger piece of work than
   anything above; size it accordingly. A cheap first probe: check whether
   the upstream checkpoint's own action head (`policy/prediction_value`,
   which *does* see images) beats the 0.0815 floor in the stationary
   near-stop group. If it doesn't, the images aren't carrying it either.

   **On the pedal-history half of this lead, and its apparent conflict with
   #6.** #6 tested pedal history as extra inputs to a *forward* model
   predicting Δv, where `speed[t]` already Markov-encodes the pedal's
   accumulated effect — hence the net loss. The inverse controller asks the
   opposite question (predict the pedal itself), and there the r = 0.94–0.98
   pedal autocorrelation that #6 cites as the *reason* lag taps fail is
   precisely what would make them succeed. The two findings are compatible;
   only the forward direction is closed. But the probe needs a guard, because
   it is trivially gameable: a controller fed its own recent action can post a
   very low L1 by copying it forward while learning nothing from the
   trajectory, and in deployment it is then autoregressive on its own output
   with no trajectory tracking. So score any such arm against (a) an
   action-history-only baseline with the trajectory removed, and (b) the
   *moving* near-stop rows, not just the stationary half — an arm that only
   wins where the car is not moving has learned to hold the last pedal value,
   which is what #10 says is unresolvable, not a resolution of it.
7. It is still worth checking whether the *existing*
   `policy/trajectory_value` predicted-trajectory arm (via `cmd_eval`'s case
   (b)) shows the same near-stop weakness — but a longer horizon for the
   upstream trajectory head itself is a retraining question, not a data one,
   and #9 now says the payoff would be in the 5–35 km/h approach-to-stop
   regime rather than at the stop.
8. **Keep the field-specific personalization split — but per-drive for `gas`
   only.** Findings #3/#4 originally read this as "per-drive for gas/brake,
   global/pooled for steering"; **#12 retires the brake half.** `brake_gain`
   has split-half reliability 0.000 under every gate tried, so there is no
   per-drive brake coefficient to calibrate and #3's 82% figure was gas alone.
   The current split is **per-drive gas, global/pooled brake and steering** —
   steering because it is genuinely constant across the fleet, brake because
   it is genuinely unmeasurable in this data. Worth keeping as the parametric
   baseline if a parametric (non-MLP) controller is ever needed for
   interpretability or deployment-simplicity reasons — but for raw accuracy
   the MLP remains the best controller across all three continuous fields.
9. The `cmd_modes` machinery (`_one_way_anova`, `_report_matched_condition`,
   the matched-condition FORWARD/INVERSE tests) is reusable if the "hidden
   speed modes" hypothesis needs re-testing on a different parquet or a
   larger drive sample (currently run at `n_drives=250`).

## How to reproduce the headline numbers

```bash
# tests
just test tests/test_dead_reckoning.py tests/test_controller.py \
    tests/test_tick_trace.py tests/test_calibration.py

# finding #8: rebuild the per-tick trace, report anchor coverage by horizon
# and dead-reckoning drift at 10s, assert the stitch reproduces the windowed
# path exactly at horizon=6, and write the tick parquet. CPU only. The tick
# table itself builds in 3.5s; --verify and the horizon sweep dominate the
# runtime, and the sweep holds several (n x horizon+1) arrays in RAM at once
# -- add `--n-drives 40 --seed 7` (and drop --out) for a fast sample run.
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller stitch \
    --horizon 30 --verify \
    --out outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.ticks.parquet

# per-drive vs global parametric dynamics + MLP comparison, and the
# near-stop stratified breakdown, were run as ad hoc scripts against:
#   nix develop --command uv run python -c "..."
# using `load(parquet, n_drives=250, seed=7)` from
# src/rmind/scripts/trajectory_action_controller.py and the fit/invert
# functions in src/rmind/components/controller.py. No standalone script
# currently persists these exact ad hoc analyses -- re-derive from the
# function signatures above plus the split protocol described under "Data
# and reproducibility" if you need to rerun them verbatim.

# the reproducible MLP checkpoint used for the near-stop MLP comparison:
ls /tmp/controller_v3.pt /tmp/controller_v3.json

# finding #9: the trajectory-horizon sweep. Full file, one shared anchor set
# and drive split across all three horizons. ~15 min on one GPU (CPU works,
# pass --device cpu; the MLP is tiny, the GPU mostly saves the val passes).
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller horizon \
    --horizons 6 15 30 --action-steps 1 --steps 20000 --seed 7 \
    --out-dir /tmp/horizon_sweep_traj

# finding #11: the control-horizon sweep at fixed trajectory horizon 30.
# ~25 min. Note --action-steps must not exceed the shortest --horizons entry.
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller horizon \
    --horizons 30 --action-steps 1 3 6 15 30 --steps 20000 --seed 7 \
    --out-dir /tmp/horizon_sweep_ctrl

# finding #10: the near-stop information floor, scored against the #9
# checkpoints. Seconds, CPU only.
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller nearstop \
    --horizon 30 --seed 7 \
    --ckpt /tmp/horizon_sweep_traj/controller_h{6,15,30}k1.pt

# finding #12: split-half reliability of every per-drive gain. ~90s, CPU
# only, reads the tick parquet alone. This is the cheap gate to run BEFORE
# building any per-drive conditioning scheme.
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller calibrate

# finding #13: the per-drive calibration as an MLP input, against permuted
# controls, 3 fit seeds per arm on one fixed split. ~35 min on one GPU.
nix develop --command uv run python -m \
    rmind.scripts.trajectory_action_controller gaincond \
    --horizon 30 --steps 20000 --fit-seeds 7 11 13 \
    --out /tmp/gasgain/gaincond.json

# finding #14: the target-clip sweep. 5 fit seeds x 6 target transforms on
# one fixed split, ~100s per arm-seed on one GPU (~50 min total). Reports both
# the raw L1 and the matched (clip-both-sides) metrics; run it from the repo
# root, it puts `src` on the path itself and writes /tmp/clip_seeds.json
# incrementally, so a partial run is still readable.
nix develop --command uv run python scripts/action_clipping/clip_target_sweep.py

# and the three cheap probes behind #14's mechanism paragraphs (seconds each,
# CPU, tick parquet only): where the actions saturate, and what #10's floor
# becomes once the target is clipped.
nix develop --command uv run python scripts/action_clipping/clip_probe.py
nix develop --command uv run python scripts/action_clipping/steer_probe.py
nix develop --command uv run python scripts/action_clipping/floor_clip.py

# the per-speed-band table in #9 comes out of the sweep's own JSON:
#   json.load(open("/tmp/horizon_sweep_traj/horizon_sweep.json"))
#   -> results[i]["val_l1_by_speed_band"][field][band]
```

A note on `seed=7` in the sweeps: it drives the drive shuffle in
`drive_split()`, the minibatch sampler, and — since bug #5 — model
initialization, so every configuration in a sweep shares one split and one
sampling stream and each fit is individually reproducible. Comparing across
sweeps run at different seeds is not meaningful at the ~1% level; comparing
rows *within* one sweep is what the design is for, but see #14's noise-floor
subsection for how large a within-sweep delta has to be before it means
anything: **brake 0.9%, gas 1.0%, steering 2.0%**.
`gaincond --fit-seeds` is the pattern to copy for anything at that scale: fix
the split, vary the fit seed, report the spread next to the delta.
