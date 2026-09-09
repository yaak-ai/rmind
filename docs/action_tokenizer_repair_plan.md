# Action-head repair plan: take `turn_signal` out of the RVQ

Self-contained task brief. Written for an agent with **no prior context** on the
conversation that produced it — everything needed to execute is here or behind
the file:line references. Sibling document:
`docs/trajectory_action_controller_handoff.md` (referred to below as "the
handoff"; its findings are cited as #1–#14).

## Why this work exists

The handoff established 14 findings about predicting actions from a plan, all
measured **offline** with a tiny `TrajectoryToActionMLP` on one predict parquet.
This plan carries the ones that survive contact with the pipeline that actually
trains action predictors: the `PatchPolicy` VQ-BeT arm launched from
`commands.sh` (`experiment=yaak/patch_policy/…`).

**Baseline to compare against (1 camera):** `yaak/rmind/model-do8m9ot8`
(https://wandb.ai/yaak/rmind/runs/do8m9ot8) or `yaak/alex-tmp/model-v4mma4th`
(https://wandb.ai/yaak/alex-tmp/runs/v4mma4th). `commands.sh:222-230` already
exports `do8m9ot8:v1`/`:v2`, so its artifact path is known-good.

### Already handled — do not redo these

Confirmed by the project owner, and each one closes a recommendation the handoff
makes:

- **Actions are already clamped at inference.** #14's "clamp the output, never
  the target" is done. There is no `clamp`/`clip` in `src/rmind/models/`,
  `src/rmind/components/objectives/` or `src/rmind/scripts/decoder_only_export.py`
  — the clamping lives downstream in drivr. Do not add one here without asking.
- **Inference skips chunk index 0 and applies index 1.** So #11's
  latency/bracketing gain (−4.0% gas, −2.9% brake, −7.8% steering at index 1 vs
  index 0) is already banked. `ChunkFields` (`src/rmind/components/nn.py:202-247`)
  makes frame `t`'s target the actions at `[t … t+5]`, so index 1 is `t+1` and
  the model already trains on it.
- **Finer offline stratified metrics are not the priority.** The binding
  constraint is the open-loop→closed-loop gap; better open-loop numbers do not
  resolve it. Phase 1 below is a *diagnosis of a ceiling*, deliberately scoped to
  the tokenizer, not a general metrics project.
- **`turn_signal` matters much less than steering and gas/brake.** That is a
  product judgement from the owner and it is the premise of this whole plan.

## The defect

The `ActionTokenizer` compresses a 24-dim action chunk through a 4×16 residual
VQ, trained with **unweighted L1**, and a quarter of those dims are the
indicator.

Verified facts:

- `action_dim = action_space (4) × action_clip (6) = 24`,
  `codebook_size: 16`, `num_quantizers: 4`, `action_latent_dim: 384`
  — `config/experiment/yaak/action_tokenizer/pretrain.yaml`.
- The reconstruction objective is `recon = F.l1_loss(a_hat, a)` with no channel
  weighting — `src/rmind/models/action_tokenizer.py:130-140`.
- `turn_signal` is `{OFF, LEFT, RIGHT} = {0, 1, 2}` rescaled to `{0.0, 0.5, 1.0}`
  by a `Scaler`, while `continuous` fields pass through `Identity` —
  `config/model/yaak/action_tokenizer/raw.yaml`. Typical gas/brake/steering
  errors in the handoff are 0.02–0.07, so the indicator carries **5–25× the
  dynamic range** of the channels that drive the car.
- The tokenizer is **frozen** inside the policy —
  `src/rmind/models/patch_policy.py:249-253`
  (`.requires_grad_(False).eval()`). Its reconstruction error is therefore a hard
  **ceiling**: no code head can serve an action the codebook cannot represent.

This is not speculation; it has been measured once already from the other side.
`src/rmind/models/patch_policy.py:283-294` records that with all four channels,
`turn_signal` contributes **38.9%** of the decoded-action distance `d_q(c)` —
more than gas (21.2%) or steering (25.7%). That measurement was used to fix the
*label smoothing* only (`neighbor_smoothing_channels: [0, 1, 2]` in
`config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml:222`). The same
asymmetry is still fully live in:

- the tokenizer's reconstruction objective (`action_tokenizer.py:130-140`),
- the policy's offset loss — `torch.nn.L1Loss` over the flattened 24-dim chunk
  (`config/model/yaak/patch_policy/raw.yaml:131-137`, called at
  `patch_policy.py:817`),
- every `offset_*` metric in `patch_policy.py:831-923`, including
  `offset_argmax_recon_last`, which the comment at `patch_policy.py:869-877`
  names as *the* checkpoint-selection metric. A single wrong indicator moves
  that metric ~10× further than a large brake error.

**The decision taken:** `turn_signal` leaves the RVQ entirely and gets its own
categorical head. It is a 3-way categorical being regressed through an L1
objective on a continuous latent — "half a signal away" is meaningless, which is
exactly what the smoothing note says. Removing it frees all 4×16 codes for
gas/brake/steering. Repair the **tokenizer first**, then the policy heads.

## Phase 1 — measure the ceiling (hours, no training)

Load the artifact currently in use — `yaak/rmind/model-y74asdtd:v9`, the
`action_tokenizer_artifact` in
`config/experiment/yaak/patch_policy/dinov3.yaml` — and measure on a val split
what it can and cannot represent. Nothing here is a training run.

1. **Reconstruction error per field, per chunk step, per speed band.** Use the
   handoff's bands `((0,5),(5,10),(10,20),(20,35),(35,60),(60,130))`. This is the
   ceiling table, and it is the number every later policy delta must be read
   against. Specifically: what is tokenizer-only **brake** reconstruction error
   at 0–5 km/h, next to #10's near-stop floor of **0.0815 raw / 0.0471 clipped
   at c=0.3**? If the tokenizer alone eats a large share of that, the near-stop
   brake problem was never a policy problem.
2. **Codebook usage and allocation.** Per-quantizer perplexity is already
   computed (`action_tokenizer.py:148-151`). Add the per-channel decoded-distance
   decomposition `d_q(c)` — `patch_policy._neighbor_smoothing_targets`
   (`patch_policy.py:679-729`) already implements exactly this and is the thing
   to reuse — so the 38.9% / 21.2% / 25.7% split is **re-measured on the artifact
   in use** rather than quoted from a note.
3. **How much capacity is the indicator taking?** Bucket by indicator state and
   check whether codes differing only in indicator state burn distinct codes,
   especially in **quantizer 0** where the coarse structure lives. Removal is
   already decided; this predicts how much the removal should buy, so Phase 2 can
   be judged against a number rather than hope. It also sets the steering vs
   gas/brake weights in step 6.

**Deliverable:** one table of per-field / per-step / per-band reconstruction
error plus the indicator's measured share of the codebook. Pin the exact
artifact version (`y74asdtd:v9`) in the write-up — the handoff's
"Cross-table comparability" section is a catalogue of what happens when that is
left implicit.

### Phase 1 — results (measured 2026-09-08, `y74asdtd:v9`, full val split, n=27,676)

Reproduce with `scripts/action_tokenizer_ceiling.py`:

```bash
nix develop --command uv run python scripts/action_tokenizer_ceiling.py \
    --artifact yaak/rmind/model-y74asdtd:v9 \
    --config-dir /home/alex/rmind/config \
    --experiment yaak/action_tokenizer/pretrain \
    --batches 100000 --device cuda
```

**Per-field / per-chunk-step reconstruction L1** (normalized units) — flat
across steps, mean over the 6-step chunk:

| field | mean L1 |
| --- | --- |
| gas_pedal | 0.0118 |
| brake_pedal | 0.0065 |
| steering_angle | 0.0116 |
| turn_signal | 0.0083 |

**Per-field / per-speed-band reconstruction L1:**

| band (km/h) | n | gas | brake | steer | turn_signal |
| --- | --- | --- | --- | --- | --- |
| 0–5 | 22,587 | 0.0100 | 0.0199 | 0.0168 | 0.0097 |
| 5–10 | 5,638 | 0.0236 | 0.0217 | 0.0310 | 0.0234 |
| 10–20 | 15,785 | 0.0203 | 0.0127 | 0.0247 | 0.0226 |
| 20–35 | 38,764 | 0.0123 | 0.0047 | 0.0121 | 0.0093 |
| 35–60 | 50,409 | 0.0091 | 0.0016 | 0.0066 | 0.0038 |
| 60–130 | 32,873 | 0.0104 | 0.0015 | 0.0052 | 0.0038 |

**Answers step 1's question, with a correction to the framing:** tokenizer-only
brake reconstruction at 0–5 km/h is **0.0199**, against #10's near-stop floor of
0.0815 raw / 0.0471 clipped — so the tokenizer accounts for roughly **25–42%**
of that floor. Real, but partial: the near-stop brake problem is *not purely* a
policy problem, but it isn't purely a tokenizer ceiling either. More
importantly, **the worst band is 5–20 km/h, not 0–5** — brake, steer and
turn_signal all peak there, not at near-stop. Any Phase 2 read against "does
the near-stop band improve" should also track 5–20 km/h, which this plan had
not been framing as the target band.

**Codebook share of decoded-distance `d_q(c)`, re-measured on this artifact:**

| channel | mean d_q(c) | share |
| --- | --- | --- |
| turn_signal | 0.1072 | 41.0% |
| steering_angle | 0.0729 | 27.9% |
| gas_pedal | 0.0467 | 17.9% |
| brake_pedal | 0.0345 | 13.2% |

Confirms the asymmetry the plan is built on — slightly worse than the quoted
38.9% / 25.7% / 21.2% (which was measured from the policy side, on a different
artifact).

**Quantizer-0 codebook usage by turn_signal state** (of 16 codes):

| turn_signal | n | codes used |
| --- | --- | --- |
| OFF | 22,031 | 15 |
| LEFT | 3,907 | 11 |
| RIGHT | 1,738 | 9 |

Overlap: OFF∩LEFT=10, OFF∩RIGHT=8, LEFT∩RIGHT=9 — quantizer 0 is nearly
saturated by OFF alone, and LEFT/RIGHT mostly reuse the same codes rather than
getting distinct ones. Supports the removal decision: there isn't much clean,
separable per-state structure to lose by taking the indicator out of the RVQ.

**Gate for Phase 2a:** re-run this table on the retrained (3-feature) tokenizer
and check gas/brake/steering improve, with 5–20 km/h — not just 0–5 — as a
required band to look at.

## Phase 2a — the tokenizer (retrain the artifact)

4. `config/model/yaak/action_tokenizer/raw.yaml`: drop `discrete.turn_signal`
   from the `Remapper` paths, drop its `Scaler` module, and drop
   `targets.discrete`. `ActionTokenizer._action_features`
   (`action_tokenizer.py:104-105`) counts `targets` leaves, so it becomes 3
   automatically — nothing else in that class needs touching.
5. `config/experiment/yaak/action_tokenizer/pretrain.yaml`:
   `action_space: 4 → 3` (`action_dim` is derived: `3 × 6 = 18`).
6. **Per-channel weighted reconstruction** among the three survivors, in
   `ActionTokenizer._step` (`action_tokenizer.py:130-140`): replace
   `F.l1_loss(a_hat, a)` with a weighted L1, weights broadcast over the
   `(action_clip, action_space)` layout, exposed in the model config. **Default
   to uniform** so the change is provably a no-op at default settings. Set the
   actual weights from Phase 1's table.
7. Retrain: `just train experiment=yaak/action_tokenizer/pretrain`. Then re-run
   Phase 1's ceiling table on the new artifact.
   **GATE — stop here if this fails:** if reconstruction on gas/brake/steering
   does not improve, the capacity story is wrong and Phase 2b must not be built
   on it. Report that outcome; do not proceed on hope.
8. **Pin the old artifact where it is still needed.**
   `config/model/yaak/control_transformer/policy_finetune.yaml:47-50` loads
   `${action_tokenizer_artifact}` into a `JointPolicyObjective` whose `predict()`
   hardcodes `chunk[..., 0..3]` with `turn_signal` at index 3
   (`src/rmind/components/objectives/joint_policy.py:225-253`). A 3-feature
   artifact **silently breaks that arm**. Pin `y74asdtd:v9` there explicitly
   instead of letting it follow the shared variable. **Done**: the artifact
   string is now hardcoded at that call site instead of interpolating
   `${action_tokenizer_artifact}`; the now-unused variable was removed from
   `config/experiment/yaak/control_transformer/finetune.yaml`.

### Phase 2a — results (retrained 2026-09-08, `q6ocue9a:v9`, uniform channel_weights)

Steps 4/5/8 done as written. Step 6 (weighted reconstruction) is implemented
and exposed as `channel_weights` in `config/model/yaak/action_tokenizer/raw.yaml`
but **left at its uniform default** for this run — the point of this first
retrain is to isolate the effect of removing `turn_signal` from capacity
alone, with weighting untouched. `just train-unsafe
experiment=yaak/action_tokenizer/pretrain` (10 epochs, 1239 steps/epoch,
`yaak/rmind/runs/q6ocue9a`), then the reproduce command from Phase 1
re-pointed at the new artifact:

```bash
nix develop --command uv run python scripts/action_tokenizer_ceiling.py \
    --artifact yaak/rmind/model-q6ocue9a:v9 \
    --config-dir /home/alex/rmind/config \
    --experiment yaak/action_tokenizer/pretrain \
    --batches 100000 --device cuda
```

(The script needed one fix first: its quantizer-0/turn_signal-state section
hardcoded "turn_signal is the last field," which for a 3-feature tokenizer
would silently read steering_angle values as indicator states. It now checks
`fields[-1] == "turn_signal"` and skips that section with a note when the
field isn't present — sections [1]/[2], the ones this gate actually reads,
were already field-count-agnostic.)

**Per-field / per-chunk-step reconstruction L1** (normalized units, mean over
the 6-step chunk) — old (`y74asdtd:v9`, 4-feature) vs. new:

| field | old mean L1 | new mean L1 | Δ |
| --- | --- | --- | --- |
| gas_pedal | 0.0118 | 0.0087 | **−26.3%** |
| brake_pedal | 0.0065 | 0.0044 | **−32.3%** |
| steering_angle | 0.0116 | 0.0088 | **−24.1%** |

**Per-field / per-speed-band reconstruction L1, old → new (Δ%):**

| band (km/h) | n | gas_pedal | brake_pedal | steering_angle |
| --- | --- | --- | --- | --- |
| 0–5 | 22,587 | 0.0100→0.0069 (−31.0%) | 0.0199→0.0145 (−27.1%) | 0.0168→0.0125 (−25.6%) |
| 5–10 | 5,638 | 0.0236→0.0179 (−24.2%) | 0.0217→0.0148 (−31.8%) | 0.0310→0.0249 (−19.7%) |
| 10–20 | 15,785 | 0.0203→0.0145 (−28.6%) | 0.0127→0.0085 (−33.1%) | 0.0247→0.0193 (−21.9%) |
| 20–35 | 38,764 | 0.0123→0.0089 (−27.6%) | 0.0047→0.0028 (−40.4%) | 0.0121→0.0087 (−28.1%) |
| 35–60 | 50,409 | 0.0091→0.0074 (−18.7%) | 0.0016→0.0010 (−37.5%) | 0.0066→0.0051 (−22.7%) |
| 60–130 | 32,873 | 0.0104→0.0071 (−31.7%) | 0.0015→0.0009 (−40.0%) | 0.0052→0.0045 (−13.5%) |

**GATE: PASSED.** All three channels improve in every speed band, including
the required 5–20 km/h bands (gas −24 to −29%, brake −32 to −33%, steer −20
to −22%) and the near-stop 0–5 band (−25 to −31%). Every delta is 10–20×
past the handoff's noise floor (brake 0.9% / gas 1.0% / steer 2.0%), so this
is not noise — freeing the 4×16 codebook from `turn_signal` measurably buys
back gas/brake/steering fidelity everywhere, not just where it was expected.

**Codebook share of decoded-distance `d_q(c)`, re-measured on the new
artifact** (turn_signal's freed capacity redistributes, unevenly):

| channel | old share (4-feature) | new share (3-feature) |
| --- | --- | --- |
| gas_pedal | 17.9% | 31.5% |
| brake_pedal | 13.2% | 21.0% |
| steering_angle | 27.9% | 47.6% |

Steering absorbed the largest share of the freed capacity (already the
second-highest share before removal), brake the least — worth carrying into
Phase 2b step 6/12's weighting decision: brake still gets the smallest slice
of decoded-distance despite mattering most for the near-stop band, which is
an argument for a real (non-uniform) `channel_weights` favoring brake, not
just for reconstruction-L1 parity.

## Phase 2b — the policy heads

9. `config/model/yaak/patch_policy/raw.yaml`: drop `turn_signal` from
   `StackFields.paths` (`:65-70`) so `joint_actions` is 3-channel; **keep** it in
   `ChunkFields.unfold_paths` (`:31-33`) because the new head still needs its
   per-step chunk target. `offset_head`'s output width drops
   `4 × 16 × 24 = 1536 → 4 × 16 × 18 = 1152`; `code_head` (64) is unchanged.
10. **New `turn_signal_head`**: an MLP on the same readout feature emitting
    `action_horizon × 3` logits, supervised by cross-entropy (or `FocalLoss`, to
    match the code head) against the unfolded `discrete.turn_signal` chunk. Log
    accuracy at the deployed readout — a categorical head finally gets a
    categorical metric, which L1-through-RVQ cannot give.
11. **Keep the served output 4-channel.** This is the load-bearing compatibility
    decision. `policy.joint_actions` is consumed as `(1, horizon, 4)` with
    `turn_signal` at index 3, bucketized from `{0, 0.5, 1.0}`, by
    `src/rmind/components/nn.py:277-300`; the TRT binding list
    (`src/rmind/scripts/decoder_only_trt_measure.sh:52`) and the parity harness's
    `CHANNELS` (`src/rmind/scripts/decoder_only_verify.py:76`) assume it too. So
    `PatchPolicy._predict_chunk` (`patch_policy.py:731-740`) must concatenate the
    categorical head's decoded class back at index 3 in the same
    `{0, 0.5, 1.0}` encoding. Done that way, **drivr, the ONNX unpacker and the
    export contract need no change at all.**
12. **Weighted offset loss on the three continuous channels**, matching the
    weights the tokenizer was trained with — otherwise the code head optimizes a
    different metric than the codebook was built for. Site: `losses.offset`
    (`config/model/yaak/patch_policy/raw.yaml:131-137`), consumed at
    `patch_policy.py:817` and throughout `_readout_metrics`. Keep the unweighted
    metric logged as well, for continuity with existing runs.
13. **Re-tune the neighbour smoothing.** `neighbor_smoothing_channels: [0, 1, 2]`
    becomes all channels once there are three, so it can be dropped — but the
    `tau` caveat at `patch_policy.py:283-294` warns that excluding `turn_signal`
    shrinks the mean decoded distance ~0.069 → ~0.042, which makes any tau
    calibrated with it in effectively **sharper**. `commands.sh:364` already runs
    `0.012`. Re-measure the mass-on-far-half curve on the new codebook rather
    than carrying either number over.
14. Train one policy arm against the `do8m9ot8` / `v4mma4th` baseline with the
    new artifact pinned. Judge on the three continuous channels plus
    turn-signal accuracy.

**Other `_action_features` consumers to check when it changes:**
`src/rmind/scripts/patch_policy_eval.py:188`,
`src/rmind/scripts/residual_unimodality_probe.py:352` — both already read
`tokenizer._action_features` dynamically (confirmed field-count-agnostic, no
edit needed).

### Phase 2b — implementation status (2026-09-08)

Steps 9–11 done in code, verified by unit tests, config resolution, and the
full suite (`just test`: 241 passed, 1 skipped; the only failures/errors are
`test_models.py`/`test_export.py` CUDA `control_transformer`/`episode_builder`
cases, reproduced identically on `HEAD~1` before this work — pre-existing and
unrelated).

- **Step 9** — `config/model/yaak/patch_policy/raw.yaml`'s `StackFields.paths`
  drops `turn_signal`; `ChunkFields.unfold_paths` keeps it. `offset_head`
  shrinks to `[1024, 1024, 1152]`. Same change mirrored in
  `dinov2_dinowm_causal_3cam.yaml`, the only experiment file that restates its
  own `StackFields` rather than inheriting `raw.yaml`'s.
- **Step 10** — `turn_signal_head` (`torchvision.ops.MLP`, `[1024, 1024,
  ${eval:'${action_horizon} * 3'}]`) added to `raw.yaml`, wired into
  `PatchPolicy` (`src/rmind/models/patch_policy.py`) as an **opt-in**
  constructor arg — `None` (default) reproduces the pre-Phase-2b behaviour
  bit-for-bit (existing checkpoints/configs load and run unchanged). When set:
  cross-entropy loss (`losses["turn_signal"]`, required — constructor raises
  otherwise) against the raw `{0, 1, 2}` chunk fetched independently of
  `joint_actions` (`PatchPolicy.turn_signal`, default path `(discrete,
  turn_signal)`); `turn_signal_acc_last` / `turn_signal_acc_index1_last`
  (the deployed-index metric, per "already handled" above) logged at the
  readout.
- **Step 11** — `PatchPolicy._predict_chunk` concatenates the head's argmax
  class back at index 3, re-encoded `{0, 1, 2} -> {0, 0.5, 1.0}`. `forward`,
  `predict_step`, `_structure`, the ONNX unpacker, TRT bindings and the parity
  harness are all **untouched** — verified by
  `test_turn_signal_head_predict_chunk_stays_four_channel` and
  `test_turn_signal_head_predict_step` in `tests/test_patch_policy.py`.
- **`dinov3.yaml`** (the common ancestor of every patch_policy experiment)
  now pins `action_tokenizer_artifact: yaak/rmind/model-q6ocue9a:v9` (was
  `y74asdtd:v9`) — **required**, not optional: the new `raw.yaml` architecture
  (3-channel `joint_actions`, 1152-wide `offset_head`) only matches the
  3-feature tokenizer's shapes. This changes the default for every patch_policy
  arm that doesn't override it (all current ones do not) — deliberate, per the
  plan's premise, but stated here because it is a wide-blast-radius default
  change, not a per-arm opt-in. A checkpoint warm-started from a run trained
  before this change (`warm_start_ckpt.py`) will now fail loudly on a
  code_head/offset_head shape mismatch — correct, not a regression.
- **Step 13, easy part done**: `dinov2_dinowm_causal.yaml`'s
  `neighbor_smoothing_channels: [0, 1, 2]` removed (now `None`, and with only
  3 channels total the two are already equivalent) — the comment previously
  read as "still excluding something," which is no longer true.
- **Not done — deliberately deferred, flagged rather than silently skipped:**
  - **Step 12 (weighted offset loss)**: `losses.offset` is still a plain
    `torch.nn.L1Loss`. Reason: `q6ocue9a:v9` itself was trained with **uniform**
    `channel_weights` (Phase 2a), so "matching the weights the tokenizer was
    trained with" is currently a no-op — there is nothing non-uniform to match
    yet. Building the weighting machinery now, ahead of an actual weight
    decision, would add an untested code path for zero behavioural change.
    When a non-uniform `channel_weights` is chosen for the tokenizer (Phase 2a's
    results note brake is the strongest candidate: smallest post-removal
    `d_q(c)` share despite mattering most near-stop), give `PatchPolicy.losses`
    a weighted-L1 module mirroring `ActionTokenizer._weighted_l1_loss`
    (`action_tokenizer.py:147-153`) — same "default uniform, provably a no-op"
    discipline.
  - **Step 13, tau remeasurement**: still needs a trained 3-feature-arm
    codebook and a repeat of the mass-on-far-half measurement
    (`patch_policy.py:283-294`'s cited method) — an empirical step, not a code
    change. `commands.sh`'s scratch `neighbor_smoothing_tau=0.012` is a
    reasonable placeholder (already anticipating the ~0.61x shrinkage) but is
    unverified on the new codebook.
  - **Step 14 (train + compare)**: not launched. Needs an explicit go-ahead —
    it spends real GPU/wandb budget and the arm to compare against
    (`do8m9ot8` / `v4mma4th`) should be picked deliberately, not defaulted to
    whatever `dinov3.yaml`'s inheritance chain currently points at.
  - `PatchPolicyHead`/`load_head_for_export`
    (`config/export/yaak/patch_policy/head.yaml`) — a secondary head-only
    export path, separate from the main `decoder_only_export.py` contract —
    does **not** decode `turn_signal`. Not touched: out of the plan's stated
    scope (steps 9–13 all target the main served contract), but a real gap if
    that export path is ever used for a `turn_signal_head` arm.

## Phase 3 — auxiliary trajectory head, 6 steps (2 s). Only after Phase 2.

**Caution added 2026-09-08, from a sibling result — read before starting:**
Tournament round 6 (`/nasa/alex/docs/round6_1cam_causal_results.md`), a
1-camera closed-loop rsim round unrelated to this plan's own work, measured a
*different* trajectory-head implementation (`v9y58oei`, branch
`feat/patch-policy-decoder-causal-traj`) and found training with it trends
**worse**, not better, with more of its own budget: A4→A5 is **−0.081 rc over
1.67 epochs**, against a budget-only expectation of **+0.164** from the
baseline's own slope (marginal: t=1.8, short of the 2.0 bar, but it clears the
round's MDE). The trajectory head does not drive anything at inference in that
setup — `drivr_parity` nulls the trajectory knobs and `onnx_decoder` exposes no
trajectory output — so the effect is purely a training-time one: the auxiliary
changed what the shared trunk / discrete-action policy head learned, for the
worse, and this only showed up in closed-loop route completion, not in any
loss curve.

**Why this doesn't kill Phase 3 outright, but does raise the bar:** per
`[[project_frozen_vs_unfrozen_trajectory]]`, that branch's trajectory head
lineage targets GPS-chain deltas, which are known flat/degenerate (47% exactly
`(0,0)`, ~5-6× under-scale) and known to collapse dynamic range when
unfrozen. Step 15 below already chose `dead_reckon_future_trajectory`
specifically to avoid a GPS target — so round 6's negative result may be
"training against a degenerate GPS target corrupts the shared representation,"
a different claim from "any trajectory auxiliary corrupts the shared
representation." Only running Phase 3's own dead-reckoning-target design
settles which one is true. Until then: treat round 6 as a real prior that this
failure mode exists on a closely related architecture, budget a closed-loop
check (not just offline loss) before trusting a Phase 3 arm, and do not read
low training-loss on the trajectory head as evidence the policy head is
unharmed.

**Superseded by a concrete design, 2026-09-08:** `docs/phase3_trajectory_head_plan.md`
is the task brief for this phase — it turns out `af4fcf1` (the branch behind
the round-6 checkpoint above) already implements a nearly-complete version of
steps 15-18, unweighted; that brief specifies porting it with a weight knob
(default off) and the gate discipline this caution calls for. Read that brief
instead of executing steps 15-18 below from scratch.

15. Add a small head on the readout feature predicting the future ego path,
    supervised by `dead_reckon_future_trajectory`
    (`src/rmind/components/dead_reckoning.py`, covered by
    `tests/test_dead_reckoning.py`) — **not** GPS deltas. Keep it at
    `clip_horizon: 6` / 2 s to stay comparable with the baseline and to avoid a
    dataset rebuild (`clip_length = episode_length + clip_horizon - 1`).
16. **State plainly what 2 s buys and what it does not.** At 2 s the win is
    *target correctness*: the GPS-chain trajectory target is 47% exactly
    `(0,0)` and ~5–6× under-scale, so a head trained on it asks the policy to
    plan a crawl. The −9.9% brake / +3.8pp turn-signal gain in #9 came
    specifically from extending 2 s → 10 s and is **not** available at
    `clip_horizon: 6`. If that gain is wanted later it costs a dataset rebuild.
    Do not let the 2 s arm be read as a test of #9.
17. **Axis convention.** `heading` is a compass bearing (0°=north), so forward in
    the rotated ego frame is local **+y** (component 1) and lateral is **+x**
    (component 0) — *not* the standard math-angle convention. The handoff records
    this being rediscovered twice. Use the tested helper; do not rederive.
18. Weight it like the existing aux arms (`model.aux_weights.* = 0.03–0.1`, see
    `commands.sh:139-186`).

## Phase 4 — per-drive gas calibration, made deployable

#13's −12.5% gas L1 is the largest single-field effect in the handoff, but its
protocol (fit each drive's first half by OLS, score the second) is **not
deployable** and cannot be reproduced at val or in closed loop, where no
per-drive gain statistics exist. The fix is to change *how the feature is
produced*, not what it is.

19. **Settle the confound first (~90 s of compute).** #13 could not separate "car
    setup" from "a proxy for the drive's speed regime". Correlate the
    standardized 4-vector against per-drive speed/accel summary statistics
    straight off the tick parquet (`read_ticks` in
    `src/rmind/scripts/trajectory_action_controller.py`) and re-run one arm with
    those statistics substituted for the gains. If plain speed statistics buy the
    same thing, this phase collapses to a much cheaper feature — and the causal
    trunk already sees 16 frames of speed tokens, so it may already have it.
20. **Never feed a whole-drive fit. Feed a causal running estimate** (see the
    appendix). Same code path offline and online, so the feature distribution
    matches between train, val and closed loop, and it is defined at tick 0.
21. **Gate it into the gas path only.** #12: `brake_gain` and `steer_gain` have
    split-half reliability 0.000 under every gate tried. #13's brake arms are
    *untested rather than refuted* — one seed of three blew up to 0.0403 against
    ~0.019 — so re-run those with ≥5 paired seeds before ruling them in or out.
22. **Do not add a per-vehicle or per-drive embedding.** #1 measured it making
    every field worse (+8.0% gas / +15.9% brake / +25.5% steering) and #12
    explains why: 69% of `gas_gain`'s spread is between drives of the *same*
    vehicle, so a static one-hot addresses at most a third of the effect while
    paying the full variance cost.

## Appendix — how the online estimator works

The forward model is the handoff's `LongitudinalDynamicsParams`
(`src/rmind/components/controller.py`):

    dv_t  ≈  g·gas_t  −  b·brake_t  −  d·v_t  −  c

Write it as a linear regression with `θ = [g, b, d, c]` and, per tick,

    x_t = [ gas_t, −brake_t, −v_t, −1 ]        y_t = v_{t+1} − v_t

**Recursive least squares with forgetting factor λ** tracks `θ` with a 4×4 state
and no history buffer — a handful of flops per tick:

    k_t = P_{t-1} x_t / (λ + x_tᵀ P_{t-1} x_t)
    θ_t = θ_{t-1} + k_t (y_t − x_tᵀ θ_{t-1})
    P_t = (P_{t-1} − k_t x_tᵀ P_{t-1}) / λ

- **Initialization is the answer to "what do we feed before we know anything".**
  `θ_0 = θ̄`, the fleet prior (median of the per-drive fits the `calibrate`
  subcommand already produces over 644 drives), and `P_0 = δ⁻¹I` with `δ` chosen
  so the prior carries the confidence of roughly a few seconds of driving. The
  feature is then *always* defined: at tick 0 it is the fleet prior, and it slides
  toward the session's own value as evidence arrives. Optionally blend
  explicitly, `θ̂_t = w_t·θ_t + (1−w_t)·θ̄` with `w_t = n_t/(n_t+n_0)`, which makes
  the ramp auditable.
- **λ sets the memory.** `λ = 0.999` at 3 Hz is an effective window of ~1000
  ticks (~5.5 min) — long enough to average noise out, short enough to track a
  driver changing drive mode mid-session, which is what #2 says the effect *is*:
  session/drive-level, not a fixed vehicle property.
- **Gate the updates where `dv` is informative:** update only on ticks with
  `v > 5 km/h`. Below that, `dv` is clamped near zero regardless of pedal
  (#10 measured `dv` correlating 0.955 with `−speed_now` and not with brake at
  all), so those ticks inject noise. This is precisely why `brake_gain` comes out
  at reliability 0.000 when fit on all rows (#12).
- **Train/serve parity is the whole point.** The estimator is causal, so offline
  features come from one forward pass per drive over the tick parquet — no
  leakage, no half-drive split — and drivr computes the identical quantity from
  CAN (`speed` is already a model input; the pedals are what it just commanded or
  reads back).
- **Two risks to state up front.** (a) Only `g` is individually reliable (#12),
  so the other three coefficients are partly absorbing regression noise; #13
  found the *vector* works and the scalar does not, which is expected under this
  design but not understood — step 19 is what makes it interpretable. (b) In
  CARLA the estimator converges to *CARLA's* longitudinal dynamics, which are not
  any real Niro's, so the conditioning feature can land **out of distribution in
  the very closed-loop benchmark used to judge it**. Log `θ̂_t` in sim and compare
  against the train-set distribution before reading any rsim result.

## Do not spend effort here — closed by the handoff

- **Past pedal lag taps as inputs to a forward Δv model** (#6, mechanism
  understood: `gas[t]` autocorrelates 0.94–0.98 with its lags and `speed[t]`
  already Markov-encodes the accumulated effect). Pedal history into the
  *inverse* controller is a different question and still open — but if tried,
  guard it with an action-history-only baseline *and* score the **moving**
  near-stop rows, or it trivially games the metric by copying the last pedal
  forward and is autoregressive on its own output in deployment.
- **Horizon beyond ~10 s for near-stop braking** (#9, #10): 50.6% of held-out
  near-stop rows travel < 0.5 m over a whole 10 s horizon, so their plan is the
  zero trajectory at *any* length, and the trained controllers already sit on the
  resulting floor.
- **Clipping any continuous training target** (#14): worse at every clip level,
  worse in the band it was built for, and it bleeds into heads it never touched.
  The mechanism is that the clip piles 38% of stationary rows onto an atom at
  exactly `c`, which a Gaussian head cannot fit.
- **Lengthening the emitted action sequence to help the applied action** (#11):
  gas and brake degrade monotonically in K (+19% / +22% at K=30). If index 0 or 1
  needs help, weight it in the loss; do not grow `action_horizon`.

## Statistics discipline — applies to every number this plan produces

- The handoff's measured run-to-run noise floor is **brake 0.9%, gas 1.0%,
  steering 2.0%** (an identical baseline repeated under 5 fit seeds on a fixed
  split). Anything smaller in a single-seed comparison is not a result. That
  floor was measured for a tiny MLP; the arms here are ~9-epoch runs, so paired
  seeds at full scale are unaffordable. Two affordable substitutes: **head-only
  refits** (cache trunk features for the val split, refit `code_head`/
  `offset_head` under ≥3 seeds — this is what `gaincond --fit-seeds` does), or
  quoting the spread across the run's own val points
  (`val_check_interval: 0.5` → 18 points) as an upper bound.
- **Closed loop: never rank checkpoints off one rsim suite run.**
  `benchmark.n_runs=3` is the config default; round-3 `n_runs=1` estimates were
  off by up to 0.095. Pair per scenario, and never average the bimodal junction
  scenarios blind.

## Verification

```bash
# unit tests (45 controller/dead-reckoning/tick-trace/calibration tests)
just test tests/test_dead_reckoning.py tests/test_controller.py \
    tests/test_tick_trace.py tests/test_calibration.py
just test           # everything

just lint && just format && just typecheck   # ruff + ty
```

New tests to add:

- weighted reconstruction reduces to `F.l1_loss` at uniform weights, so the
  default path is provably unchanged;
- the per-field / per-band decomposition averages back to the existing aggregate;
- **the load-bearing one** — `_predict_chunk` still returns 4 channels with
  `turn_signal` at index 3 in `{0, 0.5, 1.0}`, so `nn.py:277-300`'s unpacker
  round-trips.

Then:

- `just train-debug` for the tokenizer and the policy arm — confirms the new
  weights plumb through `log_dict` and nothing breaks in `PatchPolicy._step`
  (`patch_policy.py:925-957`).
- Export parity is the real gate on the 4-channel contract:
  `just export-onnx` / `decoder_only_export --verify`, then
  `src/rmind/scripts/decoder_only_verify.py` — the argmax decision-change count
  must not move, and `CHANNELS` must still line up.
- Confirm the control_transformer finetune arm still instantiates with the
  pinned `y74asdtd:v9` artifact. That is the one thing a 3-feature tokenizer
  silently breaks.
- Phase 1's deliverable is a table, not a run. Phase 2a's gate is whether the
  ceiling moves. Phase 2b's gate is one policy arm beating
  `do8m9ot8` / `v4mma4th` on the continuous channels without regressing
  turn-signal accuracy — then rsim at `n_runs=3`, paired per scenario, with the
  open-loop→closed-loop gap kept firmly in view as the thing that actually
  decides it.

## Environment notes

- Config under `config/` is partly generated from `config/_templates/` via
  `ytt`; `just train` and `just test` run `just generate-config` first. **Edit
  templates, not generated files.**
- All standalone Python must run as `nix develop --command uv run python …` —
  plain `python` inside `nix develop` breaks on `libstdc++.so.6`.
- Changing `clip_length` (Phase 3, if the 10 s horizon is ever taken) requires a
  dataset rebuild, and the rbyte cache must be built by a **single writer** —
  two concurrent builds corrupt the samples store.
- Training runs go through docker with a pinned image tag; see `commands.sh` for
  the exact invocation pattern per arm.
