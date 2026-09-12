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
1. **Codebook usage and allocation.** Per-quantizer perplexity is already
   computed (`action_tokenizer.py:148-151`). Add the per-channel decoded-distance
   decomposition `d_q(c)` — `patch_policy._neighbor_smoothing_targets`
   (`patch_policy.py:679-729`) already implements exactly this and is the thing
   to reuse — so the 38.9% / 21.2% / 25.7% split is **re-measured on the artifact
   in use** rather than quoted from a note.
1. **How much capacity is the indicator taking?** Bucket by indicator state and
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

| field          | mean L1 |
| -------------- | ------- |
| gas_pedal      | 0.0118  |
| brake_pedal    | 0.0065  |
| steering_angle | 0.0116  |
| turn_signal    | 0.0083  |

**Per-field / per-speed-band reconstruction L1:**

| band (km/h) | n      | gas    | brake  | steer  | turn_signal |
| ----------- | ------ | ------ | ------ | ------ | ----------- |
| 0–5         | 22,587 | 0.0100 | 0.0199 | 0.0168 | 0.0097      |
| 5–10        | 5,638  | 0.0236 | 0.0217 | 0.0310 | 0.0234      |
| 10–20       | 15,785 | 0.0203 | 0.0127 | 0.0247 | 0.0226      |
| 20–35       | 38,764 | 0.0123 | 0.0047 | 0.0121 | 0.0093      |
| 35–60       | 50,409 | 0.0091 | 0.0016 | 0.0066 | 0.0038      |
| 60–130      | 32,873 | 0.0104 | 0.0015 | 0.0052 | 0.0038      |

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

| channel        | mean d_q(c) | share |
| -------------- | ----------- | ----- |
| turn_signal    | 0.1072      | 41.0% |
| steering_angle | 0.0729      | 27.9% |
| gas_pedal      | 0.0467      | 17.9% |
| brake_pedal    | 0.0345      | 13.2% |

Confirms the asymmetry the plan is built on — slightly worse than the quoted
38.9% / 25.7% / 21.2% (which was measured from the policy side, on a different
artifact).

**Quantizer-0 codebook usage by turn_signal state** (of 16 codes):

| turn_signal | n      | codes used |
| ----------- | ------ | ---------- |
| OFF         | 22,031 | 15         |
| LEFT        | 3,907  | 11         |
| RIGHT       | 1,738  | 9          |

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
1. `config/experiment/yaak/action_tokenizer/pretrain.yaml`:
   `action_space: 4 → 3` (`action_dim` is derived: `3 × 6 = 18`).
1. **Per-channel weighted reconstruction** among the three survivors, in
   `ActionTokenizer._step` (`action_tokenizer.py:130-140`): replace
   `F.l1_loss(a_hat, a)` with a weighted L1, weights broadcast over the
   `(action_clip, action_space)` layout, exposed in the model config. **Default
   to uniform** so the change is provably a no-op at default settings. Set the
   actual weights from Phase 1's table.
1. Retrain: `just train experiment=yaak/action_tokenizer/pretrain`. Then re-run
   Phase 1's ceiling table on the new artifact.
   **GATE — stop here if this fails:** if reconstruction on gas/brake/steering
   does not improve, the capacity story is wrong and Phase 2b must not be built
   on it. Report that outcome; do not proceed on hope.
1. **Pin the old artifact where it is still needed.**
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
alone, with weighting untouched. `just train-unsafe experiment=yaak/action_tokenizer/pretrain` (10 epochs, 1239 steps/epoch,
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

| field          | old mean L1 | new mean L1 | Δ          |
| -------------- | ----------- | ----------- | ---------- |
| gas_pedal      | 0.0118      | 0.0087      | **−26.3%** |
| brake_pedal    | 0.0065      | 0.0044      | **−32.3%** |
| steering_angle | 0.0116      | 0.0088      | **−24.1%** |

**Per-field / per-speed-band reconstruction L1, old → new (Δ%):**

| band (km/h) | n      | gas_pedal              | brake_pedal            | steering_angle         |
| ----------- | ------ | ---------------------- | ---------------------- | ---------------------- |
| 0–5         | 22,587 | 0.0100→0.0069 (−31.0%) | 0.0199→0.0145 (−27.1%) | 0.0168→0.0125 (−25.6%) |
| 5–10        | 5,638  | 0.0236→0.0179 (−24.2%) | 0.0217→0.0148 (−31.8%) | 0.0310→0.0249 (−19.7%) |
| 10–20       | 15,785 | 0.0203→0.0145 (−28.6%) | 0.0127→0.0085 (−33.1%) | 0.0247→0.0193 (−21.9%) |
| 20–35       | 38,764 | 0.0123→0.0089 (−27.6%) | 0.0047→0.0028 (−40.4%) | 0.0121→0.0087 (−28.1%) |
| 35–60       | 50,409 | 0.0091→0.0074 (−18.7%) | 0.0016→0.0010 (−37.5%) | 0.0066→0.0051 (−22.7%) |
| 60–130      | 32,873 | 0.0104→0.0071 (−31.7%) | 0.0015→0.0009 (−40.0%) | 0.0052→0.0045 (−13.5%) |

**GATE: PASSED.** All three channels improve in every speed band, including
the required 5–20 km/h bands (gas −24 to −29%, brake −32 to −33%, steer −20
to −22%) and the near-stop 0–5 band (−25 to −31%). Every delta is 10–20×
past the handoff's noise floor (brake 0.9% / gas 1.0% / steer 2.0%), so this
is not noise — freeing the 4×16 codebook from `turn_signal` measurably buys
back gas/brake/steering fidelity everywhere, not just where it was expected.

**Scope of that gate, stated plainly (added 2026-09-11):** this is a
*tokenizer reconstruction* gate — it measures how well the RVQ round-trips an
action, not how well a policy trained on it drives. The policy arm that step
14 called for has since been run (`mv2qv1nd`, see the step-14 entry below) and
is roughly **neutral** end-to-end: gas +4.0%, steering +3.8%, brake −4.9% on
`predict/policy/score_l1`. Do not cite "GATE: PASSED" as evidence of a
policy-level win; it is evidence that the capacity story was correct and that
Phase 2b was safe to build.

**Codebook share of decoded-distance `d_q(c)`, re-measured on the new
artifact** (turn_signal's freed capacity redistributes, unevenly):

| channel        | old share (4-feature) | new share (3-feature) |
| -------------- | --------------------- | --------------------- |
| gas_pedal      | 17.9%                 | 31.5%                 |
| brake_pedal    | 13.2%                 | 21.0%                 |
| steering_angle | 27.9%                 | 47.6%                 |

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
1. **New `turn_signal_head`**: an MLP on the same readout feature emitting
   `action_horizon × 3` logits, supervised by cross-entropy (or `FocalLoss`, to
   match the code head) against the unfolded `discrete.turn_signal` chunk. Log
   accuracy at the deployed readout — a categorical head finally gets a
   categorical metric, which L1-through-RVQ cannot give.
1. **Keep the served output 4-channel.** This is the load-bearing compatibility
   decision. `policy.joint_actions` is consumed as `(1, horizon, 4)` with
   `turn_signal` at index 3, bucketized from `{0, 0.5, 1.0}`, by
   `src/rmind/components/nn.py:340-368`; the TRT binding list
   (`src/rmind/scripts/decoder_only_trt_measure.sh:52`) and the parity harness's
   `CHANNELS` (`src/rmind/scripts/decoder_only_verify.py:76`) assume it too. So
   `PatchPolicy._predict_chunk` (`patch_policy.py:731-740`) must concatenate the
   categorical head's decoded class back at index 3 in the same
   `{0, 0.5, 1.0}` encoding. Done that way, **drivr, the ONNX unpacker and the
   export contract need no change at all.**
1. **Weighted offset loss on the three continuous channels**, matching the
   weights the tokenizer was trained with — otherwise the code head optimizes a
   different metric than the codebook was built for. Site: `losses.offset`
   (`config/model/yaak/patch_policy/raw.yaml:131-137`), consumed at
   `patch_policy.py:817` and throughout `_readout_metrics`. Keep the unweighted
   metric logged as well, for continuity with existing runs.
1. **Re-tune the neighbour smoothing.** `neighbor_smoothing_channels: [0, 1, 2]`
   becomes all channels once there are three, so it can be dropped — but the
   `tau` caveat at `patch_policy.py:283-294` warns that excluding `turn_signal`
   shrinks the mean decoded distance ~0.069 → ~0.042, which makes any tau
   calibrated with it in effectively **sharper**. `commands.sh:364` already runs
   `0.012`. Re-measure the mass-on-far-half curve on the new codebook rather
   than carrying either number over.
1. Train one policy arm against the `do8m9ot8` / `v4mma4th` baseline with the
   new artifact pinned. Judge on the three continuous channels plus
   turn-signal accuracy. **DONE** — `mv2qv1nd` vs `v4mma4th`, results in
   "Phase 2b step 14 — results" below.

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
- **Step 10** — `turn_signal_head` (`torchvision.ops.MLP`, `[1024, 1024, ${eval:'${action_horizon} * 3'}]`) added to `raw.yaml`, wired into
  `PatchPolicy` (`src/rmind/models/patch_policy.py`) as an **opt-in**
  constructor arg — `None` (default) reproduces the pre-Phase-2b behaviour
  bit-for-bit (existing checkpoints/configs load and run unchanged). When set:
  cross-entropy loss (`losses["turn_signal"]`, required — constructor raises
  otherwise) against the raw `{0, 1, 2}` chunk fetched independently of
  `joint_actions` (`PatchPolicy.turn_signal`, default path `(discrete, turn_signal)`); `turn_signal_acc_last` / `turn_signal_acc_index1_last`
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
    **Still outstanding as of 2026-09-11, and now a confound**: every run in
    the step-14 comparison below passed `++model.neighbor_smoothing_tau=0.012`
    on the CLI — including the `v4mma4th` baseline, which uses the **4-channel**
    `y74asdtd:v9` codebook that 0.012 was never meant for (0.02 is the value
    measured there; 0.012 is the *scaled estimate* for the 3-channel codebook).
    So the baseline ran with an effectively sharper-than-calibrated tau. Fold
    this into any reading of the A-vs-B deltas, and remeasure before the next
    arm.
  - `PatchPolicyHead`/`load_head_for_export`
    (`config/export/yaak/patch_policy/head.yaml`) — a secondary head-only
    export path, separate from the main `decoder_only_export.py` contract —
    does **not** decode `turn_signal`. Not touched: out of the plan's stated
    scope (steps 9–13 all target the main served contract), but a real gap if
    that export path is ever used for a `turn_signal_head` arm.

### Phase 2b step 14 — results (run 2026-09-08, recorded 2026-09-11)

Step 14 ("train one policy arm against the `do8m9ot8` / `v4mma4th` baseline
with the new artifact pinned") **was launched**, ~10 h after `d66e331`, and has
finished. This section supersedes the "not launched" note that stood here.

|                               | baseline `v4mma4th`           | Phase 2b `mv2qv1nd`           |
| ----------------------------- | ----------------------------- | ----------------------------- |
| wandb                         | `yaak/alex-tmp/runs/v4mma4th` | `yaak/alex-tmp/runs/mv2qv1nd` |
| commit                        | `fd14fd1`                     | `d66e331`                     |
| tokenizer                     | `y74asdtd:v9` (4-feature)     | `q6ocue9a:v9` (3-feature)     |
| heads                         | code + offset                 | + `turn_signal_head`          |
| `offset_head`                 | `[1024, 1024, 1536]`          | `[1024, 1024, 1152]`          |
| `neighbor_smoothing_channels` | `[0, 1, 2]`                   | `null`                        |
| host                          | `renate`                      | `aboutblank`                  |
| end state                     | epoch 8, step 117,232         | epoch 8, step 117,214         |

Both arms: `experiment=yaak/patch_policy/dinov2_dinowm_causal_readout`
(**not** `dinov3.yaml` — see the inheritance note in
`docs/phase3_trajectory_head_plan.md`), 1 camera, `episode_length 32` /
`clip_length 37` / `episode_stride 31`, batch 48, bf16-mixed, seed 1337, and
the same CLI schedule `lr=2e-4 lr_warmup_steps=9850 lr_total_steps=123000`
(exactly half of `dinov2_dinowm_causal.yaml`'s `19400`/`243000`, with `lr`
doubled for the doubled batch).

**Result — `predict/policy/score_l1`, end of training:**

| field          | `v4mma4th` | `mv2qv1nd` | Δ         | vs. noise floor        |
| -------------- | ---------- | ---------- | --------- | ---------------------- |
| gas_pedal      | 0.0743     | 0.0773     | **+4.0%** | 1.0% → outside, worse  |
| brake_pedal    | 0.0325     | 0.0309     | **−4.9%** | 0.9% → outside, better |
| steering_angle | 0.0182     | 0.0189     | **+3.8%** | 2.0% → outside, worse  |

**Read: roughly neutral, not a win.** Two channels degrade and one improves,
all marginally outside the handoff's noise floor (brake 0.9% / gas 1.0% /
steer 2.0%) — and a single unpaired pair of runs is exactly what the
"Statistics discipline" section below says not to over-read. The Phase 2a
reconstruction gain (−24 to −33% per band) did **not** translate into a
policy-level gain at this budget.

`turn_signal_head` itself works: val `turn_signal_acc_last` 0.939,
`turn_signal_acc_index1_last` 0.941 (the chunk index actually served), train
0.986. Moving `turn_signal` out of the RVQ into its own classifier is
therefore a clean architectural win even though the package is score-neutral.

**Confounds — this is a package comparison, not an ablation.** The two arms
differ in *five* things at once (tokenizer artifact, RVQ channel count,
`turn_signal_head`, `offset_head` width, `neighbor_smoothing_channels`), plus
`tau=0.012` was applied to the baseline's 4-channel codebook (see step 13
above), plus they ran on different hosts. Attributing the ±4% to any single
change is not supported by these two runs.

## Phase 3 — auxiliary trajectory head, 6 steps (2 s). Only after Phase 2.

15. Add a small head on the readout feature predicting the future ego path,
    supervised by `dead_reckon_future_trajectory`
    (`src/rmind/components/dead_reckoning.py`, covered by
    `tests/test_dead_reckoning.py`) — **not** GPS deltas. Keep it at
    `clip_horizon: 6` / 2 s to stay comparable with the baseline and to avoid a
    dataset rebuild (`clip_length = episode_length + clip_horizon - 1`).
01. **State plainly what 2 s buys and what it does not.** At 2 s the win is
    *target correctness*: the GPS-chain trajectory target is 47% exactly
    `(0,0)` and ~5–6× under-scale, so a head trained on it asks the policy to
    plan a crawl. The −9.9% brake / +3.8pp turn-signal gain in #9 came
    specifically from extending 2 s → 10 s and is **not** available at
    `clip_horizon: 6`. If that gain is wanted later it costs a dataset rebuild.
    Do not let the 2 s arm be read as a test of #9.
01. **Axis convention.** `heading` is a compass bearing (0°=north), so forward in
    the rotated ego frame is local **+y** (component 1) and lateral is **+x**
    (component 0) — *not* the standard math-angle convention. The handoff records
    this being rediscovered twice. Use the tested helper; do not rederive.
01. Weight it like the existing aux arms (`model.aux_weights.* = 0.03–0.1`, see
    `commands.sh:139-186`).

## Phase 4.1 — acceleration as the longitudinal action. Runs before Phase 4.

**Status 2026-09-12: steps 23-24 done, retrained (`ujef8lzw:v9`), GATE NOT
PASSED.** See "Step 23 result" and "Step 24 result" below. Steering
reconstruction regressed in 5/6 speed bands (mean +9.1%) instead of
improving as it did at 4→3 — per this section's own stated gate ("if it
does not, the capacity story is exhausted — report and stop"), stop here
and get a decision before iterating further (retrying `channel_weights`,
more epochs, a different `Scaler` range, or abandoning the merge). Steps
25/26 (decode head, contract change) remain untouched, as scoped.

Phase 4 conditions the model on per-drive gains. Phase 4.1 attacks the same
finding from the other side: `gas_pedal`/`brake_pedal` are **actuator commands**
whose map to motion is vehicle- and session-specific (#3: a per-drive
longitudinal fit cuts gas L1 0.414 → 0.073, an 82% reduction; #2 locates the
mode at drive level; #12 puts `gas_gain` reliability at 0.500), while
acceleration is the **vehicle-invariant intent**. Take gas/brake out of the RVQ,
replace them with one `acceleration` channel, emit acceleration, and let drivr
close the longitudinal loop — and there is nothing left to condition on. Two
routes to one target, so 4.1 is written to run first: if it works, Phase 4 is
unnecessary; if it fails, Phase 4 is untouched.

### The enabling fact — it changes the premise

`VehicleMotion.acceleration_x/y/z` carries **real measured signal in m/s²**,
verified 2026-07-15 on the 15-drive predict set (2022–23 Niro + 2024 G1 mix). It
is simply never parsed: `grep acceleration_x config/ src/ scripts/` returns
exactly one hit, and it is a comment recording the absence
(`config/inference/yaak/control_transformer/policy_allfields.yaml:44-48`).

So the target does **not** have to be a 3 Hz finite difference of interpolated
CAN speed. That matters because #10 showed `dv` is degenerate in the band that
decides this: at `v < 5 km/h` with `brake > 0.3`, `dv` correlates **0.955 with
`−speed_now`** and not with brake pressure at all. A measured accelerometer
channel is a different signal with a different failure mode. **Caveat to carry:
the July check covered predict drives only, not the 619–655 train drives.**
Step 23 re-establishes it on the training population before anything is built.

### The case for

- **The capacity mechanism is already proven here.** Phase 2a took the RVQ from
  4 → 3 channels and bought **−24% to −33%** reconstruction L1 on every survivor
  in every speed band, 10–20× the noise floor. Gas+brake are the best remaining
  pair to merge: they are near-mutually-exclusive by construction — the whole
  `pedal_conflict_*` script family exists to count the physically-contradictory
  overlap — so two channels currently carry roughly one degree of freedom plus a
  sign bit.
- **It removes unidentifiable variation without clipping a target.** Brake > 0.3
  is physically inert at 0–5 km/h (the brake→Δv slope is ~0 at every pedal level)
  and is 24.8% of stationary rows, ~15% of val. Clipping that target was measured
  *worse* at every level (+2.3% to +26%), because the clip piles 38% of
  stationary rows onto an atom at exactly `c` — see "Do not spend effort here"
  #14. A change of coordinates is not a clip: in acceleration units the
  degenerate dimension does not exist to be piled up.
- **It relieves the checkpoint-selection metric.** `offset_argmax_recon_last`
  (`patch_policy.py:1152-1155`) averages L1 over all channels, so today an inert
  near-stop brake error counts as heavily as a real one.

### The case against — do not soften these

- **The identifiability wall relocates; it does not move.**
  `invert_longitudinal_dynamics` (`controller.py:176-197`) needs `brake_gain`,
  and #12 measured its split-half reliability at **0.000** under every gate tried
  (all rows / v>5 / braking only / hard braking; `r` stuck at +0.18–0.25).
  Handing the inverse to drivr does not make that coefficient estimable. drivr
  needs a **closed-loop** controller on measured acceleration, not the
  feedforward inversion this repo already implements. That is a controls project,
  not a config change.
- **`acceleration_x` is achieved, not commanded.** It carries road grade, road
  load, and — in body frame — a gravity component from brake-induced nose-dive,
  contaminating exactly the channel of interest.
- **The blast radius far exceeds Phase 2b's**, which was explicitly designed so
  the served contract stayed 4-channel and drivr/ONNX/TRT needed no change. The
  **silent-corruption** case, named explicitly because it throws no exception:
  `patch_policy.py:936` appends `turn_signal` at the *tail* of the continuous
  block, so it lands at index 3 only because `_action_features == 3`. At 2
  channels it becomes index 2 and every downstream reader takes steering for a
  turn signal — wrong numbers, no error, across ~12 hand-duplicated sites whose
  order is defined solely by YAML insertion order, with nothing cross-checking
  `StackFields.paths` against `ActionTokenizer.targets` against `offset_head`'s
  width against `CHANNELS`.
- **Measurement capacity is the binding constraint and is oversubscribed.** rsim
  MDE at `n_runs=3` is ~0.060 route completion and could not resolve a 0.025
  effect. Phase 2b came back neutral; two Phase 3 arms are mid-flight with **no
  closed-loop check ever run**. Changing the serving contract means the
  closed-loop test measures the model *and* a new controller at once — the worst
  confound structure available, and this document is already a catalogue of that
  mistake.

### Steps

23. **Data gate (hours, no training). The kill criterion lives here.** Plumb
    `acceleration_x/y/z` into `config/_templates/dataset/yaak/action_{train,val}.yaml`
    and `predict.yaml` — four sites each: proto field list (`:730-744`), aligner
    columns (`:767-781`, `method: interp`), validity filter (`:794-800`, already
    NOT-NULL-safe since proto3 scalars decode to defaults rather than null), and
    the chunk cast (`:828-832`). These are **ytt templates** — edit them, not the
    generated files, then `just generate-config`. Then answer four questions:

    1. **Populated on the train drives?** `n_unique`, range and zero-fraction
       **per vehicle generation** across the full train set. A generation that
       reads all-zero is a dead field for that subset, and 15 predict drives is
       not evidence about 619.
    1. **Which axis is longitudinal, and what sign?** Correlate each of
       `a_x/a_y/a_z` against `Δspeed/Δt` using `_tick_pairs`
       (`trajectory_action_controller.py:1594-1631`), which already gates on
       phase grid, frame advance and backwards clocks. **Do not assume `x`** —
       step 17 records the ego-frame convention being rediscovered twice.
    1. **The decisive number.** For `v < 5 km/h ∧ brake > 0.3`, does measured
       acceleration separate `brake=0.2` from `brake=0.8` where `dv` provably
       cannot? Quantify pitch/gradient contamination alongside it (a stationary
       car on a slope reads non-zero `a_x` with no pedal at all).
       **Fail → acceleration inherits the full identifiability wall; stop and
       report.** The capacity argument in step 24 may still hold, but the drivr
       rewrite would buy nothing in the band that matters most.
    1. **The invertibility ceiling.** Fit `(acceleration, speed) → (gas, brake)`
       on held-out data, per `SPEED_BANDS`. This is the hard upper bound on what
       *any* drivr-side controller can recover, and it is the number that decides
       whether the cross-repo work is justified. Read it against Phase 2a's
       ceiling table (`q6ocue9a:v9`, brake 0.0145 at 0–5 km/h).

    **Deliverable: one table, with the drive set pinned**, per the Phase 1
    precedent — not a run.

    **Step 23 result (2026-09-12, `scripts/accel_data_gate.py` →
    `docs/accel_data_gate.json`, train split, 654 drives / 2 537 955 samples,
    all Niro — this train split has no G1 drives):**

    1. Populated: `acceleration_y` has 2.33M unique values, zero-fraction
       1.2%, well-populated. `acceleration_z` is dead (always exactly 0.0,
       1 unique value) — confirms the earlier predict-set finding on the
       full train population, not just 15 drives.
    1. Axis/sign: `acceleration_y` correlates `r=0.867` with realized `Δv`;
       `acceleration_x` correlates `r=0.016` — not longitudinal. Sign is the
       same as `Δv`. (Consistent with the raw.yaml comment's earlier
       20-drive estimate of `r=0.84`.)
    1. Decisive band (`v<5 km/h, brake>0.3`, n=392 838): `acceleration_y`
       still correlates `r=0.49` with `Δv` in-band (degraded from 0.87
       but nonzero, unlike `Δv` itself which #10 showed is ~degenerate
       here) and separates brake buckets by mean (`-0.064` at
       brake 0.3–0.5 vs `-0.212` at brake 0.5–1.0, n=372 166/20 672).
       Stationary-no-pedal grade contamination: mean `+0.107`, std `0.265`
       — real but small next to the braking-band signal. **Does not fail
       the kill criterion.**
    1. Invertibility ceiling `(acceleration, speed) → (gas, brake)`, OLS
       per speed band, held-out drives: `brake_l1` ranges `0.0084` (60–130
       km/h) to `0.076` (0–5 km/h); `gas_l1` ranges `0.020` (0–5 km/h) to
       `0.094` (60–130 km/h). Directly comparable to Phase 2a's ceiling
       table (`q6ocue9a:v9`, brake 0.0145 at 0–5 km/h) — same order of
       magnitude, i.e. the representation is not obviously worse than the
       existing tokenizer's own reconstruction ceiling.

    Population percentiles (used to set the `Scaler` bounds in step 24):
    mean `-0.033`, std `0.611`; 0.1st/99.9th percentiles `-2.81`/`+2.49`;
    true min/max `-9.32`/`+10.24` are single-tick outliers (\<0.1% of mass),
    not sustained driving values.

01. **The 2-channel tokenizer.** `config/model/yaak/action_tokenizer/raw.yaml`:
    `Remapper.paths.continuous` and `targets.continuous` become
    `{acceleration, steering_angle}`. Leaf order in `targets` **is** the channel
    order (`action_tokenizer.py:120-122` — pytree leaf order is dict insertion
    order) and `channel_weights` is positional against it; keep the existing
    comment discipline there. `config/experiment/yaak/action_tokenizer/pretrain.yaml:17`:
    `action_space: 3 → 2`, so `action_dim` derives to 12 — note that **nothing
    validates `action_dim == action_clip × _action_features`**; a mismatch
    surfaces only as a runtime matmul error.

    **Normalization is a real decision, not a detail.** Every continuous field
    enters the RVQ in raw normalized units through `Identity` — there is no
    `Scaler` in the production path. Acceleration in m/s² has a different dynamic
    range from `[0,1]` pedals and `[-1,1]` steering, and Phase 1 showed the RVQ
    spends capacity in proportion to dynamic range: that was the entire
    `turn_signal` finding. Pick the scaling from step 23's measured distribution
    and state it; `Scaler` (`src/rmind/components/norm.py:33-68`) is the existing
    tool. **Done**: `in_range=[-4.0, 4.0]` (`raw.yaml`), chosen because it sits
    past the 99.9th percentile (`±2.5–2.8`) from step 23's full-population
    gate, clipping only the \<0.1% single-tick outlier tail (`±9–10`) rather
    than real driving dynamics.

    `scripts/action_tokenizer_ceiling.py` needs no edit — sections [1] and [2]
    are field-count-agnostic and auto-label from `targets`; section [3] self-skips.
    **GATE:** steering reconstruction must improve again, as it did at 4 → 3. If
    it does not, the capacity story is exhausted — report and stop.

    **Step 24 result (2026-09-12, `yaak/rmind/runs/ujef8lzw`,
    `model-ujef8lzw:v9`, 10 epochs / 1239 steps/epoch, uniform
    `channel_weights`, full val split n=27,676):**

    ```bash
    nix develop --command uv run python scripts/action_tokenizer_ceiling.py \
        --artifact yaak/rmind/model-ujef8lzw:v9 \
        --config-dir /home/alex/rmind/config \
        --experiment yaak/action_tokenizer/pretrain \
        --batches 100000 --device cuda
    ```

    **Per-field / per-chunk-step reconstruction L1** (normalized units):

    | field          | mean L1 |
    | -------------- | ------- |
    | acceleration   | 0.0163  |
    | steering_angle | 0.0096  |

    **Steering reconstruction L1, `q6ocue9a:v9` (3ch) → `ujef8lzw:v9` (2ch),
    per speed band:**

    | band (km/h) | n      | q6ocue9a steer | ujef8lzw steer | Δ          |
    | ----------- | ------ | -------------- | -------------- | ---------- |
    | 0–5         | 22,587 | 0.0125         | 0.0118          | **−5.6%**  |
    | 5–10        | 5,638  | 0.0249         | 0.0264          | +6.0%      |
    | 10–20       | 15,785 | 0.0193         | 0.0202          | +4.7%      |
    | 20–35       | 38,764 | 0.0087         | 0.0099          | +13.8%     |
    | 35–60       | 50,409 | 0.0051         | 0.0062          | **+21.6%** |
    | 60–130      | 32,873 | 0.0045         | 0.0048          | +6.7%      |
    | mean        | —      | 0.0088         | 0.0096          | +9.1%      |

    Per-channel share of decoded-distance `d_q(c)`: `acceleration` 54.1%,
    `steering_angle` 45.9% — a much less lopsided split than `turn_signal`'s
    38.9%-of-4 at Phase 1, i.e. the merge freed far less relative capacity
    than the "near-mutually-exclusive gas/brake" argument predicted.

    **GATE: NOT PASSED.** Steering regressed in 5/6 bands (only 0–5 km/h
    improved), mean +9.1%, with the 35–60 km/h band at +21.6% — well past
    the handoff's ~2% steering noise floor, so this is a real regression,
    not noise. Unlike the 4→3 transition (which improved every channel in
    every band), collapsing gas+brake into one `acceleration` channel does
    not clearly free RVQ capacity for steering here; `acceleration`'s own
    reconstruction L1 (0.0163) is higher than either individual pedal was
    under `q6ocue9a` (gas 0.0087, brake 0.0044 mean), consistent with it
    absorbing more of the codebook's budget rather than less. Per this
    section's own gate language, this is a stop-and-report point, not a
    default green light to proceed to step 25 — candidate next moves (not
    yet decided): non-uniform `channel_weights` favoring steering, more
    training epochs, revisiting the `Scaler` range, or concluding the
    capacity argument does not transfer to this merge and abandoning it.

01. **Prove the representation before touching drivr.** The end state is still
    "emit acceleration, change drivr"; this is sequencing, not a substitute. The
    open question in steps 23–24 is whether the *representation* is any good, and
    that is answerable entirely in-repo: decode
    `(acceleration_chunk, speed) → (gas, brake)` **inside** the model with a
    small head, reusing the Phase 2b `turn_signal_head` pattern verbatim
    (`patch_policy.py:913-936` — opt-in constructor arg, `None` reproduces
    current behaviour bit-for-bit, loss required when set), and keep
    `joint_actions` 4-channel. Then the arm is directly comparable to `v4mma4th`
    / `mv2qv1nd` on `predict/policy/score_l1` with **one** changed variable
    instead of five, drivr/ONNX/TRT/parity stay untouched, and it *measures* the
    accel→pedal map drivr would have to implement before that work is
    commissioned. If this comes back neutral — which is what Phase 2b's package
    delivered — the drivr rewrite proceeds with eyes open or not at all.

01. **The contract change.** Do this **first, before any index moves**: replace
    the positional channel convention with one named registry, and add an
    assertion that `offset_head.out_features == num_quantizers × codebook_size × action_horizon × tokenizer._action_features` (today a mismatch surfaces only
    as an einops failure at first forward). Sites:

    - *implicit tail-append, highest risk*: `patch_policy.py:936`, `:1417-1423`;
    - *explicit index 3*: `nn.py:361`, `patch_policy.py:1379`, `:1412`,
      `joint_policy.py:231`, `:250`;
    - *explicit gas=0 / brake=1*: `nn.py:355-356`, `patch_policy.py:1373-1374`,
      `joint_policy.py:227-228`, `:244-245`, `decoder_only_verify.py:76` and
      `:676-697` (including the `{gas_pedal, brake_pedal, steering_angle}`
      **string-set** control gate behind the 0/200 parity claim),
      `decoder_parity_orin.py:51-52`, `pedal_conflict_probe.py:49-50`,
      `patch_policy_eval.py:206,308,316`;
    - *derived widths*: `config/model/yaak/patch_policy/raw.yaml:151`
      (`1152 → 768`), and the **duplicated** `StackFields`/`ChunkFields` in
      `dinov2_dinowm_causal_3cam.yaml:48-55,86-92`.

    Pin the old artifact wherever it is still needed, as Phase 2a step 8 did —
    `control_transformer/{raw,policy_finetune}.yaml` already pin `m35hyvbp:v9` /
    `y74asdtd:v9` for `JointPolicyObjective`'s hardcoded 4-channel `predict()`.

    The `pedal_conflict_*` / `codebook_conflict_surface` probe family becomes
    **vacuous** — simultaneous gas and brake is structurally impossible in this
    representation — so retire it explicitly rather than leaving it to rot.
    `neighbor_smoothing_tau` goes stale a second time (4 → 3 already shrank mean
    decoded distance 0.069 → ~0.042) and is *already* an unresolved confound in
    the step-14 comparison. **drivr is unmapped**: scoping its controller is its
    own task, not part of this brief.

01. **Closed loop, staged so model and controller stay separable.**
    (a) **Controller replay parity first** — run the new drivr acceleration
    controller against *logged* acceleration from a known-good run and confirm it
    reproduces the pedal trace, with the model held fixed.
    (b) **Only then** let a new model drive it, at `benchmark.n_runs=3` (the
    config default), paired per scenario, never averaging the bimodal junction
    scenarios blind, with the MDE computed from the two arms actually being
    compared — variance is strongly model-dependent (per-arm sd 0.0116–0.0453).
    (c) **Log achieved-vs-commanded acceleration in sim** against the train-set
    distribution. The Phase 4 appendix already flags this for the RLS estimator
    and it applies verbatim: CARLA's longitudinal dynamics are not any real
    Niro's, so the commanded acceleration can land out of distribution in the
    very benchmark used to judge it.

### Verification notes for whoever executes this

`just generate-config` after any template edit. The `just test` baseline is 241
passed / 1 skipped, with pre-existing CUDA `control_transformer` /
`episode_builder` failures unrelated to this work. Scope ruff to the exact edited
files — the repo sets `fix=true` and `unsafe-fixes=true`, so a full-repo
`ruff check .` silently rewrites untouched files. The noise floor is brake 0.9% /
gas 1.0% / steering 2.0%; see "Statistics discipline" below.

New tests required:

- the 2-channel tokenizer round-trips and `_action_features == 2`;
- **the load-bearing one** — `turn_signal` lands where the contract says, not at
  whatever index the tail happens to reach (template:
  `tests/test_patch_policy.py:597`);
- the `offset_head`-width assertion fires on a deliberate mismatch;
- step 25's decode head at zero weight reproduces the baseline bit-for-bit
  (pattern: `test_trajectory_weight_zero_reproduces_baseline_bit_for_bit`).

Export parity is the real gate on any contract change: `just export-onnx` /
`decoder_only_export --verify`, then `decoder_only_verify.py` — whose
control-channel gate is a **string set** that must be updated, or it will report
parity against a channel set that no longer exists.

## Phase 4 — per-drive gas calibration, made deployable

**Read Phase 4.1 first.** It attacks the same finding (#3) by removing the
vehicle-specific part from the action space rather than feeding it in as a
conditioning feature. If 4.1 succeeds this phase is unnecessary; if 4.1 fails at
its step-23 data gate, this phase is untouched and still the route to #3.

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
01. **Never feed a whole-drive fit. Feed a causal running estimate** (see the
    appendix). Same code path offline and online, so the feature distribution
    matches between train, val and closed loop, and it is defined at tick 0.
01. **Gate it into the gas path only.** #12: `brake_gain` and `steer_gain` have
    split-half reliability 0.000 under every gate tried. #13's brake arms are
    *untested rather than refuted* — one seed of three blew up to 0.0403 against
    ~0.019 — so re-run those with ≥5 paired seeds before ruling them in or out.
01. **Do not add a per-vehicle or per-drive embedding.** #1 measured it making
    every field worse (+8.0% gas / +15.9% brake / +25.5% steering) and #12
    explains why: 69% of `gas_gain`'s spread is between drives of the *same*
    vehicle, so a static one-hot addresses at most a third of the effect while
    paying the full variance cost.

## Appendix — how the online estimator works

The forward model is the handoff's `LongitudinalDynamicsParams`
(`src/rmind/components/controller.py`):

```
dv_t  ≈  g·gas_t  −  b·brake_t  −  d·v_t  −  c
```

Write it as a linear regression with `θ = [g, b, d, c]` and, per tick,

```
x_t = [ gas_t, −brake_t, −v_t, −1 ]        y_t = v_{t+1} − v_t
```

**Recursive least squares with forgetting factor λ** tracks `θ` with a 4×4 state
and no history buffer — a handful of flops per tick:

```
k_t = P_{t-1} x_t / (λ + x_tᵀ P_{t-1} x_t)
θ_t = θ_{t-1} + k_t (y_t − x_tᵀ θ_{t-1})
P_t = (P_{t-1} − k_t x_tᵀ P_{t-1}) / λ
```

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
  `turn_signal` at index 3 in `{0, 0.5, 1.0}`, so `nn.py:340-368`'s unpacker
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
  decides it. **Status 2026-09-11:** the offline half of that gate ran
  (`mv2qv1nd`) and came back neutral, not beating — turn-signal accuracy is
  fine (0.939) but gas/steering are marginally worse. The rsim half has not
  been run. On the offline evidence alone Phase 2b is not a pass; it is a
  clean refactor with no measured policy cost or gain.

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
