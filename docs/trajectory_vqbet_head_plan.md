# Task brief: VQ-BeT trajectory head for `PatchPolicy`

Self-contained task brief, written for an agent with **no prior context**.
Everything needed is here or behind the `file:line` references. Branch:
`exp/cond-t2`. The tokenizer work this builds on is in
`docs/trajectory_tokenizer_codebook_plan.md` (read its §3 and §10 only).

## 1. Why this work exists

`PatchPolicy` (`src/rmind/models/patch_policy.py`) predicts the future ego
trajectory with `trajectory_head`. That head is a direct-regression MLP with 5
hypotheses, trained winner-takes-all, plus a `mode_head` classifier that
learns which hypothesis the oracle would pick (`patch_policy.py:219-253`,
`942-967`). The plan is to replace it with a **VQ-BeT head**, the same design
the action head already uses (`patch_policy.py:906-1045`, mirroring
`components/objectives/joint_policy.py`):

- classify one code per residual-VQ level of the frozen trajectory tokenizer,
- predict a code-conditioned continuous offset,
- output `tokenizer.invert(codes) + offset`.

The tokenizer is frozen as **`model-403b4rrh:v26`**. Level 0 of its codebook
is heavily skewed: about 94% of clips go to 8 "straight, at speed X" codes,
and only about 5.8% to the 4 turn codes (§3.2). The project owner's concern
is that turns are too rare for a plain cross-entropy classifier, and that
class weighting or focal loss may be needed. **The second job of this brief
is to measure that concern before building it into the full run** (§4).

## 2. Decisions already made: do not revisit

- **Tokenizer: the yaw-augmented retrain of `403b4rrh`** (experiment
  `trajectory_tokenizer/pretrain_yaw`, same recipe, 4 levels × 16 codes,
  `representation=absolute`; run id and version in §7). Decided by the
  project owner on 2026-10-15, after step 0 showed that `model-403b4rrh:v26`
  breaks on yaw-rotated targets (§5 step 0). The codebook-size arms
  (B1/C8/C32/D20) of the tokenizer brief stay SHELVED. Do not resize the
  tokenizer.
- **Code loss: `FocalLoss` with γ = 2**, the action head's loss, at every
  level. Decided by the project owner on 2026-10-15. Class weighting and
  logit adjustment are not used, and the §4 loss probe is SHELVED.
- **Target:** the existing per-frame dead-reckoned `trajectory_target`
  `(b, t, 5, 3)` (`patch_policy.py:318`, `components/nn.py`
  `TrajectoryTarget`), the same one the tokenizer was trained on. Do not
  change the horizon or the step.
- **Opt-in, default off.** With the new heads unset the model must reproduce
  current behaviour bit for bit. That is the repo's rule for every
  `PatchPolicy` head (see `trajectory_head`, `turn_signal_head`).
- **The new heads replace `trajectory_head` + `mode_head` and cannot be
  combined with them.** Configuring both is a `ValueError` in `__init__`.
  Keeping the MLP head alongside would muddy the trunk comparison in §5
  step 5.
- **The existing `trajectory_weight` knob scales the new losses too**
  (`code_*` + offset). Its gate semantics stay as they are: at 0.0 the heads
  are built and exercised but contribute no gradient.
- **Decode with argmax per level** for metrics, predict and the
  counterfactual probes, matching commit `bc9b4b85` ("argmax everywhere").
  Sampled decoding is out of scope.
- **Training-time and offline only.** Like `trajectory_head` today, the new
  head does not feed `joint_actions`, ONNX export or serving (§9).

## 3. Established facts (cite these; don't re-derive them)

### 3.1 Tokenizer API (`src/rmind/models/trajectory_tokenizer.py`)

| call             | maps                                            | line |
| ---------------- | ----------------------------------------------- | ---- |
| `forward(poses)` | `(*b, 5, 3)` poses → `(*b, 4)` codes            | :185 |
| `decode(z_q)`    | latent → `(*b, 5, 3)` absolute poses, θ wrapped | :192 |
| `invert(codes)`  | `(*b, 4)` codes → `(*b, 5, 3)` poses            | :197 |

- Poses are `(x, y, θ)`: x lateral, y forward, θ in radians, in meters.
  Normalization buffers load with the checkpoint.
- `quantizer.num_quantizers == 4`, `quantizer.codebook_size == 16`.
- Load with `TrajectoryTokenizer.load_from_wandb_artifact(...)`, the same way
  `PatchPolicy` already loads its frozen `tokenizer` / `goal_encoder`
  (`patch_policy.py:351-366`: `requires_grad_(False).eval()`, re-forced to
  eval in `train()` at :561-566).
- Prefix decoding (setting later levels to `-1`) works and is unit-tested
  (`tests/test_trajectory_tokenizer.py`). It isn't needed here.

### 3.2 Level-0 code distribution (val, 27,676 clips)

From `outputs/traj-audit/audit_403b4rrh_v26.json` (`usage.q0`,
`level0_modes`), produced by `src/rmind/scripts/trajectory_tokenizer_audit.py`:

| codes                     | share                                                        | meaning                                            |
| ------------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| 4, 3, 15, 13, 8, 1, 14, 9 | 20.4 / 16.0 / 15.6 / 12.0 / 11.1 / 10.4 / 5.7 / 3.1 % (≈94%) | straight (\|θ\| < ~4°), speed bins ~0.4–26 m/s     |
| 5 / 6                     | 2.3% / 1.4%                                                  | turns at ~4 m/s, final heading ≈ −33° / +30°       |
| 11 / 0                    | 1.1% / 1.0%                                                  | gentler turns at ~10–11 m/s, ≈ −12° / +17°         |
| 12, 7, 10                 | 4, 1 and 1 clips                                             | ~180° final heading: ground-truth heading glitches |
| 2                         | 0                                                            | unused                                             |

- Level-0 class-mean ADE is 0.82 m (one global mean: 5.85 m). Levels 0+1
  give 0.25 m. Full decode gives 0.114 m.
- Deeper levels also have near-empty codes (for example q1 codes 4, 12 and 15
  hold 1–3 clips each).
- **All of this is val only (5 drives).** No train-set histogram exists yet.
  §5 step 3 produces one.

### 3.3 Machinery that already exists: reuse it, don't rebuild it

- **VQ-BeT action head in `PatchPolicy`:**

  - `_heads` (:917)
  - `_gather_offset` (:969, shape-generic, works for `(b, t, g, c, a)`)
  - `_offset` with the optional `offset_scale` tanh bound (:975)
  - the teacher-forced offset at the ground-truth codes (`_compute_metrics`,
    :1102-1160)

  Mirror that structure. The trajectory offset table is `(g, c, 15)`, i.e.
  5 poses × 3.

- **`FocalLoss`** (`components/loss.py:17`) is the action code head's loss
  (`config/model/yaak/patch_policy/raw.yaml:178-181`). With `gamma=0.0` it
  equals cross-entropy exactly, so one loss class covers CE and focal arms.

- **Logit adjustment already exists for `ControlTransformer` only:**

  - `LogitBiasCrossEntropyLoss` / `LogitBiasFocalLoss` (`loss.py:64-87`) add
    `log prior` to the logits during training only.
  - `callbacks/logit_bias.py` sets the prior from the train set.
  - The callback is typed to `ControlTransformer` and reads its
    `episode_builder`, so it **cannot** be pointed at `PatchPolicy`. §5
    step 2 does the equivalent with a precomputed prior.

- **Pose loss and metrics:**

  - `winner_takes_all_pose_l1(_components)` (`loss.py:217-298`) with `Q = 1`
    is a plain single-trajectory L1(xy) + wrapped-heading loss, using the same
    `xy_weight` / `heading_weight` the experiment config pins
    (`dinov3.yaml:48-49`).
  - Use it for the offset loss so the magnitudes are comparable with today's
    `trajectory` loss.

- **Counterfactual probes:**

  - `_counterfactual_plan` (`patch_policy.py:897`) calls
    `expected_plan(_predict_trajectory, _predict_mode)`
    (`components/counterfactual.py:57`).
  - These produce `val/policy/metric/goal_sensitivity_*` and `speed_echo*`,
    which the speed/goal-conditioning work relies on, so they must keep
    working with the new head.

### 3.4 Training-time yaw augmentation rotates the target

- The current causal arms (`*_r1a_yaw.yaml` and its children) apply
  `YawShiftCrop` (`components/augment.py:60`) to 50% of training clips. It
  rotates `trajectory_target` by up to about ±4.7° (θ − ψ, xy rotated).
- It is a no-op in eval (`augment.py:138`).
- The tokenizer was trained **without** this rotation. Rotated straight
  trajectories may decode worse or land in the gentle-turn codes 11/0. §5
  step 0 measures this before the full run relies on it.

### 3.5 Baseline runs and cost

- **Trunk / feature source:**
  - T1 `uqvyzh38` (`alex-tmp` project). Its settings: goal dropout 0.1,
    speed dropout 0.5, a 4-epoch screen, `lr_total_steps 95800`, warmup 7660.
  - It is the matched baseline for the `exp/cond-t2` arm
    (`/nasa/alex/docs/speed_goal_conditioning_rmind_plan.md` §T1/§T2b).
  - Read its resolved config from W&B. **Confirm `trajectory_weight > 0` and
    a configured `mode_head`** before using its features.
  - Pin an explicit `:vN` at epoch 4 (step ≈ 95.7k). Never use `:latest`:
    it is not the version that was benchmarked (see the owner's training
    gotchas).
- **Cost:** a full 4-epoch `PatchPolicy` screen takes about 1–2 days on one
  GPU. Comparing the 6 loss arms × 3 seeds of §4 in full runs is
  unaffordable. They run as a **frozen-feature probe** instead (§5 step 3),
  and only the winner gets a full run.

### 3.6 Environment (aboutblank)

Read §3.6 of `docs/trajectory_tokenizer_codebook_plan.md` before running
anything. It covers:

- the local-`.venv` test command,
- the Docker image and GPU flags,
- `resume=true`,
- the commit hook shims,
- scoping Ruff to the files you edit.

The rule that matters most here:

- **Instantiating a datamodule wipes its sample store, and `resume=true` does
  not protect it.**
- **Val:** instantiate only `cfg.datamodule.val`.
- **Train (needed in step 3):** point the script at a **copy** of the train
  cache directory (`cp -r` into `outputs/`). Never point it at the live
  store a training container is using.

## 4. Experiment design: is level-0 imbalance a problem? (SHELVED 2026-10-15)

> The project owner chose focal loss (γ = 2) without running the probe. This
> section is kept as the rationale, and as the design to pick up if the full
> run (§5 step 5) shows poor turn recall or a high false-turn rate.

### 4.1 Why imbalance may not matter, and why it might

- **Why it may not:** the turn codes are rare **marginally** (5.8%), but the
  classifier is conditioned on the camera, the route waypoints and speed. On
  a frame whose route turns, the turn codes are not rare. Imbalance only
  hurts where the features fail to separate the classes, or where the easy
  majority dominates the gradient.
- **The majority is easy here.** Speed is a near-copy input (the T0 finding
  "speed copy universal": echo ≈ 4.4 m vs 4.63 m for a pure copier). The 8
  straight speed-bin codes should therefore be almost free to classify. That
  is the regime focal loss was designed for: it down-weights easy examples
  without changing what the classifier believes.
- **Risk in the other direction.** Inverse-frequency class weighting changes
  the classifier's decision rule: weights of 1/0.204 ≈ 5 vs 1/0.010 ≈ 98 are
  a ~20× ratio. On ambiguous frames it will **over-predict turns**, and a
  phantom turn on a straight road is exactly the lateral-drift failure the
  r1a/r1b arms are trying to fix.
- **Logit adjustment** (Menon et al., 2021, arXiv:2007.07314) gets
  balanced-error behaviour while keeping inference on raw logits. It is the
  principled middle option, and the repo already uses it for
  `ControlTransformer` (§3.3).

### 4.2 Arms (frozen-feature probe; all levels use the same loss)

| arm     | `gamma` | logit adjustment τ | class-weight power p | answers                                                           |
| ------- | ------- | ------------------ | -------------------- | ----------------------------------------------------------------- |
| P-CE    | 0       | 0                  | 0                    | plain baseline: are turns actually under-predicted?               |
| P-F     | 2       | 0                  | 0                    | the house default (the action head's loss)                        |
| P-CE-LA | 0       | 1                  | 0                    | logit-adjusted CE                                                 |
| P-F-LA  | 2       | 1                  | 0                    | focal + logit adjustment                                          |
| P-CE-W½ | 0       | 0                  | 0.5                  | √-inverse-frequency class weights                                 |
| P-CE-W1 | 0       | 0                  | 1                    | full inverse-frequency weights (the literal class-weighting idea) |

- Class weights are `w_c = prior_c^(−p)`, normalized so their mean under the
  prior is 1. The training loss is then the per-sample-weighted mean.
- The logit adjustment adds `τ · log prior` to the training logits only.
  Inference uses raw logits.
- `prior` is the train-set code histogram per level (§5 step 2), with +1
  smoothing as in `callbacks/logit_bias.py:101`.
- **3 seeds per arm.** σ is the across-seed std of a metric within an arm,
  pooled over arms.

### 4.3 Metrics (val, every frame, argmax decode + offset)

1. **ADE / FDE / final-heading error** of the decoded trajectory, over all
   frames.
1. **Turn-frame ADE / FDE**: frames whose ground-truth final heading has
   |θ₅| > 10°.
1. **False-turn rate**: among frames with ground-truth |θ₅| < 4°, the share
   whose **decoded** |θ₅| > 10°. This is the cost of over-predicting turns.
   It is defined on decoded headings, not codes, so it stays comparable if
   the code assignment shifts.
1. **Level-0 calibration**:
   - recall per code for the 4 turn codes (0, 5, 6, 11),
   - predicted turn-code share against the ground-truth share (5.8%),
   - level-0 NLL,
   - 15-bin ECE.
1. Accuracy per level (diagnostic).

Report every metric as mean ± std over the 3 seeds.

### 4.4 Decision rule

- **Default: P-F** (focal γ = 2, no adjustment). It is the house default and
  the cheapest to justify.
- **Switch to another arm only if all three hold:**
  - its turn-frame ADE beats P-F's by more than 2σ,
  - its false-turn rate is no more than 1 percentage point (absolute) above
    P-F's,
  - its all-frame ADE is no more than 2σ worse than P-F's.
- If P-CE already has turn recall ≥ 0.8 and a predicted turn share within
  ±1 pp of the ground truth, **record that imbalance is not a problem.** Then
  keep P-F, unless P-CE is strictly better on metrics 1–3.
- If no arm clears the rule, keep P-F and say the result is inconclusive.

## 5. Work steps (in dependency order; each leaves the tree green)

### Step 0: tokenizer sanity checks on policy targets (cheap, val only)

> **DONE 2026-10-15: gate FAILED for `v26`.** Script:
> `src/rmind/scripts/trajectory_tokenizer_yaw_audit.py`, run on the
> tokenizer val set (27,676 clips; the same `TrajectoryTarget`). Output:
> `outputs/traj-audit/yaw_403b4rrh_v26{,_masked}.json`. ψ was drawn as in
> training: a 32-frame walk, one random frame, |ψ| p50 2.2°, p90 4.7°.
>
> |                              | ADE        | p95 ADE | straight → turn code                     |
> | ---------------------------- | ---------- | ------- | ---------------------------------------- |
> | unrotated                    | 0.114 m    | 0.29 m  | 0%                                       |
> | rotated, `v26` as-is         | **10.2 m** | 23.8 m  | 88% → glitch codes                       |
> | rotated, dead feature masked | 0.28 m     | 0.73 m  | 3.1% (9.4% at 8–15 m/s; 13% at ψ = 4.7°) |
>
> **Cause:** pose 0's `x` is identically 0 in the dead-reckoned target,
> because the first segment integrates along the anchor heading. Its
> normalization std is therefore clamped to 1e-6. Any rotation makes
> x₀ = −y₀ sin ψ ≠ 0, the input explodes (z ≈ 10³–10⁵), and every rotated
> clip collapses to one code tuple. The error is flat from ψ = 0.5° upwards.
> Even with the feature masked, the gate still fails: ADE 2.5× worse, and the
> decoder can never output x₀ ≠ 0.
>
> **Fix (owner's choice):** retrain the tokenizer with
> `YawRotateTrajectory` (`components/augment.py`), the label-only half of
> `YawShiftCrop` with the same sampler, margin and pixel realisation, at
> p = 0.5. Experiment: `trajectory_tokenizer/pretrain_yaw`. Before freezing
> it, re-run both audits on the new checkpoint and require:
>
> - unrotated ADE within ~0.015 m of `v26`,
> - rotated ADE (training marginal) < 0.2 m,
> - straight → turn under 2%,
> - the level-0 mode table still has the 4 turn modes.
>
> The original instructions follow.

Use the audit script's loading code
(`trajectory_tokenizer_audit.py:60-97`) or a new small script, with the
policy experiment's **val** datamodule only.

1. **Matching histograms.** Tokenize the policy val `trajectory_target`.
   Check that its level-0 histogram has the same shape as §3.2. The drives
   differ, so the shares won't match exactly, but turn codes should be a
   single-digit percentage.
1. **Rotation robustness (§3.4).**
   - Rotate the val targets with ψ ~ U(−4.7°, 4.7°). Use the same rotation
     `YawShiftCrop` applies; import its `rotate_ego_frame`, don't rewrite it.
   - Report the full-decode ADE, and the share of straight clips
     (|θ₅| < 4° before rotation) that move to codes 0/11 after rotation.
   - **Gate:** if the rotated ADE is > 0.2 m, or more than 10% of straight
     clips move to the turn codes, **stop and report to the owner.** The fix
     (retraining the tokenizer with rotation augmentation, or dropping the
     code loss on augmented clips) is their call.
1. **Glitch count.** Count val frames with |θ₅| > 90°, and record 3 examples
   (drive and frame). Masking them is §5 step 1. Finding the root cause
   (sensor glitch vs ±π wrap) is out of scope (§9).

### Step 1: `PatchPolicy` wiring

New opt-in constructor arguments, all defaulting to `None` / off:

- `trajectory_tokenizer`: a frozen `TrajectoryTokenizer`, handled like
  `tokenizer` (`requires_grad_(False).eval()`, re-forced in `train()`).
- `trajectory_code_head`: features → `(g·c)` logits.
- `trajectory_offset_head`: features → `(g·c·15)`.
- `losses["trajectory_code"]`: required when the heads are set; a
  `FocalLoss`.
- `losses["trajectory_offset"]`: required; a `WinnerTakesAllPoseLoss` used at
  `Q = 1`, or a thin single-trajectory wrapper around
  `winner_takes_all_pose_l1`.
- `trajectory_code_prior`: `(g, c)` probabilities, or `None`. Stored as a
  buffer, and also logged in hparams.
- `trajectory_logit_adjustment_tau: float = 0.0`
- `trajectory_class_weight_power: float = 0.0`
- `trajectory_masked_codes: tuple[int, ...] = ()`: level-0 codes forbidden at
  train and inference time.

Behaviour:

- **Validation in `__init__`:**
  - the code and offset heads must be set together,
  - neither may be combined with `trajectory_head` / `mode_head`,
  - `tau > 0` or `power > 0` requires `trajectory_code_prior`,
  - `losses["trajectory_code"]` must be a `FocalLoss` (the only code loss
    that takes per-sample weights, see below).
- **Targets:** `trajectory_tokenizer(trajectory_target)` under `no_grad`
  gives `(b, t, 4)` codes, computed once per step.
- **Masking:**
  - Set level-0 logits of `trajectory_masked_codes` to `-inf` everywhere:
    loss, argmax and predict.
  - Drop frames whose ground-truth level-0 code is masked from **all**
    trajectory losses: code levels 0–3 and the offset. Their deeper codes and
    targets are garbage too.
  - Log the dropped share as `trajectory_masked_frac`.
- **Code loss:** for each level q, apply the loss to
  `logits_q + τ · log prior_q` (τ = 0 → raw logits). Apply per-sample weights
  `w[target]` when the power p > 0.
  - Add an optional `sample_weight` argument to `FocalLoss.forward` (a
    weighted mean). It must be bit-for-bit backward compatible when `None`,
    and its docstring must say so.
- **Offset:**
  - Teacher-forced at the ground-truth codes, as the action head does
    (:1145-1156).
  - `pred = invert(gt_codes) + gather(offsets, gt_codes)` in pose space,
    (meters, radians).
  - Re-wrap θ after adding the offset.
  - The existing `offset_scale` bound is **not** reused. It is tuned for
    normalized action units, not meters.
- **Scale:** multiply both losses by `trajectory_weight`.
- **Metrics:** log the §4.3 metrics 1–5 under the existing `policy/metric/`
  prefix: `trajectory_ade`, `trajectory_fde`, `trajectory_turn_ade`,
  `trajectory_false_turn_rate`, `trajectory_code_{q}_acc`,
  `trajectory_turn_recall`, `trajectory_turn_share_pred`, plus the existing
  `trajectory_loss_xy` / `_heading` names where they apply. Keep ECE out of
  the training step; it is probe-only.
- **`predict_step`:** emit `result["trajectory"]` with the same keys as
  today (:1697-1735):
  - `prediction`: `(b, 1, p, 3)`, a single hypothesis, so existing
    consumers' `Q` axis still works,
  - `ground_truth`,
  - plus `codes` and `level0_probs`.
- **Counterfactual:** `_counterfactual_plan` dispatches to argmax decode +
  offset when the VQ-BeT trajectory head is configured. Relax the
  `counterfactual_val_frames` validation (:412) to accept either head.

Tests (`tests/test_patch_policy.py`, following the existing trajectory/mode
tests at :1238-1500):

- the new heads are absent by default, and the default model is bit-for-bit
  unchanged (follow `test_trajectory_weight_zero_reproduces_baseline_bit_for_bit`);
- each validation error fires;
- gradients reach both heads, and the frozen tokenizer gets none;
- `trajectory_weight = 0` gives a zero loss contribution;
- masked codes never win argmax, and masked frames contribute no loss;
- τ > 0 changes the training loss but not argmax at inference;
- `p = 0` with `sample_weight=None` matches the old `FocalLoss` exactly
  (`tests/test_loss.py` or wherever `FocalLoss` is tested);
- `predict_step` shapes;
- the counterfactual metrics run with the new head.

Use a tiny randomly initialized `TrajectoryTokenizer`, as
`tests/test_trajectory_tokenizer.py` does.

Configs:

- Add the head and loss blocks to `config/model/yaak/patch_policy/raw.yaml`,
  all defaulting to `null` / off.
- Add a new experiment
  `config/experiment/yaak/patch_policy/<T1-or-T2b parent>_vqbet_traj.yaml`
  that sets them, inheriting from the baseline run's experiment (step 0
  tells you which).
- Head widths: code head `[1024, 1024, 64]` and offset head
  `[1024, 1024, 960]` (4 × 16 × 15), the same MLP family as the action heads
  (`raw.yaml:141-151`).

### Step 2: train-set code prior (REDUCED 2026-10-15)

> With focal loss and no class weighting or logit adjustment, no prior is
> needed. `trajectory_code_prior`, `trajectory_logit_adjustment_tau` and
> `trajectory_class_weight_power` are **not built**. Set
> `trajectory_masked_codes` to the level-0 codes that the new tokenizer's
> codebook audit (`trajectory_tokenizer_audit.py`) shows holding glitch
> outliers (|mean final heading| > 90°). Do not mask codes that are merely
> empty on val. The train-histogram pass below is skipped.

- Tokenize the train-set `trajectory_target` on the **copied** train cache
  (§3.6) and write the per-level histogram, raw counts and shares, to
  `outputs/traj-vqbet/prior_403b4rrh_v26.json`.
  - This closes the "train-set histogram" open item in the tokenizer brief's
    §9.
- Paste the `(4, 16)` shares into the experiment YAML as
  `trajectory_code_prior`, so the run is self-contained and versioned.
- Set `trajectory_masked_codes` to the level-0 codes with train share
  < 1e-4. Expect `[2, 7, 10, 12]`, but use what the data says; code 2 may
  turn out to be live in train.

Steps 2 and 3 share the train-set pass, so do them in one script run.

### Step 3: frozen-feature probe (answers §4) (SHELVED 2026-10-15, see §4)

Add `src/rmind/scripts/trajectory_vqbet_probe.py` with two subcommands.

**`extract`:**

- Load the pinned baseline checkpoint (§3.5) in eval mode, with no
  augmentation.
- Run `_features` → readout features `(b, t, d)` for:
  - the whole val set,
  - about 200k train frames, taking random clips and every 4th frame: frames
    within a clip are highly correlated.
- Save features (fp16), `trajectory_target`, and the tokenizer codes to
  `outputs/traj-vqbet/features/`.
- The train pass uses the copied cache (§3.6).

**`fit`:**

- Train only the two heads on the cached features for each §4.2 arm × 3
  seeds.
- Reuse the Step 1 `PatchPolicy` methods for the losses, masking, logit
  adjustment and decoding. Don't reimplement them: the probe must measure
  the code that the full run will train.
- Compute the §4.3 metrics on val and write one JSON per arm and seed, plus a
  summary table.
- Use AdamW, run to convergence on train (early-stop on a held-out 10% of the
  train features, not on val), and fix the epoch budget across arms.

Caveat: the trunk's features were shaped by the MLP head's training, not by
this head's. Treat the probe as a ranking of the losses, not as a forecast
of the full run's numbers.

### Step 4: commit and push

- Commit steps 0–3: code, tests, configs, the probe script, and this brief.
- Include the probe summary table in `docs/trajectory_vqbet_head_results.md`
  (§5 step 6).
- `just train` requires a committed, pushed tree.

### Step 5: full run with the chosen loss

- Launch one 4-epoch screen of the step 1 experiment with the §4.4 winner,
  using the baseline run's settings, `trajectory_weight` and schedule
  (`lr_total_steps 95800`, warmup 7660; set these explicitly).
- Launch it the same way the baseline was launched: Docker, a pinned image
  tag, `WANDB_GIT_COMMIT`, a `trap … chown` on exit.
- Compare at matched steps (epoch 4, step ≈ 95.7k) against the baseline's
  MLP head. Use the comparable metrics (§8 row 1):
  - argmax ADE/FDE against the MLP's **expected-plan** ADE (the mode-weighted
    plan rsim serves) and its **mode-argmax hypothesis** ADE,
  - turn-frame ADE, false-turn rate,
  - `goal_sensitivity_yaw`, `speed_echo`,
  - the action-head metrics, to check the trunk did not regress.

### Step 6: write up

Write `docs/trajectory_vqbet_head_results.md` with:

- the step 0 gate results,
- the train prior and the masked codes,
- the probe table (§4.3 × §4.2, mean ± std),
- the decision under §4.4 and why,
- the full-run comparison.

Then mark this brief's status line DONE with the date.

## 6. Tests and verification

- Run `tests/test_patch_policy.py` and the `FocalLoss` tests with the local
  `.venv` command (tokenizer brief §3.6). All must pass, including the
  pre-existing trajectory/mode tests.
- Run Ruff on the edited files only, and check that `ty` output doesn't
  mention them.
- **Probe check before trusting it:** decode the **ground-truth** codes with
  zero offset through the probe's decode path. Its val ADE must equal the
  tokenizer's own full-reconstruction ADE on the same frames (≈ 0.11 m on
  the tokenizer val set; recompute on the policy val set). A mismatch means
  the probe decodes differently from the tokenizer.
- **Smoke test the full run:** `just train-debug`-style, with 3 episodes,
  no compile and W&B off, on the new experiment before the real launch.

## 7. Status

- 2026-10-15: brief written. Focal loss chosen and §4 shelved (owner).
  Step 0 failed for `v26`; `YawRotateTrajectory` added and the
  `pretrain_yaw` tokenizer run launched (run id: see the commit/launch notes
  below).

## 8. Risk register

| risk                                                                  | effect                                                                                              | mitigation                                                                                           |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Comparing the VQ-BeT argmax ADE to the MLP's logged `trajectory` loss | that loss is winner-takes-all **min over 5 hypotheses**, an oracle; the comparison flatters the MLP | compare against expected-plan and mode-argmax ADE only (step 5)                                      |
| Class weighting over-predicts turns                                   | phantom turns on straight roads, i.e. lateral drift                                                 | false-turn rate is a gating metric in §4.4, not a diagnostic                                         |
| The prior comes from val instead of train                             | wrong logit adjustment and weights                                                                  | step 2 computes it on train; val is used only for metrics                                            |
| Instantiating the train datamodule                                    | wipes the live train sample store                                                                   | work on a copied cache (§3.6)                                                                        |
| Yaw augmentation produces rotated targets the tokenizer never saw     | mislabeled codes on 50% of training clips                                                           | step 0 gate                                                                                          |
| Glitch clips (~180°) as code targets                                  | garbage labels; rare codes learn nonsense                                                           | `trajectory_masked_codes` + drop those frames from all trajectory losses                             |
| Levels are predicted independently (as in the action head)            | inconsistent level combinations (a level-1 residual meant for a different level 0)                  | accepted, same as the action head; an autoregressive level head is a §9 follow-up if step 5 shows it |
| The probe's frozen features are biased toward the MLP head            | probe ranking differs from the full-run ranking                                                     | the probe ranks losses only; step 5 is the real test                                                 |
| Using `:latest` checkpoints                                           | not the benchmarked version                                                                         | pin `:vN` everywhere (§3.5)                                                                          |
| An unset LR schedule for a short run                                  | the warmup eats the run                                                                             | set `lr_total_steps` and warmup explicitly (step 5)                                                  |
| The val set is 5 drives                                               | turn-frame metrics rest on ~1.6k turn clips                                                         | report seed std; read differences under 2σ as noise                                                  |

## 9. Out of scope (recorded so nobody drops it silently)

- **Serving and export:** ONNX, rsim and the trajectory → action controller
  path. They need their own brief once step 5 shows a win offline.
  Closed-loop validation is the real gate: round 6 found an unweighted MLP
  trajectory head trending **negative** on route completion.
- **Root cause of the ~180° heading glitches**: a sensor glitch, or a ±π
  wrap in `TrajectoryTarget` / dead reckoning. A wrap bug would also corrupt
  the current MLP head's target. It's worth a separate look; this brief only
  masks the glitches.
- Sampled decoding, temperature, and a min-of-K evaluation over the top joint
  codes.
- Autoregressive per-level code prediction.
- Neighbour-aware label smoothing (`neighbor_smoothing_tau`) for trajectory
  codes. Decoded-trajectory distance is a natural metric for it; try it after
  §4 if turn confusion turns out to be among the turn codes rather than
  between turns and straights.
- Any change to the tokenizer: its size, levels, threshold, or retraining
  with rotation augmentation (unless the step 0 gate fails, in which case
  the owner decides).
