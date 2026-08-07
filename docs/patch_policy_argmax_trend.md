# Argmax (deployment) validation trend for the causal patch-policy arms

Companion to `patch_policy_causal_offline_val.md`, which established the method and
the epoch-0 numbers. This document answers: **is either causal arm actually getting
worse, or does it only look worse because every logged metric is sampled-decode
while `PatchPolicy.load_for_export` serves `sample_codes=False` (argmax)?**

Runs under investigation:

| run                   | name              | host                | commit    | state at audit | epoch |
| --------------------- | ----------------- | ------------------- | --------- | -------------- | ----- |
| `yaak/rmind/do8m9ot8` | vague-shape-587   | kitkat (RTX 5090)   | `815d4d2` | step 162.3k    | 2     |
| `yaak/rmind/p8d6fxao` | sage-universe-592 | sisyphos (RTX 5090) | `231ede7` | step 107.9k    | 1     |

## 0. Headline: the two runs are the SAME run

`p8d6fxao` is not an independent arm. It is a **bit-exact replica** of `do8m9ot8`.

- `231ede7` is `815d4d2` + one commit that adds only
  `docs/causal_patch_policy_architecture.svg` (209 lines, docs-only).
  **The training code at the two commits is identical.**
- Both were resumed from the *same* file: `do8m9ot8` from
  `~/Code/rmind-rqv/ckpt_ep0_step80797.ckpt`, `p8d6fxao` from
  `~/Code/rmind-rqv/scratchpad/art_do8m9ot8_v0/model.ckpt` — both are
  `model-do8m9ot8:v0` = `epoch=0-step=80797.ckpt`.
- Their wandb configs differ in **6 of 4890 keys**, all of them absolute-vs-relative
  path strings (`paths.rbyte.cache`, the two `run_folder`s, the two `cache_dir`s,
  `ckpt_path`). Same `seed: 1337`, same `val_check_interval: 0.25`, same batch 24,
  same label smoothing, same everything else.
- Both ran on an RTX 5090, so even the kernels match.

Consequence, and the proof: at the one `global_step` where both have logged a val
pass, **every logged val float is identical to the last bit** — including
`offset_sampled_recon`, which is a *multinomial draw* and therefore only reproduces
under identical RNG state:

| key @ step 100995                             | do8m9ot8              | p8d6fxao              |
| --------------------------------------------- | --------------------- | --------------------- |
| `val/loss/total`                              | 5.581413745880127     | 5.581413745880127     |
| `val/policy/loss/code_0`                      | 1.3063782453536987    | 1.3063782453536987    |
| `val/policy/loss/code_1`                      | 1.538063883781433     | 1.538063883781433     |
| `val/policy/loss/code_2`                      | 1.1232223510742188    | 1.1232223510742188    |
| `val/policy/loss/code_3`                      | 1.606469988822937     | 1.606469988822937     |
| `val/policy/loss/offset`                      | 0.0072804708033800125 | 0.0072804708033800125 |
| `val/policy/metric/offset_sampled_recon`      | 0.07885747402906418   | 0.07885747402906418   |
| `val/policy/metric/offset_sampled_recon_last` | 0.08005140721797943   | 0.08005140721797943   |
| `val/policy/metric/code_partial_window`       | 1.388339877128601     | 1.388339877128601     |
| `val/policy/metric/code_full_window`          | 1.3981165885925293    | 1.3981165885925293    |
| `val/policy/metric/offset_partial_window`     | 0.007262369617819786  | 0.007262369617819786  |
| `val/policy/metric/offset_full_window`        | 0.007296448107808828  | 0.007296448107808828  |

14 of 14 metric fields identical; only wandb's internal `_step` differs (4298 vs 811,
because `do8m9ot8` carries the epoch-0 history and `p8d6fxao` started its log at 0).

`p8d6fxao` therefore contains **zero information** that `do8m9ot8` does not already
have, and it is the slower of the two: since the shared resume point it has done
27.3k steps in 13.33 h (**2048 steps/h** on contended sisyphos) against `do8m9ot8`'s
81.7k steps in 26.5 h (**3083 steps/h** on kitkat), i.e. 1.5x slower.
(`do8m9ot8`'s wandb `_runtime` of 54.20 h is cumulative over both segments of the
same run id — 27.6 h for epoch 0 plus 26.5 h since the resume — not the resume
segment alone.) Every argmax measurement below is made on `do8m9ot8`
checkpoints and applies verbatim to `p8d6fxao` at the matching step.

## 1. Checkpoint inventory — only two checkpoints exist in total

`ModelCheckpoint(every_n_epochs=1, save_on_train_epoch_end=true)` with the default
`save_top_k=1`, and no `save_last`. So one artifact per finished epoch, nothing
mid-epoch.

| artifact            | file                       | step   | epoch           |
| ------------------- | -------------------------- | ------ | --------------- |
| `model-do8m9ot8:v0` | `epoch=0-step=80797.ckpt`  | 80797  | 0               |
| `model-do8m9ot8:v1` | `epoch=1-step=161594.ckpt` | 161594 | 1               |
| `model-p8d6fxao:*`  | —                          | —      | **none logged** |

`p8d6fxao` resumed *at* the epoch-0 boundary, so its first checkpoint will only be
written at step 161594 (epoch-1 end); it is at 107.9k, i.e. ~7.5 h away. A disk
sweep of `~/Code/rmind-rqv-resume` and `~/Code/rmind-rqv` on sisyphos found no
`last.ckpt` or mid-epoch checkpoint for it. **There is no p8d6fxao checkpoint to
evaluate, and by §0 there is no need for one.**

So the per-checkpoint argmax table below spans the two checkpoints that exist. The
sampled-decode val curve (4 points, `val_check_interval: 0.25`) fills in the trend
between them.

## 2. The logged (sampled) val curve — do8m9ot8, epoch 1

All four val points fall in epoch 1; epoch 0 logged none (see the companion doc for
why the `val_check_interval: 1.0` modulo never fired).

| step                 | code mean (smoothed) | offset (teacher-forced) | sampled recon | code partial 0-14 | code full 15-31 | full vs partial | val/loss/total |
| -------------------- | -------------------- | ----------------------- | ------------- | ----------------- | --------------- | --------------- | -------------- |
| 80797 (offline, ep0) | 1.3709               | 0.007224                | 0.0779        | 1.3682            | 1.3732          | +0.4%           | —              |
| 100995               | 1.3935               | 0.007280                | 0.07886       | 1.3883            | 1.3981          | +0.7%           | 5.581          |
| 121194               | 1.5035               | 0.007624                | 0.08175       | 1.4890            | 1.5162          | +1.8%           | 6.021          |
| 141393               | 1.5570               | 0.007892                | 0.08243       | 1.5375            | 1.5741          | +2.4%           | 6.236          |
| 161592               | 1.6190               | 0.008151                | 0.08296       | 1.5940            | 1.6411          | +3.0%           | 6.484          |

Monotone in every column: code loss +16.2% from 100995 to 161592, teacher-forced
offset +12.0%, sampled recon +5.2%, full-window penalty +0.7% -> +3.0%.

### The decoding artifact cannot explain this

`val/policy/loss/offset` is `L1Loss` **teacher-forced at the ground-truth codes**.
It never touches the code head's decision and is identical under argmax and under
sampling. It rose +12.0%. Simultaneously the train-side offset stayed ~0.0059 and
`quality/gap/policy/loss/code_0` (train-minus-val) widened 0.758 -> 1.096. That is
overfitting, measured on a decoding-independent quantity.

Note the *direction* of the sampling bias on the trend: sampled recon moved only
+5.2% while the two decoding-independent quantities moved +16.2% and +12.0%.
Partly saturated against the untrained-offset noise floor, sampled recon
**compresses** the code-head degradation rather than inventing it.

But do not read that as "it is worse than it looks" — §4 measures the argmax
decoding and finds it **flat**, so the logged curve is misleading in *both*
directions at once: it understates the rate at which the code head is degrading
(+5.2% against +16.2% / +12.0%) **and** it overstates the degradation of the
deployed output, which is +0.1%. Neither the sampled curve nor the code loss is a
proxy for what gets served. See §4 and §5-Q1.

## 3. Train-side counter-curve (why this is overfitting, not a bug)

`do8m9ot8`, train code loss averaged over the four quantizers, and the offset-head
dead-gradient fraction:

| step   | train code mean  | train offset | train sampled recon | train partial | train full | dead_grad offset_head | lr        |
| ------ | ---------------- | ------------ | ------------------- | ------------- | ---------- | --------------------- | --------- |
| 0      | —                | —            | —                   | —             | —          | 0.2415                | —         |
| 10150  | —                | —            | —                   | —             | —          | 0.8009                | —         |
| 20299  | 1.0436           | 0.006500     | 0.07303             | 1.0357        | 1.0506     | 0.8167 (@30400)       | 1.000e-04 |
| 50699  | 0.9104           | 0.005153     | 0.05261             | 0.8785        | 0.9385     | 0.8077 (@40550)       | 9.544e-05 |
| 91199  | 0.7393           | 0.006827     | 0.06309             | 0.7702        | 0.7119     | 0.8002 (@81050)       | 7.733e-05 |
| 111399 | 0.7049           | 0.007071     | 0.07378             | 0.7203        | 0.6912     | 0.7994 (@101250)      | 6.467e-05 |
| 131599 | 0.6666           | 0.007315     | 0.07236             | 0.6741        | 0.6601     | 0.7866 (@121450)      | 5.087e-05 |
| 151799 | 0.5657           | 0.004835     | 0.06703             | 0.6034        | 0.5324     | 0.7777 (@141650)      | 3.700e-05 |
| 162300 | 0.5628 (@162.1k) | 0.005944     | 0.06562             | 0.5841        | 0.5910     | 0.7733                | 3.025e-05 |

Train code loss falls monotonically 1.044 -> 0.563 over the same window in which val
code loss rises 1.371 -> 1.619. The scissor opens from step ~90k onward.

**`dead_grad_frac/offset_head` is NOT still climbing.** It ramps 0.24 -> 0.80 inside
the first ~10k steps and is then flat-to-declining (0.817 @30k -> 0.773 @162k). The
"0.29 -> 0.83 over 27k steps" observation was that early ramp, i.e. the offset
table reaching its steady-state coverage of the codebook, not a progressive decay.
Prediction from this: the sampled/argmax recon ratio should be roughly *stable*
across the two checkpoints rather than widening.

## 4. The argmax (deployment) measurement — the point of this exercise

Method: `rmind.scripts.patch_policy_eval` at `adf29a4`, full clip37 val split
(14,365 clips), batch 12, `--serial-build`, seed 1337, bf16 autocast under
`no_grad`, both decodings taken from the **same** logits. Run on aboutblank
(RTX 5090, same arch as kitkat/sisyphos), one checkpoint at a time.

**Control first.** Re-evaluating `model-do8m9ot8:v0` on aboutblank reproduced all
**21 of 21** val scalars from `patch_policy_causal_offline_val.json` (measured on
sisyphos) **bit-identically** — argmax recon 0.04148835316300392 both times, and the
tails, per-cluster and per-position tables match digit for digit. The v1 numbers
below are therefore directly comparable to the epoch-0 record.

### Headline

| metric | ep0 (v0, step 80797) | ep1 (v1, step 161594) | change |
| --- | --- | --- | --- |
| **recon L1 ARGMAX — what is served** | **0.041488** | **0.041543** | **+0.1%** |
| recon L1 sampled — what is logged | 0.077927 | 0.082828 | +6.3% |
| sampled / argmax ratio | 1.878x | 1.994x | +6.2% |
| offset L1 (teacher-forced) | 0.007224 | 0.008148 | +12.8% |
| code focal, smoothed, mean q0-3 | 1.3709 | 1.6196 | +18.1% |
| code focal, UNSMOOTHED, mean q0-3 | 1.0825 | 1.3662 | +26.2% |
| top-1 code acc (mean over q) | 0.4688 | 0.4481 | -4.4% |
| joint code acc (all 4 q correct) | 0.0918 | 0.0838 | -8.8% |
| p(GT) | 0.3294 | 0.3304 | **+0.3%** |
| entropy (nats; uniform = 2.77) | 1.5835 | 1.5169 | **-4.2%** |

Per quantizer:

| quantizer | ep0 smoothed | ep1 smoothed | ep0 unsmoothed | ep1 unsmoothed | ep0 top1 (last) | ep1 top1 (last) | ep0 p_gt | ep1 p_gt | ep0 entropy | ep1 entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q0 | 1.3256 | 1.5612 | 1.0337 | 1.3093 | 0.5197 | 0.5154 | 0.3803 | 0.3794 | 1.4613 | 1.4514 |
| q1 | 1.4662 | 1.8297 | 1.1835 | 1.5998 | 0.3692 | 0.3426 | 0.2810 | 0.2700 | 1.6072 | 1.5331 |
| q2 | 1.1257 | 1.2603 | 0.8119 | 0.9638 | 0.6044 | 0.5689 | 0.3929 | 0.3993 | 1.5529 | 1.4875 |
| q3 | 1.5661 | 1.8273 | 1.3010 | 1.5918 | 0.3806 | 0.3461 | 0.2656 | 0.2666 | 1.6967 | 1.5758 |

Tails:

| series | ep0 mean/p50/p95/p99/max | ep1 mean/p50/p95/p99/max |
| --- | --- | --- |
| `argmax_recon/all` | 0.0415 / 0.0278 / 0.1141 / 0.1852 / 0.2863 | 0.0415 / 0.0280 / 0.1120 / 0.2034 / 0.2732 |
| `argmax_recon/last` | 0.0417 / 0.0226 / 0.1411 / 0.2676 / 0.4701 | 0.0421 / 0.0226 / 0.1497 / 0.2646 / 0.4788 |
| `sampled_recon/all` | 0.0779 / 0.0680 / 0.1525 / 0.2114 / 0.3219 | 0.0828 / 0.0739 / 0.1555 / 0.2140 / 0.2812 |
| `sampled_recon/last` | 0.0780 / 0.0452 / 0.2755 / 0.3489 / 0.5532 | 0.0833 / 0.0501 / 0.2778 / 0.3534 / 0.5953 |
| `offset/all` | 0.0072 / 0.0047 / 0.0200 / 0.0328 / 0.0697 | 0.0081 / 0.0052 / 0.0223 / 0.0353 / 0.0708 |
| `offset/last` | 0.0073 / 0.0042 / 0.0236 / 0.0492 / 0.2176 | 0.0083 / 0.0047 / 0.0269 / 0.0544 / 0.2154 |

The argmax mean is flat and its distribution is essentially unchanged (p50 +0.7%,
p95 -1.8%, p99 +9.8%, max -4.6% on `/all`). The full per-position, per-cluster and
window-bucket tables are in `diag_results/argmax_trend/tables.md`; the raw numbers
are `argmax_v0.json` / `argmax_v1.json` in the same directory.

### Why the served metric is flat while the loss explodes

`p(GT)` is unchanged (+0.3%) and entropy **fell** 4.2%. So the code head did not
flatten — it **sharpened**. It keeps the same probability mass on the correct code
and spends its new confidence on the wrong ones. Focal cross-entropy punishes
exactly that (a confidently wrong prediction is what `(1-p)^gamma` weights most),
which is why the code loss moves +18% / +26% while `p_gt` does not move at all.

And the errors that top-1 accuracy newly counts (-4.4%, -8.8% joint) are
**decode-equivalent**: the 4-level residual codebook is redundant in action space,
so picking a neighbouring code changes the reconstructed gas/brake/steer almost not
at all. Argmax recon is flat *because* the extra mistakes land on codes whose
decoded action is nearly the same. This is the mirror image of the sampled-decode
artifact: sampling lands on codes with an untrained *offset* entry, which does hurt.

A rough consistency check on that claim, from the measured numbers rather than
asserted: argmax recon is the sum of a codebook-selection error and an offset error.
The offset term rose **+12.8%** in absolute terms (0.007224 -> 0.008148, i.e.
+0.00092) against a total argmax recon of ~0.0415. Had nothing else changed, that
alone should have pushed argmax recon up by roughly +2%. It moved +0.13%
(+0.000055). So the codebook-selection contribution must have *improved* by very
nearly the amount the offset term worsened — which is exactly the decode-equivalence
claim, and it is also consistent with p(GT) being flat and the head sharpening. This
is a consistency check, not a clean decomposition: the offset L1 is measured only at
ground-truth codes, so it does not partition argmax recon exactly, and the two terms
are not additive in L1. But the order of magnitude is unambiguous — something had to
cancel a 2%-scale regression, and only the selection term can have.

The per-position curve shows the trade explicitly — under argmax the **shallow**
readouts improved and the **deep** ones degraded, crossing over around t=13:

| position | ep0 recon_argmax | ep1 recon_argmax | change |
| --- | --- | --- | --- |
| t=0 (1f) | 0.0481 | 0.0476 | -1.1% |
| t=5 (6f) | 0.0416 | 0.0410 | -1.4% |
| t=11 (12f) | 0.0408 | 0.0407 | -0.3% |
| t=13 (14f) | 0.0409 | 0.0409 | 0.0% |
| t=21 (22f) | 0.0409 | 0.0414 | +1.2% |
| t=31 (32f) | 0.0417 | 0.0421 | +1.0% |

### Per-cluster argmax L1 at the last readout — where it DID get worse

| cluster | n | gas ep0 -> ep1 | brake ep0 -> ep1 | steer ep0 -> ep1 |
| --- | --- | --- | --- | --- |
| cruise | 5238 | 0.0804 -> 0.0795 | 0.0019 -> 0.0017 | 0.0065 -> 0.0062 |
| idle_coast | 3463 | 0.0458 -> 0.0491 | 0.0644 -> 0.0532 | 0.0213 -> 0.0191 |
| highway | 1418 | 0.1574 -> 0.1682 | 0.0012 -> 0.0009 | 0.0033 -> 0.0031 |
| braking | 1261 | 0.0139 -> 0.0151 | 0.1086 -> 0.1168 | 0.0392 -> 0.0324 |
| cruise_turn | 932 | 0.0730 -> 0.0716 | 0.0149 -> 0.0161 | 0.0661 -> 0.0560 |
| acceleration | 786 | 0.0638 -> 0.0626 | 0.0175 -> 0.0168 | 0.0150 -> 0.0143 |
| gas_release | 689 | 0.0657 -> 0.0616 | 0.0014 -> 0.0014 | 0.0070 -> 0.0067 |
| **braking_turn** | **578** | 0.0077 -> 0.0115 | 0.1086 -> 0.1311 | **0.1079 -> 0.2049** |

The flat mean hides one real regression: `braking_turn` steering **+90%**, the
rarest and most safety-relevant cluster (brake >= 0.02 AND |steer| >= 0.05), plus
`braking` brake +7.5% and `highway` gas +6.9%. Everything common (cruise,
acceleration, gas_release, cruise_turn) is flat or slightly better. n = 578 for
`braking_turn`, so treat the magnitude as indicative, but the sign is consistent
with overfitting eating the tail of the action distribution first.

## 5. Answers

### Q1 — is argmax recon improving, flat, or degrading?

**FLAT.** 0.041488 -> 0.041543 over epoch 0 -> epoch 1, i.e. +0.13% on 80,797 extra
optimizer steps. Not improving either. On the deployment decoding this half-epoch of
training bought **nothing measurable**, and the p50/p95/max of the argmax
distribution are unchanged.

This holds for both runs, because (§0) they are the same run.

So the honest answer has two halves and they point in opposite directions:

* **The model is genuinely overfitting.** Val code loss +18% smoothed / +26%
  unsmoothed, teacher-forced offset L1 +12.8% (decoding-independent), top-1 -4.4%,
  joint -8.8%, train-val gap widening 0.76 -> 1.10 while train code loss falls
  1.04 -> 0.56. Monotone across all four logged val points. This is not a sampling
  artifact and it is not a logging bug.
* **But the served output has not degraded.** Argmax recon is flat, `p_gt` is flat,
  the head is sharpening rather than collapsing, and every high-population driving
  cluster is flat or better. The degradation is real in probability space and
  invisible in action space, except in the `braking_turn` tail.

The correct reading of "the curves look like they are degrading" is therefore:
**yes, the curves are honest about the code head degrading; no, that degradation has
not (yet) reached the deployed metric.** The logged sampled curve overstates the
*absolute* recon error (1.88-1.99x) and understates the *rate* of the code-head
degradation (+6.3% vs +18%/+12.8%) — it is misleading in both directions at once.

### Q2 — does the sampled/argmax ratio widen with training?

**Yes, but only slightly: 1.878x -> 1.994x (+6.2%).** And the cause is **not** the
code head flattening — entropy *fell* 4.2% (1.5835 -> 1.5169) and every quantizer
sharpened. It is the offset table's off-GT region: teacher-forced offset L1 rose
+12.8% even *at* the ground-truth codes, and `dead_grad_frac/offset_head` shows the
off-GT entries were already frozen out long ago — it ramps 0.24 -> 0.80 within the
first ~10k steps and then sits flat-to-declining (0.817 @30k -> 0.773 @162k). The
earlier "0.29 -> 0.83 over 27k steps" was that initial ramp to steady state, not a
progressive decay, which is exactly why the ratio only creeps rather than blowing up.

A sharper head draws off-argmax less often, which pushes the ratio *down*; a worse
offset table pushes it *up*; the second effect wins by 6%.

### Q3 — partial (0-14) vs full (15-31) window under ARGMAX

**The 16-frame window still does not pay off on held-out data, and it has got
worse.**

| bucket comparison (full vs partial; negative = full is better) | ep0 | ep1 |
| --- | --- | --- |
| recon_argmax | **-1.8%** | **-0.0%** |
| code_focal | +0.4% | +3.0% |
| code_plain | +0.5% | +4.0% |
| top1_acc | +0.5% (full better) | -0.9% (full worse) |
| joint_acc | +3.7% (full better) | +0.4% |
| p_gt | +1.1% (full better) | -0.1% |
| offset (teacher-forced) | +0.7% | +2.2% |

At epoch 0 the argmax buckets gave the full window a marginal -1.8% edge while the
code loss gave it a marginal +0.4% penalty — the two disagreed in sign, which is why
the epoch-0 doc called it "gone or marginally reversed". At epoch 1 the disagreement
resolves **against** the full window: the argmax edge has decayed to exactly 0.0%
and the code-loss penalty has grown 7x to +3.0% (+4.0% unsmoothed). Meanwhile the
*train*-side full-window advantage kept growing (partial 0.6034 vs full 0.5324 at
step 151799, -11.8%). Long context is being memorized, not learned.

The per-position argmax curve says the same thing more sharply: it bottoms out at
**t=9-12, i.e. 10-13 frames of context** (0.0407) and then rises monotonically to
0.0421 at t=31. Depth beyond ~12 frames is not merely unhelpful, it is mildly
harmful on val.

So yes — the conclusion the epoch-0 doc drew survives and strengthens: a `window: 6`
arm is worth running. Caveats before treating it as free: the argmax spread across
the whole position range is only 0.0476 (t=0) -> 0.0407 (t=11) -> 0.0421 (t=31),
i.e. ~3% between the best and the deepest, so a short window buys compute, not
quality, and the t=0-2 positions ARE clearly worse (0.0476 / 0.0431 / 0.0418), so a
window of 6 should still be given >= 4-6 frames of context at inference, not 1.

### Q4 — divergence between the two runs

**There is none, and that is the finding.** Not "no config difference beyond step
count" — the two runs are the *same computation*:

* code identical (`231ede7` = `815d4d2` + a docs-only SVG),
* 4884 of 4890 config keys identical, the 6 that differ are path strings,
* same `seed: 1337`, same `val_check_interval: 0.25`, same resume checkpoint
  (`model-do8m9ot8:v0`), same GPU model,
* and every logged val float agrees to the last bit at the one matched step,
  including the multinomial `offset_sampled_recon` (§0).

The only real differences are **step count** (162.5k vs 108.1k) and **throughput**
(3083 vs 2048 steps/h — host contention on sisyphos, not configuration).
`sage-universe-592` is not an independent arm and cannot serve as a seed replicate;
it is a duplicate. Whatever it computes, `vague-shape-587` has already computed
53k steps earlier.

## 6. Verdict and recommendation

**Is either run actually degrading?** In probability space yes, monotonically and
identically in both. In deployment space no — argmax recon is flat. Neither run is
worth continuing, but for a different reason than "it is degrading": **it has
stopped improving on the metric that ships**, and the code head is now moving
backwards, so continued training is at best neutral and at worst is eroding the tail
(`braking_turn` steer +90%).

Concretely:

Nothing in this investigation touched either run: both were live throughout, no
process was stopped or started, and the evaluations ran on a third box (aboutblank).
The actions below are recommendations for the operator.

1. **Stop `p8d6fxao` (sage-universe-592).** It is a bit-exact duplicate that is
   1.5x slower than the run it duplicates, and it holds a 5090 on the most contended
   box. It produces zero information. This is a free GPU.
2. **Do not simply let `do8m9ot8` run on.** Epoch 1 -> 2 bought +0.13% on the served
   metric for 80.8k steps. The `model-do8m9ot8:v1` checkpoint (or better, an
   epoch-0/epoch-1-early one, since the shallow-readout argmax numbers are the ones
   that improved) is the artifact to keep. If it is left running, the epoch-2
   checkpoint should be evaluated the same way before anything is exported.
3. **Restarting at the current `feat/patch-policy-decoder-causal` HEAD (`50be44f`)
   is worth it only in combination with a real change.** HEAD does fix the
   observability hole — `_compute_metrics` there logs `offset_argmax_recon`,
   `offset_argmax_recon_last`, `code_acc_{q}_last` and `code_acc_joint_last`, which
   is exactly the set this document had to reconstruct offline, and it would have
   made the "is it degrading?" question answerable from the wandb page in seconds.
   But re-running the identical recipe with better logging just re-derives these
   numbers at full cost. Land the HEAD metrics and change the recipe in the same
   launch.
4. **The recipe change to make is regularization/length, not window alone.** The
   diagnosis is overfitting with a *sharpening* head: label smoothing 0.1 is not
   preventing overconfidence on val, and the useful signal saturates at ~10-13
   frames. So the cheap high-information arm is `window: 6` (≈3x cheaper attention
   per readout, and by Q3 no measured argmax cost) run for **fewer** steps — epoch 1
   is already past the useful point. Add the offset-head fix as a separate arm: with
   `teacher_force_offset: true` and 79% of the offset head dead-gradient, the off-GT
   offset region never trains, which is what makes sampled decoding useless as a
   monitor and what drove the ratio to 2x.

## 7. Caveats

* **Only two checkpoints exist**, and they are one epoch apart.
  `ModelCheckpoint(every_n_epochs=1, save_top_k=1)` with no `save_last` and no
  mid-epoch saving means there is no way to get a finer argmax trend without new
  training. The 4-point sampled val curve (§2) is the only intra-epoch resolution
  available, and it is monotone, so the flat argmax result is bracketed by two
  measurements 80.8k steps apart — a non-monotone excursion in between cannot be
  ruled out.
* **`p8d6fxao` was never evaluated under argmax** because it has no checkpoint at
  all. The claim that the argmax numbers apply to it rests on the bit-exactness
  argument in §0, which is strong (identical multinomial draws) but is established
  at one step only. Its first checkpoint lands at step 161594; if it is not stopped,
  that checkpoint should reproduce `model-do8m9ot8:v1` exactly, which is a free
  falsification test of §0.
* Argmax recon being flat is a statement about the **mean over 14,365 clips and 32
  readouts**. It is not a statement about safety. The `braking_turn` steering cell
  nearly doubled, and small-n clusters (578, 689, 786) are noisy. A flat aggregate
  with a degrading tail is the failure mode this table would hide.
* The val split is 5 drives with a fixed mix; both checkpoints see exactly the same
  clips, so the ep0-vs-ep1 comparison is paired and tight, but the absolute level
  does not generalize to other drive mixes.
* Label smoothing 0.1 puts an irreducible floor on the smoothed code loss; the
  unsmoothed column is the one comparable to the pre-smoothing arms
  (stellar-hill, dashing-dream). Note the unsmoothed loss degraded *more* than the
  smoothed (+26.2% vs +18.1%), because the smoothing floor is a constant that
  dilutes the relative change.
* The train-side references are logged in TRAIN mode with drop-path 0.1 and head
  dropout active, so every train-val gap quoted is a **lower bound**.
* Eval was on aboutblank, not on the training hosts. The 21/21 bit-identical
  reproduction of the sisyphos-measured v0 numbers is the evidence that this does
  not matter (all three boxes are RTX 5090); it would not hold on a 4090.
* `recon_sampled` in both evaluations is a fresh multinomial draw, so the two
  checkpoints' sampled numbers carry independent sampling noise. The argmax,
  offset, code, `p_gt` and entropy columns are deterministic.
