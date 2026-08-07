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
have, and it is the slower of the two (~1785 steps/h on contended sisyphos vs
~2860 steps/h on kitkat). Every argmax measurement below is made on `do8m9ot8`
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

Note also the *direction* of the sampling bias on the trend: sampled recon moved
only +5.2% while the two decoding-independent quantities moved +16.2% and +12.0%.
Sampled recon is partly saturated against the untrained-offset noise floor, so it
**compresses** the degradation rather than inventing it. The logged curve
understates how bad the trend is.

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
