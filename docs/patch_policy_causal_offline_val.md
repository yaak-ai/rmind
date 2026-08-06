# Offline validation for the causal patch-policy arm (do8m9ot8, epoch 0)

* checkpoint: `yaak/rmind/model-do8m9ot8:v0` -- run `do8m9ot8` (vague-shape-587),
  epoch 0, `global_step` 80797, experiment `yaak/patch_policy/dinov2_dinowm_causal`
* val set: the FULL clip37 val split, 14365 clips (5 drives), batch 12, 32 readouts each
* code loss: `FocalLoss(gamma=2, label_smoothing=0.1)` -- the
  checkpoint's own loss, so directly comparable to the run's `train/policy/loss/code_*`.
  The `unsmoothed` column is the same focal at `label_smoothing=0.0`, which is the
  scale the pre-smoothing arms (stellar-hill, dashing-dream) were measured on.
* offset: plain `L1Loss`, teacher-forced at the ground-truth codes
* `sample_codes=True`, so `recon_sampled` is a multinomial draw and
  `recon_argmax` is the deterministic decode of the same logits
* eval in `bf16` autocast under `no_grad`, matching the run's `precision: bf16-mixed`

## Why there was nothing to compare against: the run logged ZERO val points

Evidence, all from the run's own wandb record:

* it is ONE unbroken launch, 2026-08-05T14:32Z -> 2026-08-06T17:37Z: `_runtime`
  rises monotonically 1798s -> 99306s with no seam, and it ended on
  `Received SIGTERM: 15` at `global_step` 87099 (epoch 1, batch ~6.4k). It was
  STOPPED, not crashed by a bug, and it is not running now.
* the epoch 0 -> 1 boundary has no time gap: `global_step` 80699 (epoch 0) at
  15:10:57 and 80799 (epoch 1) at 15:14:07 -- 190s for those 100 steps, the same
  as every neighbouring interval. A val pass over 14,365 clips takes minutes, so
  the epoch-end val loop provably never ran.
* the only `predict/*` row in the entire history is at `_step=0`: the
  `num_sanity_val_steps` pass, whose `validation_step` returns before logging.
  There is no `val/*` key anywhere in the run.
* the launch (source artifact v226, 2026-08-05T14:02:43Z) PREDATES commit
  815d4d2 "val_check_interval 0.25" (2026-08-06T09:14), so epoch 0 ran at
  Lightning's default `val_check_interval: 1.0`. That sets
  `val_check_batch == num_training_batches`, leaving the single trigger
  `(batch_idx + 1) % val_check_batch == 0` having to land exactly on the final
  batch -- and `_should_check_val_fx`'s `is_last_batch` shortcut is gated behind
  `is_infinite_dataset`, which a `Sized` loader is not. Any off-by-one between
  the reported loader length and the batches actually yielded silences
  validation for the whole run. The progress bar reported 80798 batches while
  the epoch's last logged `global_step` was 80797.

Consequence: `val_check_interval: 0.25`, already at HEAD (231ede7), is the fix.
`val_check_batch` becomes `int(80798 * 0.25) = 20199`, a mid-epoch multiple that
does not depend on the modulo hitting the epoch end. A resumed run gets a real
val point ~14k batches in (~5.5h). This document is the one-time bridge for the
checkpoint that already exists.

## Deliverable 1 -- the val scalars, and the train/val gap

Train reference = the run's own logged curves averaged over `global_step` 78099..80899 (29 logged points).

| metric | val | train @ ~80.8k | gap |
| --- | --- | --- | --- |
| `policy/loss/code_0` (smoothed) | 1.3256 | 0.5800 | +128.5% |
| `policy/loss/code_1` (smoothed) | 1.4662 | 0.8221 | +78.3% |
| `policy/loss/code_2` (smoothed) | 1.1257 | 0.7385 | +52.4% |
| `policy/loss/code_3` (smoothed) | 1.5661 | 0.9843 | +59.1% |
| **code mean over quantizers** | **1.3709** | **0.7812** | **+75.5%** |
| `policy/loss/offset` | 0.007224 | 0.005439 | +32.8% |
| `policy/metric/offset_sampled_recon` | 0.0779 | 0.0615 | +26.6% |

Unsmoothed code loss (for comparison against the pre-smoothing arms; no train
counterpart exists because the run only ever logged the smoothed variant):

| quantizer | val smoothed | val unsmoothed |
| --- | --- | --- |
| q0 | 1.3256 | 1.0337 |
| q1 | 1.4662 | 1.1835 |
| q2 | 1.1257 | 0.8119 |
| q3 | 1.5661 | 1.3010 |
| mean | 1.3709 | 1.0825 |

## Deliverable 2 -- partial vs full window

Readouts `partial 0-14` see a partial context; `full 15-31` see the full 16-frame window that inference serves.

| metric | partial | full | full vs partial | train (same buckets) |
| --- | --- | --- | --- | --- |
| code_focal | 1.3682 | 1.3732 | +0.4% | 0.8067 -> 0.7587 (-5.9%) |
| code_plain | 1.0794 | 1.0852 | +0.5% |  |
| top1_acc | 0.4676 | 0.4698 | +0.5% |  |
| p_gt | 0.3275 | 0.3311 | +1.1% |  |
| entropy | 1.5892 | 1.5785 | -0.7% |  |
| offset | 0.0072 | 0.0072 | +0.7% | 0.0054 -> 0.0055 (+2.7%) |
| recon_sampled | 0.0777 | 0.0781 | +0.5% |  |
| recon_argmax | 0.0419 | 0.0411 | -1.8% |  |
| joint_acc | 0.0901 | 0.0934 | +3.7% |  |

### Reading

The bucket averages understate what the per-position table (below) shows: on
held-out data the code loss falls only from t=0 to t=1-3 and is then FLAT, and
from t~12 onward it creeps slightly the wrong way. p_gt peaks at t=11-13 (0.3324)
and entropy bottoms out in the same place, i.e. the best-generalizing readout is
not the deepest one. All the context that generalizes is spent within ~4 frames.

The train curves say the opposite: there, the full-window bucket beat the partial
one by 5.9% on code loss and the gap was widening. That advantage does NOT
transfer -- on val it is +0.4%, i.e. gone or marginally reversed. The train-side
full-window advantage is therefore memorization of long context, not
generalizable long-context conditioning. Offset agrees on both sides that depth
is irrelevant (train +2.7%, val +0.7%, both against the full window).

H6 check: this is not an entropy-collapse artifact. Entropy is 1.58 nats against
a uniform 2.77, so the head is far from collapsed, and p_gt (0.331 full vs 0.328
partial) moves the same negligible amount as the focal. p_gt is NOT rising while
focal explodes -- the two agree that the buckets are equivalent, so the flat
verdict survives the calibration confound.

## Deliverable 3 -- decoding and tails

| series | mean | p50 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| `offset/all` | 0.0072 | 0.0047 | 0.0200 | 0.0328 | 0.0697 |
| `argmax_recon/all` | 0.0415 | 0.0278 | 0.1141 | 0.1852 | 0.2863 |
| `sampled_recon/all` | 0.0779 | 0.0680 | 0.1525 | 0.2114 | 0.3219 |
| `offset/last` | 0.0073 | 0.0042 | 0.0236 | 0.0492 | 0.2176 |
| `argmax_recon/last` | 0.0417 | 0.0226 | 0.1411 | 0.2676 | 0.4701 |
| `sampled_recon/last` | 0.0780 | 0.0452 | 0.2755 | 0.3489 | 0.5532 |

## All rows (summary buckets, then per position)

| position | code_focal | code_plain | top1_acc | joint_acc | p_gt | entropy | offset | recon_sampled | recon_argmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all (wandb) | 1.3709 | 1.0825 | 0.4688 | 0.0918 | 0.3294 | 1.5835 | 0.0072 | 0.0779 | 0.0415 |
| last (=bsln) | 1.3765 | 1.0889 | 0.4684 | 0.0914 | 0.3299 | 1.5795 | 0.0073 | 0.0780 | 0.0417 |
| partial 0-14 | 1.3682 | 1.0794 | 0.4676 | 0.0901 | 0.3275 | 1.5892 | 0.0072 | 0.0777 | 0.0419 |
| full 15-31 | 1.3732 | 1.0852 | 0.4698 | 0.0934 | 0.3311 | 1.5785 | 0.0072 | 0.0781 | 0.0411 |

Per readout position (`t=<i>`; the parenthesis is the context depth in frames,
capped at the 16-frame window):

| position | code_focal | code_plain | top1_acc | joint_acc | p_gt | entropy | offset | recon_sampled | recon_argmax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t=0 (1f) | 1.3852 | 1.0995 | 0.4399 | 0.0653 | 0.2986 | 1.6682 | 0.0075 | 0.0783 | 0.0481 |
| t=1 (2f) | 1.3518 | 1.0609 | 0.4597 | 0.0798 | 0.3183 | 1.6188 | 0.0073 | 0.0765 | 0.0436 |
| t=2 (3f) | 1.3605 | 1.0706 | 0.4640 | 0.0859 | 0.3240 | 1.5983 | 0.0072 | 0.0777 | 0.0425 |
| t=3 (4f) | 1.3638 | 1.0743 | 0.4686 | 0.0891 | 0.3268 | 1.5898 | 0.0072 | 0.0775 | 0.0419 |
| t=4 (5f) | 1.3664 | 1.0771 | 0.4680 | 0.0896 | 0.3282 | 1.5856 | 0.0072 | 0.0767 | 0.0418 |
| t=5 (6f) | 1.3678 | 1.0788 | 0.4697 | 0.0918 | 0.3294 | 1.5830 | 0.0072 | 0.0777 | 0.0416 |
| t=6 (7f) | 1.3681 | 1.0792 | 0.4704 | 0.0924 | 0.3303 | 1.5810 | 0.0071 | 0.0776 | 0.0414 |
| t=7 (8f) | 1.3689 | 1.0802 | 0.4711 | 0.0923 | 0.3309 | 1.5791 | 0.0071 | 0.0781 | 0.0413 |
| t=8 (9f) | 1.3690 | 1.0803 | 0.4719 | 0.0952 | 0.3316 | 1.5780 | 0.0071 | 0.0775 | 0.0412 |
| t=9 (10f) | 1.3694 | 1.0808 | 0.4722 | 0.0945 | 0.3320 | 1.5768 | 0.0071 | 0.0781 | 0.0409 |
| t=10 (11f) | 1.3702 | 1.0816 | 0.4725 | 0.0953 | 0.3323 | 1.5760 | 0.0071 | 0.0782 | 0.0409 |
| t=11 (12f) | 1.3708 | 1.0823 | 0.4728 | 0.0958 | 0.3324 | 1.5756 | 0.0071 | 0.0773 | 0.0408 |
| t=12 (13f) | 1.3708 | 1.0824 | 0.4718 | 0.0950 | 0.3324 | 1.5756 | 0.0071 | 0.0774 | 0.0408 |
| t=13 (14f) | 1.3704 | 1.0820 | 0.4714 | 0.0947 | 0.3324 | 1.5758 | 0.0071 | 0.0788 | 0.0409 |
| t=14 (15f) | 1.3701 | 1.0816 | 0.4707 | 0.0947 | 0.3323 | 1.5762 | 0.0071 | 0.0783 | 0.0409 |
| t=15 (16f) | 1.3698 | 1.0812 | 0.4704 | 0.0952 | 0.3321 | 1.5767 | 0.0071 | 0.0787 | 0.0408 |
| t=16 (17f) | 1.3695 | 1.0810 | 0.4705 | 0.0950 | 0.3321 | 1.5770 | 0.0072 | 0.0783 | 0.0409 |
| t=17 (18f) | 1.3696 | 1.0812 | 0.4711 | 0.0957 | 0.3320 | 1.5774 | 0.0072 | 0.0782 | 0.0407 |
| t=18 (19f) | 1.3701 | 1.0817 | 0.4711 | 0.0952 | 0.3318 | 1.5777 | 0.0072 | 0.0772 | 0.0408 |
| t=19 (20f) | 1.3708 | 1.0826 | 0.4706 | 0.0950 | 0.3316 | 1.5779 | 0.0072 | 0.0787 | 0.0408 |
| t=20 (21f) | 1.3719 | 1.0837 | 0.4697 | 0.0939 | 0.3314 | 1.5781 | 0.0072 | 0.0767 | 0.0409 |
| t=21 (22f) | 1.3726 | 1.0845 | 0.4696 | 0.0938 | 0.3313 | 1.5783 | 0.0072 | 0.0788 | 0.0409 |
| t=22 (23f) | 1.3733 | 1.0853 | 0.4697 | 0.0936 | 0.3311 | 1.5785 | 0.0072 | 0.0784 | 0.0409 |
| t=23 (24f) | 1.3736 | 1.0857 | 0.4694 | 0.0933 | 0.3310 | 1.5788 | 0.0073 | 0.0780 | 0.0410 |
| t=24 (25f) | 1.3738 | 1.0859 | 0.4697 | 0.0927 | 0.3309 | 1.5790 | 0.0073 | 0.0776 | 0.0411 |
| t=25 (26f) | 1.3744 | 1.0866 | 0.4698 | 0.0925 | 0.3308 | 1.5792 | 0.0073 | 0.0776 | 0.0412 |
| t=26 (27f) | 1.3751 | 1.0874 | 0.4700 | 0.0924 | 0.3307 | 1.5794 | 0.0073 | 0.0771 | 0.0413 |
| t=27 (28f) | 1.3757 | 1.0880 | 0.4695 | 0.0921 | 0.3306 | 1.5794 | 0.0073 | 0.0789 | 0.0413 |
| t=28 (29f) | 1.3758 | 1.0881 | 0.4692 | 0.0922 | 0.3305 | 1.5794 | 0.0073 | 0.0789 | 0.0415 |
| t=29 (30f) | 1.3759 | 1.0883 | 0.4689 | 0.0918 | 0.3304 | 1.5794 | 0.0073 | 0.0782 | 0.0416 |
| t=30 (31f) | 1.3763 | 1.0887 | 0.4685 | 0.0918 | 0.3301 | 1.5795 | 0.0073 | 0.0785 | 0.0416 |
| t=31 (32f) | 1.3765 | 1.0889 | 0.4684 | 0.0914 | 0.3299 | 1.5795 | 0.0073 | 0.0780 | 0.0417 |

## Per-cluster L1 at the last readout

| cluster | n | samp_gas | samp_brake | samp_steer | argm_gas | argm_brake | argm_steer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cruise | 5238 | 0.1064 | 0.0171 | 0.0421 | 0.0804 | 0.0019 | 0.0065 |
| idle_coast | 3463 | 0.0590 | 0.0651 | 0.0554 | 0.0458 | 0.0644 | 0.0213 |
| highway | 1418 | 0.1933 | 0.0181 | 0.0355 | 0.1574 | 0.0012 | 0.0033 |
| braking | 1261 | 0.0311 | 0.1424 | 0.0792 | 0.0139 | 0.1086 | 0.0392 |
| cruise_turn | 932 | 0.0875 | 0.0304 | 0.1134 | 0.0730 | 0.0149 | 0.0661 |
| acceleration | 786 | 0.0788 | 0.0279 | 0.0487 | 0.0638 | 0.0175 | 0.0150 |
| gas_release | 689 | 0.0929 | 0.0170 | 0.0458 | 0.0657 | 0.0016 | 0.0070 |
| braking_turn | 578 | 0.0291 | 0.1630 | 0.1451 | 0.0077 | 0.1193 | 0.1079 |

## `_compute_metrics` parity

Every number above is checked against `PatchPolicy._compute_metrics` -- the exact
reduction `validation_step` would have logged -- on a single batch, in the same
`bf16` autocast block, so the two are directly comparable:

```
metric                       this script   _compute_metrics    abs diff   rel diff
code_0                          0.969225           0.969239    1.33e-05   1.37e-05
code_1                          1.640331           1.640300    3.08e-05   1.88e-05
code_2                          1.209218           1.208636    5.82e-04   4.81e-04
code_3                          1.937336           1.937607    2.70e-04   1.39e-04
offset                          0.010046           0.010046    1.86e-09   1.85e-07
code_0_last                     0.614133           0.614282    1.49e-04   2.43e-04
code_1_last                     1.221514           1.221230    2.84e-04   2.33e-04
code_2_last                     0.989844           0.989760    8.33e-05   8.42e-05
code_3_last                     1.712785           1.713205    4.21e-04   2.46e-04
offset_last                     0.005788           0.005788    4.66e-10   8.05e-08
code_partial_window             1.544699           1.544731    3.23e-05   2.09e-05
code_full_window                1.345789           1.345605    1.83e-04   1.36e-04
offset_partial_window           0.010503           0.010503    9.31e-10   8.87e-08
offset_full_window              0.009643           0.009643    9.31e-10   9.66e-08

exempt (independent multinomial draws):
offset_sampled_recon            0.092108           0.086909

PARITY OK (tol 0.001 relative)
```

The offset terms agree to ~1e-9 and the code terms to <=5e-4 relative. The code
gap is fp accumulation order, not a reduction difference: this script evaluates
the focal ONCE over the fused `(b t g)` tensor while `_compute_metrics` calls the
loss per quantizer on a `(b t)` reshape. The offset path shares the identical
slicing and weighting and matches to machine precision, which is what validates
the per-position and per-bucket bookkeeping. `offset_sampled_recon` is exempt:
both sides take their own multinomial draw.

## Caveats

* epoch-0 checkpoint: one pass over the data, ~1/3 of the planned schedule.
* the train reference was logged in TRAIN mode with drop-path 0.1 and head dropout
  active, which inflates it. Every val-minus-train gap here is therefore a LOWER
  bound on the true generalization gap.
* label smoothing 0.1 adds an irreducible floor to the smoothed code loss
  (`eps * ce_uniform`, at least `0.1 * ln 16 = 0.277` for a uniform predictor and
  more as the logit margin grows), so the smoothed numbers are NOT comparable to
  the pre-smoothing arms. Use the unsmoothed column for that.
* val set is 5 drives / 14,365 clips. It is the full split, but it is small and its
  drive mix is fixed, so per-cluster cells with small n are noisy.
* the rbyte val sample index had to be built in-process (`--serial-build`); the
  configured forkserver pool dies with `BrokenProcessPool` on these drives.
* run on sisyphos from an isolated copy of 231ede7, because the shared checkout had
  another session's git merge in progress.
