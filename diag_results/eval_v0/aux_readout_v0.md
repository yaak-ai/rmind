# eval_v0: Arm MV auxiliary max-speed head readout (wandb run 0nr1ydjm)

Source: wandb `yaak/rmind/0nr1ydjm` (mapMV-warmstart-b1), pulled 2026-08-03 at
global_step ~5.6k (mid epoch 1 of 5; the evaluated artifact model-0nr1ydjm:v0
is the epoch-1 checkpoint). Zero-compute readout, per the standing protocol.

## Train-side curves (batch-level, known targets only)

| step | epoch | aux loss (w=0.2) | aux acc | known_frac |
|---|---|---|---|---|
| 99 | 0 | 0.325 | 0.446 | 0.63 |
| 499 | 0 | 0.029 | 0.950 | 0.63 |
| 1699 | 0 | 0.008 | 0.985 | 0.69 |
| 3299 | 0 | 0.002 | 1.000 | 0.63 |
| 4499 | 1 | 0.015 | 0.975 | 0.63 |
| 5299 | 1 | 0.007 | 0.984 | 0.66 |

Last-20-batches means: **aux acc 0.980**, aux loss 0.011, known_frac 0.669.

## Comparison to floors

- Linear-probe floor (frozen DINOv2 features -> 11-class limit, held-out
  drives): **39.2%** top-1 / 42.6% balanced.
- Majority-class baseline: **6.9%** (chance ~9.1% balanced).
- Aux head TRAIN accuracy: **~98%** -- far above both, **but not comparable**:
  the aux head reads the trunk at the max-speed TOKEN position, and with input
  dropout 0.3 the ground-truth token is VISIBLE in the input on ~70% of
  frames. ~98% is consistent with near-perfect copying on visible frames plus
  an unknown accuracy on the ~30% dropped frames; wandb does not log the
  dropped-frame-only accuracy, so the "vision reads the limit" claim cannot be
  isolated from this metric.

## Gaps (to fix before the final-checkpoint eval)

1. **No usable VAL aux metrics**: the single val pass so far (step 3850,
   end of epoch 0) logged `val/policy/metric/max_speed_aux_known_frac = 0` --
   the 5-drive val split has NO map-GT sidecars (none of the 5 val drives were
   attempted in the 638-drive sidecar build; verified in
   `caches/map_gt/build_summary.json` / `build_failures.json`). All val
   max-speed targets are UNKNOWN, which the aux loss ignores, so val aux
   acc/loss log as 0 and mean nothing.
2. **No per-class aux metrics** logged (only acc + known_frac) -- cannot
   check the 130-unsigned-motorway class that motivated the token.
3. **No dropped-frame-only accuracy** logged (see above) -- the single most
   informative number for Arm MV's "pressure vision to encode the limit"
   hypothesis is missing.
4. Arm M (1n0ih44y) has logged NO val pass at all yet -- its `best` artifact
   alias is not validation-selected.

Action items implied: build map-GT sidecars for the 5 val drives
(Niro115-HQ/2023-05-16--10-47-33, Niro104-HQ/2022-12-20--13-57-20,
Niro107-HQ/2023-05-12--12-05-15, Niro122-HQ/2023-04-05--12-06-39,
Niro102-HQ/2022-12-03--11-30-20), and log aux acc split by
token-visible/token-dropped and per-class.
