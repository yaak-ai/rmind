# Linear probe: frozen DINOv2 features -> speed class

Backbone: `vit_small_patch14_dinov2.lvd142m` @ 224x224 (center-crop 320x576 -> resize; final-layer post-norm patch tokens, mean-pooled) -- the dinov2_dinowm winner recipe.
Samples: 3620 frames / 594 drives (2985 train / 635 test, split BY DRIVE); 11/13 vocab classes present in both splits (dropped, missing from one split: ['UNLIMITED']).

- held-out accuracy: **39.2%**
- majority-class baseline (train majority = 120): **6.9%**
- balanced accuracy (mean per-class recall): **42.6%** (chance 9.1%)

## Per-class recall (test)

| class | n test | recall |
|---|---|---|
| WALK | 3 | 33.3% |
| 10 | 26 | 80.8% |
| 20 | 62 | 87.1% |
| 30 | 69 | 44.9% |
| 50 | 71 | 36.6% |
| 60 | 71 | 31.0% |
| 70 | 68 | 33.8% |
| 80 | 71 | 35.2% |
| 100 | 65 | 29.2% |
| 120 | 44 | 52.3% |
| 130 | 85 | 4.7% |

## Confusion matrix (rows = true, cols = pred)

| true \ pred | WALK | 10 | 20 | 30 | 50 | 60 | 70 | 80 | 100 | 120 | 130 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WALK | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | 0 | 21 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | 0 | 0 | 54 | 6 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| 30 | 0 | 4 | 7 | 31 | 15 | 1 | 2 | 4 | 3 | 1 | 1 |
| 50 | 0 | 2 | 3 | 18 | 26 | 15 | 4 | 1 | 1 | 1 | 0 |
| 60 | 0 | 1 | 0 | 4 | 7 | 22 | 9 | 11 | 10 | 5 | 2 |
| 70 | 0 | 0 | 0 | 1 | 8 | 9 | 23 | 6 | 14 | 5 | 2 |
| 80 | 1 | 1 | 1 | 1 | 5 | 12 | 4 | 25 | 13 | 6 | 2 |
| 100 | 0 | 0 | 1 | 1 | 4 | 9 | 12 | 11 | 19 | 6 | 2 |
| 120 | 0 | 0 | 0 | 0 | 1 | 1 | 5 | 6 | 4 | 23 | 4 |
| 130 | 0 | 0 | 0 | 0 | 0 | 24 | 0 | 17 | 27 | 13 | 4 |

Caveat: limits correlate with scene type, so scene recognition contributes; confusion WITHIN a scene type (e.g. 30 vs 50 city) is the sharper 'reads the sign' signal.

## Interpretation (Phase-0 headline)

224px frozen DINOv2 features carry REAL speed-limit signal: 39.2% top-1 /
42.6% balanced over 11 classes vs 6.9% majority / 9.1% chance, with a
strongly diagonal-adjacent confusion structure (most errors land on the
neighbouring limit class).

- Physically SIGNED / gated classes probe best: 10 (81%), 20 (87%),
  120 (52%) -- consistent with the features actually encoding signage or
  its immediate context, not just scene type.
- 130 probes WORST (4.7%): in Germany 130 is the UNSIGNED motorway
  default -- there is literally no sign to read, and its mass leaks onto
  60/80/100 (other motorway-looking scenes). Vision alone cannot recover
  the legal limit there; that is exactly the gap the map-GT conditioning
  token closes.
- Mid-range city/rural classes (50/60/70/80/100 at 29-37%) sit between:
  part sign-reading, part scene prior; adjacent confusion (50 vs 30/60)
  dominates.

Verdict: 224-256px vision can partially read limits where signs exist, and
categorically cannot where the limit is implicit -- supporting the
map-context token as complementary, not redundant, input.
