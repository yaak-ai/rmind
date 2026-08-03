# CFG amplification sweep — armMV v1 (model-0nr1ydjm:v1), map-context PatchPolicy

`diag_results/eval_v0/cfg_sweep_mv_v1.npz` — 768 val samples (24x32 cached batches, micro-batch 8, kitkat),
argmax decoding, last-frame readout, action chunk horizon 6, 4 quantizers x 16 codes.

Guided logits per quantizer: `l_g = l_u + w*(l_c - l_u)`, `l_u` = UNKNOWN (override=None; the cached val batches carry all-NaN `max_speed`, so this IS the unconditional path — verified bitwise in the earlier NaN-flood check), `l_c` = `max_speed_override=<cond>`. Guided codes = `argmax(l_g)`; actions = the model's own decode `tokenizer.invert(codes_g) + _offset(offsets, codes_g)`, i.e. the real offset head at the guided codes (the decode path re-enters cleanly — no fallback needed).

Offset policies: **primary** = conditional `offsets_c` at guided codes; `offu` = unconditional offsets at guided codes (pure code-flip effect); `offcfg` = `offsets_u + w*(offsets_c - offsets_u)` (full CFG, incl. offsets).

Anchors: w=0 code mismatches vs baseline = 0 (must be 0); w=1 max |chunk - plain_override_chunk| = 0 (must be 0).

All actions are in the tokenizer's NORMALIZED space (gas/brake ~[0,1], steer ~[-1,1]); deltas are means over samples x horizon vs the UNKNOWN baseline.

## cond = 30

| w | flip% | q0 | q1 | q2 | q3 | Δgas (±SE) | Δbrake (±SE) | Δsteer | Δgas·flipped | Δbrake·flipped | TS flip% | viol>0.05 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -2.6e-07 | n/a | n/a | 0.00 | 0.00 |
| 1 | 0.52 | 0.78 | 0.52 | 0.39 | 0.39 | +0.0009 ± 0.0003 | +0.0000 ± 0.0000 | -0.0001 | +0.0426 | +0.0004 | 0.02 | 0.00 |
| 2 | 0.98 | 0.91 | 0.91 | 0.52 | 1.56 | +0.0012 ± 0.0004 | +0.0000 ± 0.0000 | -0.0001 | +0.0299 | +0.0003 | 0.04 | 0.00 |
| 3 | 1.43 | 1.30 | 1.30 | 0.65 | 2.47 | +0.0016 ± 0.0005 | +0.0000 ± 0.0000 | -0.0001 | +0.0299 | +0.0002 | 0.04 | 0.00 |
| 5 | 2.34 | 2.21 | 2.21 | 1.30 | 3.65 | +0.0015 ± 0.0007 | +0.0002 ± 0.0001 | +0.0002 | +0.0171 | +0.0019 | 0.17 | 0.00 |
| 8 | 3.71 | 3.78 | 3.65 | 2.47 | 4.95 | +0.0025 ± 0.0008 | +0.0002 ± 0.0001 | -0.0004 | +0.0196 | +0.0013 | 0.54 | 0.00 |
| 12 | 5.99 | 5.60 | 6.90 | 4.04 | 7.42 | +0.0034 ± 0.0010 | +0.0002 ± 0.0003 | -0.0013 | +0.0178 | +0.0012 | 0.67 | 0.00 |

## cond = 5(WALK)

| w | flip% | q0 | q1 | q2 | q3 | Δgas (±SE) | Δbrake (±SE) | Δsteer | Δgas·flipped | Δbrake·flipped | TS flip% | viol>0.05 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.0003 ± 0.0001 | -0.0000 ± 0.0001 | -9.5e-06 | n/a | n/a | 0.00 | 0.00 |
| 1 | 16.50 | 16.15 | 17.06 | 11.98 | 20.83 | +0.0143 ± 0.0018 | -0.0002 ± 0.0006 | -0.0020 | +0.0333 | -0.0004 | 1.65 | 0.04 |
| 2 | 35.51 | 34.24 | 36.59 | 31.12 | 40.10 | +0.0280 ± 0.0031 | +0.0025 ± 0.0013 | -0.0039 | +0.0434 | +0.0039 | 4.54 | 0.18 |
| 3 | 46.13 | 43.49 | 48.31 | 43.10 | 49.61 | +0.0419 ± 0.0041 | +0.0040 ± 0.0022 | -0.0047 | +0.0580 | +0.0056 | 7.03 | 0.30 |
| 5 | 59.57 | 56.38 | 60.68 | 58.07 | 63.15 | +0.0466 ± 0.0051 | +0.0166 ± 0.0034 | -0.0065 | +0.0573 | +0.0205 | 11.15 | 0.61 |
| 8 | 69.86 | 66.41 | 72.01 | 69.01 | 72.01 | +0.0413 ± 0.0055 | +0.0266 ± 0.0042 | -0.0137 | +0.0465 | +0.0299 | 15.76 | 0.85 |
| 12 | 77.73 | 73.83 | 78.78 | 77.99 | 80.34 | +0.0368 ± 0.0058 | +0.0356 ± 0.0048 | -0.0200 | +0.0395 | +0.0382 | 19.94 | 0.98 |

## cond = 100

| w | flip% | q0 | q1 | q2 | q3 | Δgas (±SE) | Δbrake (±SE) | Δsteer | Δgas·flipped | Δbrake·flipped | TS flip% | viol>0.05 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | -0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +7.8e-07 | n/a | n/a | 0.00 | 0.00 |
| 1 | 0.26 | 0.00 | 0.39 | 0.39 | 0.26 | +0.0001 ± 0.0001 | +0.0000 ± 0.0000 | -0.0001 | +0.0117 | +0.0006 | 0.02 | 0.00 |
| 2 | 0.42 | 0.13 | 0.39 | 0.65 | 0.52 | +0.0000 ± 0.0002 | +0.0000 ± 0.0000 | -0.0001 | +0.0009 | +0.0004 | 0.04 | 0.00 |
| 3 | 0.62 | 0.39 | 0.52 | 0.78 | 0.78 | +0.0003 ± 0.0003 | +0.0000 ± 0.0000 | -0.0001 | +0.0120 | +0.0014 | 0.04 | 0.00 |
| 5 | 1.01 | 0.52 | 0.78 | 1.30 | 1.43 | +0.0004 ± 0.0003 | +0.0000 ± 0.0000 | +0.0005 | +0.0104 | +0.0010 | 0.09 | 0.00 |
| 8 | 1.50 | 0.91 | 1.56 | 1.95 | 1.56 | +0.0009 ± 0.0005 | +0.0001 ± 0.0000 | +0.0003 | +0.0147 | +0.0013 | 0.09 | 0.00 |
| 12 | 2.21 | 1.04 | 2.47 | 2.60 | 2.73 | +0.0013 ± 0.0005 | +0.0001 ± 0.0001 | +0.0006 | +0.0153 | +0.0011 | 0.09 | 0.00 |

## cond = -1(UNLIMITED)

| w | flip% | q0 | q1 | q2 | q3 | Δgas (±SE) | Δbrake (±SE) | Δsteer | Δgas·flipped | Δbrake·flipped | TS flip% | viol>0.05 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | +3.4e-06 | n/a | n/a | 0.00 | 0.00 |
| 1 | 1.20 | 1.17 | 1.17 | 1.30 | 1.17 | +0.0006 ± 0.0004 | +0.0001 ± 0.0001 | -0.0006 | +0.0117 | +0.0020 | 0.02 | 0.00 |
| 2 | 2.70 | 2.86 | 2.99 | 1.69 | 3.26 | +0.0016 ± 0.0007 | +0.0002 ± 0.0001 | -0.0004 | +0.0159 | +0.0023 | 0.30 | 0.00 |
| 3 | 3.39 | 3.39 | 4.17 | 1.95 | 4.04 | +0.0026 ± 0.0009 | +0.0002 ± 0.0001 | -0.0004 | +0.0223 | +0.0015 | 0.33 | 0.00 |
| 5 | 5.60 | 6.12 | 6.77 | 3.39 | 6.12 | +0.0052 ± 0.0011 | -0.0001 ± 0.0003 | -0.0008 | +0.0306 | -0.0007 | 0.52 | 0.00 |
| 8 | 8.43 | 8.98 | 10.29 | 5.60 | 8.85 | +0.0089 ± 0.0014 | +0.0001 ± 0.0004 | -0.0010 | +0.0383 | +0.0004 | 0.76 | 0.00 |
| 12 | 12.24 | 12.24 | 13.80 | 8.98 | 13.93 | +0.0112 ± 0.0016 | +0.0001 ± 0.0006 | -0.0030 | +0.0374 | +0.0003 | 1.02 | 0.02 |

Baseline (UNKNOWN) reference: mean gas 0.1116, brake 0.0333, steer -0.0036; material violations (>0.05 outside the box) 0.00% of slots, mean excess 0.00028; any-violation (incl. hairline) 94.5% of samples — the baseline itself, so only the >0.05 column tracks CFG-induced degeneracy.

## Headline — guided-30 minus guided-100 on speed > 50 km/h (n=213)

A policy that READS the token should show **negative** Δgas and **positive** Δbrake here. Paired per-sample contrast (same samples, mean over horizon); +- is 1 paired SE.

| w | Δgas (30-100) | Δbrake (30-100) | Δsteer (30-100) | flip% 30 | flip% 100 |
|---|---|---|---|---|---|
| 0 | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 0.00 | 0.00 |
| 1 | +0.0011 ± 0.0006 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 0.82 | 0.35 |
| 2 | +0.0021 ± 0.0009 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 1.64 | 0.59 |
| 3 | +0.0033 ± 0.0014 | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | 2.70 | 0.70 |
| 5 | +0.0029 ± 0.0020 | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | 4.58 | 1.29 |
| 8 | +0.0053 ± 0.0022 | +0.0000 ± 0.0000 | -0.0000 ± 0.0001 | 6.92 | 1.64 |
| 12 | +0.0070 ± 0.0028 | +0.0007 ± 0.0007 | +0.0001 ± 0.0001 | 11.03 | 2.23 |

## WALK(5) minus 100 on speed > 50 km/h (n=213)

| w | Δgas | Δbrake | Δsteer |
|---|---|---|---|
| 0 | +0.0007 ± 0.0002 | +0.0000 ± 0.0000 | +0.0002 ± 0.0001 |
| 1 | +0.0286 ± 0.0045 | +0.0013 ± 0.0011 | +0.0002 ± 0.0001 |
| 2 | +0.0503 ± 0.0071 | +0.0044 ± 0.0018 | -0.0009 ± 0.0013 |
| 3 | +0.0661 ± 0.0090 | +0.0093 ± 0.0028 | +0.0017 ± 0.0030 |
| 5 | +0.0609 ± 0.0116 | +0.0303 ± 0.0070 | +0.0026 ± 0.0086 |
| 8 | +0.0461 ± 0.0122 | +0.0459 ± 0.0085 | -0.0361 ± 0.0159 |
| 12 | +0.0315 ± 0.0133 | +0.0560 ± 0.0089 | -0.0469 ± 0.0187 |

## Per-sample direction split (material moves, |Δ| > 0.05 on the horizon-mean pedal, vs the UNKNOWN baseline)

Population means can hide opposite per-sample moves, so: share of the 768 samples whose gas rises / brake rises / both rise materially, plus the share of (sample, step) slots with simultaneously high pedals (gas>0.1 AND brake>0.1) — a physical-incoherence indicator (baseline: 0.00%).

| cond | w | gas↑ only % | brake↑ only % | both↑ % | gas↓ % | both-pedals-high % |
|---|---|---|---|---|---|---|
| 30 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 1 | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 2 | 1.17 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 | 3 | 1.69 | 0.00 | 0.00 | 0.13 | 0.00 |
| 30 | 5 | 2.34 | 0.00 | 0.00 | 0.65 | 0.00 |
| 30 | 8 | 3.39 | 0.00 | 0.00 | 1.04 | 0.00 |
| 30 | 12 | 5.08 | 0.26 | 0.00 | 1.69 | 0.00 |
| 5(WALK) | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5(WALK) | 1 | 13.93 | 1.17 | 0.00 | 3.39 | 0.04 |
| 5(WALK) | 2 | 24.74 | 4.30 | 0.13 | 7.68 | 0.15 |
| 5(WALK) | 3 | 29.95 | 4.82 | 0.13 | 9.24 | 0.33 |
| 5(WALK) | 5 | 33.85 | 8.85 | 0.52 | 14.19 | 0.67 |
| 5(WALK) | 8 | 35.03 | 13.28 | 0.78 | 18.75 | 1.15 |
| 5(WALK) | 12 | 35.16 | 16.80 | 0.65 | 22.66 | 1.11 |
| 100 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 1 | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 2 | 0.13 | 0.00 | 0.00 | 0.13 | 0.00 |
| 100 | 3 | 0.39 | 0.00 | 0.00 | 0.13 | 0.00 |
| 100 | 5 | 0.39 | 0.00 | 0.00 | 0.13 | 0.00 |
| 100 | 8 | 0.78 | 0.00 | 0.00 | 0.39 | 0.00 |
| 100 | 12 | 1.17 | 0.13 | 0.00 | 0.39 | 0.00 |
| -1(UNLIMITED) | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| -1(UNLIMITED) | 1 | 1.17 | 0.00 | 0.00 | 0.13 | 0.00 |
| -1(UNLIMITED) | 2 | 2.47 | 0.13 | 0.00 | 0.65 | 0.00 |
| -1(UNLIMITED) | 3 | 3.12 | 0.13 | 0.00 | 0.65 | 0.00 |
| -1(UNLIMITED) | 5 | 5.86 | 0.26 | 0.00 | 0.78 | 0.00 |
| -1(UNLIMITED) | 8 | 8.20 | 0.52 | 0.00 | 1.17 | 0.00 |
| -1(UNLIMITED) | 12 | 10.16 | 0.65 | 0.00 | 1.69 | 0.00 |

## Offset-policy attribution (mean Δgas / Δbrake vs baseline, all samples)

| cond | w | primary (offsets_c) | offu (code-flip only) | offcfg (full CFG) |
|---|---|---|---|---|
| 30 | 0 | +0.0000 / -0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| 30 | 1 | +0.0009 / +0.0000 | +0.0009 / +0.0000 | +0.0009 / +0.0000 |
| 30 | 2 | +0.0012 / +0.0000 | +0.0012 / +0.0000 | +0.0012 / +0.0000 |
| 30 | 3 | +0.0016 / +0.0000 | +0.0016 / +0.0000 | +0.0017 / +0.0000 |
| 30 | 5 | +0.0015 / +0.0002 | +0.0015 / +0.0002 | +0.0015 / +0.0002 |
| 30 | 8 | +0.0025 / +0.0002 | +0.0024 / +0.0002 | +0.0025 / +0.0002 |
| 30 | 12 | +0.0034 / +0.0002 | +0.0034 / +0.0002 | +0.0035 / +0.0002 |
| 5(WALK) | 0 | +0.0003 / -0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| 5(WALK) | 1 | +0.0143 / -0.0002 | +0.0140 / -0.0001 | +0.0143 / -0.0002 |
| 5(WALK) | 2 | +0.0280 / +0.0025 | +0.0274 / +0.0026 | +0.0286 / +0.0024 |
| 5(WALK) | 3 | +0.0419 / +0.0040 | +0.0412 / +0.0040 | +0.0433 / +0.0039 |
| 5(WALK) | 5 | +0.0466 / +0.0166 | +0.0456 / +0.0169 | +0.0508 / +0.0157 |
| 5(WALK) | 8 | +0.0413 / +0.0266 | +0.0403 / +0.0268 | +0.0487 / +0.0253 |
| 5(WALK) | 12 | +0.0368 / +0.0356 | +0.0357 / +0.0358 | +0.0487 / +0.0333 |
| 100 | 0 | -0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| 100 | 1 | +0.0001 / +0.0000 | +0.0001 / +0.0000 | +0.0001 / +0.0000 |
| 100 | 2 | +0.0000 / +0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| 100 | 3 | +0.0003 / +0.0000 | +0.0003 / +0.0000 | +0.0003 / +0.0000 |
| 100 | 5 | +0.0004 / +0.0000 | +0.0004 / +0.0000 | +0.0004 / +0.0000 |
| 100 | 8 | +0.0009 / +0.0001 | +0.0009 / +0.0001 | +0.0008 / +0.0001 |
| 100 | 12 | +0.0013 / +0.0001 | +0.0013 / +0.0001 | +0.0013 / +0.0001 |
| -1(UNLIMITED) | 0 | +0.0000 / -0.0000 | +0.0000 / +0.0000 | +0.0000 / +0.0000 |
| -1(UNLIMITED) | 1 | +0.0006 / +0.0001 | +0.0006 / +0.0001 | +0.0006 / +0.0001 |
| -1(UNLIMITED) | 2 | +0.0016 / +0.0002 | +0.0015 / +0.0002 | +0.0016 / +0.0002 |
| -1(UNLIMITED) | 3 | +0.0026 / +0.0002 | +0.0025 / +0.0002 | +0.0026 / +0.0002 |
| -1(UNLIMITED) | 5 | +0.0052 / -0.0001 | +0.0052 / -0.0001 | +0.0054 / -0.0001 |
| -1(UNLIMITED) | 8 | +0.0089 / +0.0001 | +0.0089 / +0.0001 | +0.0090 / +0.0001 |
| -1(UNLIMITED) | 12 | +0.0112 / +0.0001 | +0.0112 / +0.0001 | +0.0114 / +0.0001 |

## Artifact indicators — material box violations (% of (sample, step, field) slots more than 0.05 outside the valid range)

| cond | w | primary | offu | offcfg | mean excess (primary) |
|---|---|---|---|---|---|
| 30 | 0 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 30 | 1 | 0.00 | 0.00 | 0.00 | 0.00027 |
| 30 | 2 | 0.00 | 0.00 | 0.00 | 0.00027 |
| 30 | 3 | 0.00 | 0.00 | 0.00 | 0.00027 |
| 30 | 5 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 30 | 8 | 0.00 | 0.00 | 0.00 | 0.00027 |
| 30 | 12 | 0.00 | 0.00 | 0.00 | 0.00026 |
| 5(WALK) | 0 | 0.00 | 0.00 | 0.00 | 0.00029 |
| 5(WALK) | 1 | 0.04 | 0.04 | 0.04 | 0.00029 |
| 5(WALK) | 2 | 0.18 | 0.15 | 0.22 | 0.00052 |
| 5(WALK) | 3 | 0.30 | 0.36 | 0.35 | 0.00074 |
| 5(WALK) | 5 | 0.61 | 0.69 | 0.84 | 0.00119 |
| 5(WALK) | 8 | 0.85 | 0.93 | 1.61 | 0.00161 |
| 5(WALK) | 12 | 0.98 | 1.05 | 2.66 | 0.00177 |
| 100 | 0 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 100 | 1 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 100 | 2 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 100 | 3 | 0.00 | 0.00 | 0.00 | 0.00027 |
| 100 | 5 | 0.00 | 0.00 | 0.00 | 0.00028 |
| 100 | 8 | 0.00 | 0.00 | 0.00 | 0.00029 |
| 100 | 12 | 0.00 | 0.00 | 0.00 | 0.00027 |
| -1(UNLIMITED) | 0 | 0.00 | 0.00 | 0.00 | 0.00028 |
| -1(UNLIMITED) | 1 | 0.00 | 0.00 | 0.00 | 0.00026 |
| -1(UNLIMITED) | 2 | 0.00 | 0.00 | 0.00 | 0.00024 |
| -1(UNLIMITED) | 3 | 0.00 | 0.00 | 0.00 | 0.00024 |
| -1(UNLIMITED) | 5 | 0.00 | 0.00 | 0.00 | 0.00026 |
| -1(UNLIMITED) | 8 | 0.00 | 0.00 | 0.02 | 0.00025 |
| -1(UNLIMITED) | 12 | 0.02 | 0.01 | 0.04 | 0.00027 |

## Critical-w geometry (from the dumped logits, no decode involved)

Per (sample, quantizer): the smallest w at which some code overtakes the UNKNOWN top-1. Percentiles are over slots where a finite crossing exists.

| cond | finite-crossing slots % | crit-w p1 | p5 | p25 | p50 |
|---|---|---|---|---|---|
| 30 | 94.7 | 1.83 | 9.49 | 56.71 | 167.59 |
| 5(WALK) | 98.4 | 0.07 | 0.28 | 1.40 | 3.35 |
| 100 | 94.5 | 4.51 | 20.94 | 95.96 | 230.21 |
| -1(UNLIMITED) | 97.9 | 0.68 | 4.22 | 30.05 | 91.34 |

## Verdict

CFG is mechanically live and correctly plumbed — w=0 reproduces the baseline codes exactly, w=1 is bit-identical to the plain-override run, and guided code flips grow monotonically with w (cond 30: 0.5% -> 6.0%, UNLIMITED: 1.2% -> 12.2%, WALK: 16.5% -> 77.7% of (sample, quantizer) slots) — so the weak conditioning CAN be amplified into real, decodable action changes.
But the amplified direction is the WRONG one: guided-30 minus guided-100 on speed>50 frames grows from +0.0011 ± 0.0006 to +0.0070 ± 0.0028 gas (more throttle in the slower zone, ~2.5 paired SE from zero) while brake stays at noise (+0.0007 ± 0.0007), i.e. CFG scales up the token's regime association, not speed compliance — the eval_v0 defect grows ≈6x from w=1 to w=12 instead of reversing.
The only strongly responsive class, WALK, does shift population brake up with w (Δbrake +0.017 → +0.036 while Δgas recedes from its w=5 peak +0.047 → +0.037), but per sample it is a diffuse regime split, not compliance: at w=12, 35% of samples get materially MORE throttle vs 17% more brake (both rise on only 0.7%), simultaneous high-gas-and-high-brake slots appear where the baseline had none (0 → 1.1%), and all of it sits past 60-78% code flips, 20% turn-signal flips and material out-of-box action slots rising from ~0 to 1.0% (2.7% if offsets are guided too) — decode degeneracy.
The logit geometry says this is structural rather than a bad choice of w: the median (sample, quantizer) needs w ≈ 168 (cond 30) or w ≈ 230 (cond 100) — p25 still 57 / 96 — before any code overtakes the UNKNOWN top-1, so the numeric speed classes are ~1.5-2 orders of magnitude too weak for guidance in the usable range, and the offset head is irrelevant here (primary ≈ offsets-frozen variant to 4 decimals; extrapolating offsets only adds artifacts, `offset_scale=None` so nothing saturates them).
**No usable w exists** — every w that moves actions moves them the wrong way or into degeneracy, so a serving-side CFG trick cannot rescue zone conditioning on this checkpoint; the levers stay the training-side ones (compliance-conditioned targets, rare-class oversampling, stronger aux supervision).
