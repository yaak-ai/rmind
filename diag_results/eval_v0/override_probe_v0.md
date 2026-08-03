# eval_v0: counterfactual override probe + warm-start fidelity

Samples: 768 (last-frame readout, argmax decoding, seed 1337, real val batches, shared across all checkpoints/conditions).
Vehicle speed (last frame): mean 35.2, p50 32.9, p90 65.3, max 95.5 (units as stored; >50 subset n=213).
Batch map GT max_speed (last frame): known 0/768 (0.0%)

## armM: override sweep (deltas vs None baseline)

None(UNKNOWN) decoded means: gas 0.1056, brake 0.0369, steer -0.0060

| override | gas | brake | steer | dgas | dbrake | dsteer | |dgas| | |dbrake| | KL(None‖ov) | code_flips | max|dlogit| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5(WALK) | 0.1056 | 0.0369 | -0.0060 | +0.0001 | -0.0000 | -0.0000 | 0.0005 | 0.0000 | 0.0002 | 0.0016 | 3.264 |
| 10 | 0.1056 | 0.0369 | -0.0058 | +0.0000 | +0.0000 | +0.0002 | 0.0001 | 0.0002 | 0.0003 | 0.0020 | 5.946 |
| 30 | 0.1056 | 0.0369 | -0.0060 | +0.0000 | +0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.431 |
| 50 | 0.1056 | 0.0369 | -0.0060 | +0.0000 | +0.0000 | -0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0010 | 0.238 |
| 100 | 0.1056 | 0.0369 | -0.0060 | +0.0000 | +0.0000 | -0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0007 | 0.411 |
| -1(UNLIMITED) | 0.1058 | 0.0370 | -0.0057 | +0.0003 | +0.0001 | +0.0002 | 0.0004 | 0.0002 | 0.0008 | 0.0039 | 8.441 |

Sanity: identical override pairs: none (PASS); max|logit(None) - logit(all-NaN max_speed)| = 0 (None == UNKNOWN flood, PASS)

### armM: speed-conditioned contrast (headline)

| subset | n | gas@30 | gas@100 | gas 30-100 | brake@30 | brake@100 | brake 30-100 |
|---|---|---|---|---|---|---|---|
| all | 768 | 0.1056 | 0.1056 | +0.0000 | 0.0369 | 0.0369 | +0.0000 |
| speed>50 | 213 | 0.1752 | 0.1752 | -0.0000 | -0.0001 | -0.0001 | -0.0000 |
| speed>70 | 55 | 0.2111 | 0.2111 | +0.0000 | -0.0000 | -0.0000 | -0.0000 |
| speed<=50 | 555 | 0.0789 | 0.0789 | +0.0000 | 0.0511 | 0.0511 | +0.0000 |

## armMV: override sweep (deltas vs None baseline)

None(UNKNOWN) decoded means: gas 0.1151, brake 0.0363, steer -0.0001

| override | gas | brake | steer | dgas | dbrake | dsteer | |dgas| | |dbrake| | KL(None‖ov) | code_flips | max|dlogit| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5(WALK) | 0.1171 | 0.0355 | -0.0006 | +0.0020 | -0.0008 | -0.0005 | 0.0212 | 0.0051 | 0.1110 | 0.1396 | 33.453 |
| 10 | 0.1148 | 0.0363 | -0.0002 | -0.0002 | +0.0000 | -0.0000 | 0.0015 | 0.0001 | 0.0004 | 0.0065 | 3.537 |
| 30 | 0.1153 | 0.0364 | -0.0004 | +0.0003 | +0.0001 | -0.0003 | 0.0005 | 0.0001 | 0.0002 | 0.0039 | 1.877 |
| 50 | 0.1149 | 0.0365 | -0.0002 | -0.0002 | +0.0002 | -0.0001 | 0.0005 | 0.0003 | 0.0001 | 0.0039 | 2.168 |
| 100 | 0.1150 | 0.0363 | -0.0005 | -0.0000 | +0.0000 | -0.0004 | 0.0003 | 0.0000 | 0.0000 | 0.0029 | 0.260 |
| -1(UNLIMITED) | 0.1153 | 0.0359 | -0.0006 | +0.0002 | -0.0004 | -0.0005 | 0.0017 | 0.0011 | 0.0033 | 0.0133 | 7.301 |

Sanity: identical override pairs: none (PASS); max|logit(None) - logit(all-NaN max_speed)| = 0 (None == UNKNOWN flood, PASS)

### armMV: speed-conditioned contrast (headline)

| subset | n | gas@30 | gas@100 | gas 30-100 | brake@30 | brake@100 | brake 30-100 |
|---|---|---|---|---|---|---|---|
| all | 768 | 0.1153 | 0.1150 | +0.0003 | 0.0364 | 0.0363 | +0.0000 |
| speed>50 | 213 | 0.1887 | 0.1886 | +0.0001 | 0.0006 | 0.0006 | -0.0000 |
| speed>70 | 55 | 0.2229 | 0.2238 | -0.0009 | -0.0000 | -0.0000 | -0.0000 |
| speed<=50 | 555 | 0.0872 | 0.0868 | +0.0004 | 0.0501 | 0.0500 | +0.0001 |

## Warm-start fidelity: arm None-condition vs parent (same batches)

Parent decoded means: gas 0.1084, brake 0.0322, steer -0.0100

| arm | dgas | dbrake | dsteer | |dgas| | |dbrake| | |dsteer| | code_agree | KL(parent‖arm) | max|dlogit| |
|---|---|---|---|---|---|---|---|---|---|
| armM | -0.0028 | +0.0047 | +0.0040 | 0.0240 | 0.0101 | 0.0174 | 0.7868 | 0.1916 | 47.921 |
| armMV | +0.0067 | +0.0041 | +0.0099 | 0.0258 | 0.0101 | 0.0214 | 0.7865 | 0.1955 | 50.929 |

## Extreme-override contrasts by speed subset (signed deltas vs None)

| arm | subset | n | WALK dgas | WALK dbrake | UNLIM dgas | UNLIM dbrake | 10 dgas | 10 dbrake |
|---|---|---|---|---|---|---|---|---|
| armM | all | 768 | +0.0001 | -0.0000 | +0.0003 | +0.0001 | +0.0000 | +0.0000 |
| armM | speed>50 | 213 | +0.0002 | +0.0000 | +0.0001 | +0.0000 | +0.0001 | +0.0000 |
| armM | speed<=50 | 555 | -0.0000 | -0.0000 | +0.0004 | +0.0001 | -0.0000 | +0.0000 |
| armMV | all | 768 | +0.0020 | -0.0008 | +0.0002 | -0.0004 | -0.0002 | +0.0000 |
| armMV | speed>50 | 213 | +0.0064 | +0.0002 | +0.0009 | -0.0007 | -0.0010 | +0.0000 |
| armMV | speed<=50 | 555 | +0.0003 | -0.0012 | -0.0001 | -0.0002 | +0.0000 | +0.0000 |

armMV WALK per-sample code-flip rate: mean 0.140, frac samples with >=1 flip 0.387
  by speed: >50 km/h mean 0.183; <=50 mean 0.123

## armM v1 (epoch-2) override sweep, same 768 samples
None means: gas 0.1065 brake 0.0336 steer -0.0017
| override | dgas | dbrake | |dgas| | KL | flips | max|dlogit| |
|---|---|---|---|---|---|---|
| 5(WALK) | -0.0000 | +0.0002 | 0.0007 | 0.0004 | 0.0055 | 4.89 |
| 10 | +0.0001 | +0.0001 | 0.0018 | 0.0025 | 0.0114 | 8.31 |
| 30 | -0.0000 | -0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.52 |
| 50 | +0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.32 |
| 100 | +0.0000 | +0.0000 | 0.0001 | 0.0000 | 0.0013 | 1.06 |
| -1(UNLIMITED) | -0.0002 | +0.0001 | 0.0002 | 0.0001 | 0.0013 | 3.78 |
30-vs-100 all (n=768): dgas -0.00002 dbrake -0.00003
30-vs-100 speed>50 (n=213): dgas -0.00000 dbrake +0.00000
30-vs-100 speed>70 (n=55): dgas -0.00000 dbrake -0.00000
None-vs-NaNflood max|dlogit| = 0
vs parent: dgas -0.0019 dbrake +0.0014 |dgas| 0.0227 code_agree 0.7865 KL 0.1858
