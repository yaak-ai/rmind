# Override probe — armMV v1 (model-0nr1ydjm:v1, epoch-2), 768 val samples, argmax

Same batches/protocol as eval_v0 (probe_dump_mv_v1.npz; micro-batch 8 on kitkat).

| cond | KL vs None | code flip % | Δgas | Δbrake | Δsteer |
|---|---|---|---|---|---|
| WALK(5) | 0.2075 | 16.50 | +0.0167 | +0.0014 | −0.0032 |
| UNLIMITED(−1) | 0.0015 | 1.20 | +0.0005 | +0.0001 | −0.0006 |
| 10 | 0.0004 | 0.75 | +0.0004 | +0.0001 | −0.0002 |
| 30 | 0.0002 | 0.52 | +0.0008 | +0.0000 | −0.0001 |
| 100 | 0.0000 | 0.26 | +0.0001 | +0.0000 | −0.0002 |
| 50 | 0.0000 | 0.42 | +0.0000 | +0.0000 | −0.0001 |
| NaNflood | 0.0000 | 0.00 | ±0 (≡ None, exact) | | |

Headline (speed>50, n=213): 30-vs-100 Δgas **+0.0010**, Δbrake +0.0000 — still no correct
directional compliance signal (and the tiny gas delta points the WRONG way).

## vs v0 (same protocol)
- WALK sensitivity ~2×: KL 0.111→0.208, flips 14.0→16.5%, Δgas +0.006→+0.0167 — the token
  matters MORE with training but pushes gas UP under a walking-pace token. Likely a
  regime association: WALK frames in training are parking/creep maneuvers where
  demonstrators apply gas from near-rest; class has only 52k frames.
- 30/50/100 remain ~inert (KL ≤ 2e-4). UNLIMITED halved (0.0033→0.0015).
- Verdict: v1 = more responsive, not more correct. BC epochs strengthen the token's
  regime association, not speed compliance. Reinforces the eval_v0 conclusion: the
  levers are compliance conditioning + rare-class oversampling (+ aux weight), not
  more epochs.
