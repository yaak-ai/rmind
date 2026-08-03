# eval_v0 synthesis: Phase-2 offline eval of the epoch-1 map-arm checkpoints

Checkpoints: Arm M `model-1n0ih44y:v0` (dinov2_dinowm_maxspeed_warmstart,
epoch 1), Arm MV `model-0nr1ydjm:v0` (+aux head, epoch 1), parent
`model-ifuusvwq:v8` (dinowm winner). A v1 (epoch-2) artifact for Arm M
appeared mid-suite (06:38Z) and was probed as a bonus; Arm MV has no v1.
All numbers: real val batches, argmax decoding, seed 1337, identical inputs
across checkpoints (768 samples for the shared-subset comparisons; 6400 for
the Arm M standing run).

## Verdict in one line

The conditioning seam is mechanically live but behaviorally INERT at
epoch 1 -- no directional speed-limit response yet in either arm; driving
quality did NOT regress (both arms slightly beat the parent); Arm MV is the
stronger arm on every axis that currently separates them.

## 1. Counterfactual conditioning: not directionally correct yet

- THE headline check -- on frames with vehicle speed > 50 km/h, does
  override=30 raise brake / lower gas vs override=100? **No.** Deltas are
  ~1e-4 in decoded units in both arms (armM: dgas -0.0000, dbrake -0.0000
  at n=213; armMV: dgas +0.0001, dbrake -0.0000). Even WALK(5 km/h) at
  speed>50 moves gas the WRONG way (+0.006) in Arm MV.
- Mechanics all PASS in both arms: every override pair produces distinct
  logits, and override=None is bitwise-identical to an all-NaN max_speed
  input (None == UNKNOWN flood).
- Sensitivity (KL, code flips) tells the arms apart sharply:
  - Arm M is dead: KL <= 8e-4, code flips <= 0.4% for EVERY override.
  - Arm MV reacts at the vocabulary extremes: WALK KL 0.111, 14.0% code
    flips (39% of samples flip >= 1 code; 18.3% flip rate on fast frames),
    UNLIMITED KL 0.0033 / 1.3% flips -- ~100x Arm M. The aux head is doing
    what it was added to do (couple the token into the trunk); the coupling
    just has not reached the action head with any speed-consistent sign.
- Bonus Arm M v1 (epoch-2, same 768 samples): still inert (max KL 0.0025 on
  override=10; 30-vs-100 contrast ~1e-5). One more epoch did not wake up
  Arm M's token.

## 2. Driving regression vs parent: none

Same-subset (768 samples) last-frame, argmax protocol
(parent / armM / armMV):

- joint top-1 (exact behavior token): 0.0560 / 0.0638 / **0.0729**
- marginal top-1: 0.4040 / 0.4160 / **0.4189**
- argmax recon L1: 0.0464 / 0.0443 / **0.0435**

Arm M cross-checked on the full 6400-sample standing run: top1 0.4123,
joint 0.0602, argmax recon 0.0470 (subset representative). Cluster-level
movements (braking better in both arms, highway gas slightly worse in
Arm M) are within noise at n=36-84. Sampled-decoding columns flatter the
parent because the arms run hotter (entropy 0.79 vs 0.65 nats) -- argmax is
the standing verdict column.

## 3. Aux head (Arm MV): unmeasurable on val, inflated on train

- Train aux acc ~0.98 (aux loss 0.011, known_frac ~0.65-0.67) -- far above
  the 39.2% linear-probe floor and 6.9% majority baseline, but NOT
  comparable: with 0.3 input dropout the GT token is visible in the input
  on ~70% of frames, so ~98% is consistent with near-perfect copying.
  Dropped-frame-only accuracy (the real "vision reads the limit" number) is
  not logged -- gap.
- Val aux metrics are all zero because **none of the 5 val drives have
  map-GT sidecars** (never attempted in the 638-drive build) -- the val
  split carries max_speed = NaN everywhere (confirmed empirically: 0/768
  known in the probe batches). Second gap; also means training never
  validated the token path on known targets.

## 4. UNKNOWN-flood / warm-start fidelity: intact, moderate drift

Since the val split is all-UNKNOWN, override=None IS the UNKNOWN-flood
condition. Vs the parent on identical batches (768 samples, last frame):

- Arm M: code agreement 78.7%, KL(parent||arm) 0.192, mean dgas -0.003,
  dbrake +0.005 (|dgas| 0.024)
- Arm MV: code agreement 78.7%, KL 0.196, dgas +0.007, dbrake +0.004
  (|dgas| 0.026)
- Arm M v1: agreement 78.7%, KL 0.186 -- stable across the extra epoch.

The arms under missing map input still behave like the parent (~79% of
codes identical, tiny mean action shifts) while scoring slightly BETTER on
GT -- the warm start held; the drift is ordinary fine-tuning drift, not
UNKNOWN-flood damage.

## Which arm is stronger

**Arm MV.** Best joint top-1 (+1.7pt over parent, +0.9pt over Arm M), best
argmax recon, and the only arm whose trunk measurably reads the token
(100x Arm M's counterfactual sensitivity). Cost: none visible. Arm M at
epoch 2 (v1) is still token-dead, strengthening the case that the aux
pressure, not just more epochs, is what wires the token in.

## Epoch-1 caveats + what to re-run on final checkpoints

These are epoch-1 (of 5) checkpoints with zero-init token embeddings; the
action-conditioning verdict is expected to be pending, not failed. Before /
at the final checkpoints:

1. Build map-GT sidecars for the 5 val drives (Niro115-HQ/2023-05-16,
   Niro104-HQ/2022-12-20, Niro107-HQ/2023-05-12, Niro122-HQ/2023-04-05,
   Niro102-HQ/2022-12-03) so val aux metrics and known-token val behavior
   become measurable.
2. Log aux acc split by token-visible vs token-dropped frames (+ per-class).
3. Re-run the override probe (eval_v0_dump + batch cache, minutes of GPU)
   per checkpoint epoch to catch when conditioning wakes up; the speed>50
   30-vs-100 brake/gas contrast is the go/no-go number.
4. Re-run the three-way patch_policy_eval on the full 200-batch subset,
   sequentially/single-writer (the parallel datamodule spins corrupted a
   pipefunc samples pickle in .rbyte_cache -- EOFError; rebuild that store
   first or keep using the batch-cache path).
5. Rollout-level compliance eval around limit transitions (transitions.md
   template) once the token actually moves actions.

Artifacts: override_probe_v0.md (probe + fidelity tables),
driving_regression_v0.md (ppe tables), aux_readout_v0.md (wandb readout),
probe_dump.npz / probe_dump_v1.npz + logs on aboutblank
diag_results/eval_v0/; scripts: src/rmind/scripts/eval_v0_dump.py,
diag_results/eval_v0/ppe_cached.py.
