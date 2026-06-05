# Flow Action Expert — Improvement Backlog

Ways to improve the flow-matching action generator (`FlowPolicyObjective` +
`FlowActionDecoder`), ranked by expected leverage. Grounded in the oracle
experiments and the real-data diagnostics below, not generic flow-matching
advice.

## What we already established (don't re-litigate these)

### Synthetic oracle (`flow_oracle.py`, overfit one batch, action-dim 3, heun/8, logit-normal)

| Config | flow_mse | sample_l1 floor |
|---|---|---|
| dim 32, 2L | 0.116 | 0.102 |
| dim 256, 2L (old default) | 0.010 | 0.021 |
| dim 256, **4L** | 0.007 | **0.017** ← best |
| dim 384, 2L | 0.011 | 0.023 (no gain) |
| dim 256, 8→32 sampling steps | — | 0.026 → 0.022 (negligible) |
| uniform vs logit-normal time | — | 0.030 vs 0.017 (logit-normal wins) |

- **The decoder/sampler is NOT the bottleneck** — it collapses a clean task to
  ~0.02. Capacity is tapped: depth 2→4 helps (~18%), width past 256 doesn't,
  NFE past 8 Heun steps doesn't (on the oracle). logit-normal > uniform.
- Open-loop L1 on a unimodal overfit structurally favors the Gaussian baseline
  (regression-to-the-mean). It can't show flow's value — that needs best-of-N /
  closed-loop.

### Real-data overfit diagnostics (single drive, 4L, frozen encoder)

**best-of-N val curves** (190 epochs; plateau by ~40, flat after):
bo1 ≈ 0.055 → bo32 ≈ 0.042. Small gap (~24%) = low sample diversity, no real
multimodality on an overfit (expected). bo32 floor 0.042 ≫ oracle 0.017 →
the conditional mapping is underfit; more epochs don't move it.

**Sample concentration** (`just fan`, 32 draws/frame, first-step steering,
legacy ckpt `model-f8zxhm29:v100`):

| | spike frames | flat frames |
|---|---|---|
| hit@0.05 | 35.8% | 54.8% |
| hit@0.1 | 60.6% | 85.7% |
| hit@0.2 | 80.6% | 98.6% |
| mean-draw L1 | 0.130 | 0.055 |
| best-of-32 L1 | 0.010 | 0.003 |

→ **The distribution's mass is on the GT maneuver** (~20/32 draws within 0.1
at spikes; the fan follows spikes to full ±1 magnitude). Best-of-N is not
fishing a lucky tail. Meanwhile the Gaussian baseline *undershoots* the same
spikes (mode-averaging — structural, unfixable by training). This is the case
for flow: **its failure (spread) is reducible; the baseline's (undershoot) is
not.** Remaining job: concentrate the mass ("covers → commits").

**Image-conditioning experiment — NULL RESULT (decisive).** Added 256 image
patch tokens to the condition (`model-tfuv76yx:v100`): every metric within
sampling noise of the legacy run (spike hit@0.05 33.6% vs 35.8%, flat L1 0.058
vs 0.055). Ablating image tokens from that ckpt at inference destroys it
(L1 ≈ 0.9, near-marginal) → the decoder **uses image tokens instead of the
summaries, not in addition** — it redistributes reliance across whatever
context it gets and lands at the **same floor** either way.
**Conditioning information is NOT the current bottleneck.**

### Architecture facts worth remembering (from `mask.py` attend rules)

- `foresight` attends only to observations + itself → action-independent at
  the encoder level (no leakage; safe to condition on).
- `obs_summary`/`obs_history` attend **only to foresight** (not raw obs!) —
  the policy's summary conditioning sees the world exclusively through the
  imagined-future bottleneck. (This motivated the image-token experiment;
  the null result says the decoder finds equivalent signal either way.)
- Waypoints = ordered sequence of ~10 future route points (`Linear(2→d)` per
  point), window slides one step per frame.

## Already done this session

- [x] `num_layers` 2 → 4 in `policy_finetune.yaml` (oracle-best depth).
- [x] Added `(Modality.CONTEXT, "waypoints")` then `(Modality.IMAGE,
      "cam_front_left")` to `POLICY_CONDITION_TOKENS`.
- [x] Val metrics: `sample_l1` = honest single-sample floor; `sample_l1_bo{k}`
      curve logged. Predict/plot path = single honest draw.
- [x] **Fixed period-`batch_size` prediction artifact**: `predict()` used
      `_validation_generator`, re-seeded to the same seed every batch →
      identical noise per batch → noise was a function of within-batch
      position → sawtooth with period = predict batch size (16). Fix: predict
      samples with the global RNG (`generator=None`); run-level
      reproducibility via `seed_everything`. (`compute_metrics` keeps the
      fixed-seed generator deliberately — scalar metric, no visible artifact.)
- [x] **`just fan` — sample-concentration diagnostic**
      (`src/rmind/scripts/flow_sample_fan.py`): N draws/frame in one batched
      call → interactive plotly fan (samples + GT + per-frame hit-rate, spike
      regions shaded), spike/flat hit rates and L1 stats, `.npz` cache for
      instant replots, NaN-aware stats.
      `just fan inference=yaak/control_transformer/policy model.artifact=… \
        '+fan.out=….html' ['+fan.legacy_condition=true'] ['+fan.sampling_steps=N'] \
        ['+fan.data=….npz']`
      `legacy_condition` is REQUIRED for ckpts trained before image tokens —
      `POLICY_CONDITION_TOKENS` is baked into code, not the ckpt, so loading
      an old ckpt with new code silently changes its conditioning.
- [x] Env-gated NaN locator (`RMIND_NAN_DEBUG`) (debug-only; strip when done).

## Known bugs / sharp edges

- **NaN-speed frames poison condition embeddings** → 1 frame on the overfit
  drive yields all-NaN samples. Now load-bearing (pollutes metrics/plots);
  cleanup promoted from "nice to have".
- `POLICY_CONDITION_TOKENS` is a module-level constant → ckpt behavior depends
  on code version at load time (see `legacy_condition` above). Lift to a
  config/hparam so the ckpt carries its own conditioning. (Was already on the
  list; now demonstrably a footgun.)
- Plot-reading gotchas burned before: dataloader worker interleaving can fake
  periodicity if plotted in row order (sort by `frame_idx`); flat-GT segments
  make single-sample spread look like noise; average L1 hides maneuver
  performance.

---

## Current bottleneck hunt (the only question that matters)

Both conditioning variants plateau at the same per-draw spread
(flat σ≈0.07 vs oracle ≈0.02). The floor is set by something both share.
Suspects, with the decision tree:

1. **Integration error** — **CONFIRMED by sweep** (image ckpt; ckpt hparams =
   **heun**/8, so the sweep was Heun 8/32/128 = NFE 16/64/256):
   flat mean-draw L1 0.058 → 0.040 → 0.039; flat hit@0.05 53% → 73% → 74%;
   spike hit@0.05 34% → 42% → 43%. **Saturates at heun/32 (64 NFE).** Spikes
   improved only ~8% (vs 31% flat) → remaining spike spread is field width,
   not integration. Oracle's "heun 8→32 negligible" does NOT transfer: the
   real learned field is curvier (harder conditional map, stiffer ODE) than
   the synthetic oracle's.

   **FINAL integrator sweep** (image ckpt; flat hit@0.05 / flat mean L1 /
   flat bo32; spike hit@0.05):

   | sampler | NFE | flat | bo32 | spike |
   |---|---|---|---|---|
   | heun/8 (old default) | 16 | 53% / .058 | .0031 | 34% |
   | **euler/16** | 16 | **89% / .025** | .0029 | **49%** |
   | heun/16 | 32 | 72% / .041 | .0027 | 42% |
   | midpoint/16 | 32 | 69% / .043 | .0027 | 40% |
   | euler/32 (×2, reproduced) | 32 | 81% / .0335 | .0024 | 45% |
   | heun/32 | 64 | 73% / .040 | .0024 | 42% |
   | euler/64 | 64 | 78% / .036 | .0025 | 44% |
   | heun/128 | 256 | 74% / .039 | .0024 | 43% |

   Reading (the numbers force it):
   - **Exact flow of the learned field = heun plateau**: 74% / 0.040 / bo32
     0.0024. That is the honest field quality. → **val metrics: heun/32**
     (set in `finetune_overfit_flow.yaml`).
   - **Coarser Euler is monotonically "better"** (64→32→16: 78→81→89%) while
     **bo32 worsens** (0.0024→0.0029): the signature of **variance
     reduction**, not accuracy. Coarse Euler under-resolves the conditional
     distribution and contracts draws toward its mean — trading the sample
     diversity flow exists for, to win unimodal single-draw metrics. The
     limit of this trend is a 1-step mean regressor (≈ the Gaussian
     baseline). euler/8/4 untested; the curve presumably keeps "rising".
   - **midpoint/16 ≈ heun/16** → the t=1-endpoint-query hypothesis is dead;
     the mechanism is step-coarseness smoothing/contraction, period.
   - **Deployment: deliberately open.** On unimodal overfit data euler/16
     dominates, but partly by collapsing the distribution — the property
     that would erase flow's multimodal edge on real data. Choose per
     checkpoint/data via fan + closed-loop, not as doctrine. Re-sweep after
     EMA (better field ⇒ less to gain from contraction).

   Gotchas: `sample()` already has euler/midpoint/heun — check the ckpt's
   `flow_sampling_method` hparam before reasoning about integrators. All
   training-time val curves before this change were heun/8 → their *levels*
   include ~0.015 integrator penalty (comparisons valid; absolute floors not).
2. **Training noise (no EMA)** — flow-matching loss is per-batch noisy (random
   t, random x₀ per visit); sampled field = one jittery SGD iterate, and
   integration compounds the wobble. bo-curves plateauing with wiggles at
   epoch 40 fit this. → add decoder-only EMA (encoder frozen, copy is tiny;
   `torch.optim.swa_utils.AveragedModel` or small Lightning callback that
   swaps weights for val/predict).
3. **Irreducible target noise** — real driver steering is noisy; if
   near-identical conditions carry different GT actions, no model closes the
   gap. → aliasing check: distribution of GT-action differences across
   condition-space near-duplicates. (Flat best-of-32 = 0.003 argues this
   isn't the whole story.)

## Demoted / re-ranked

- **A. Conditioning enrichment** (speed, history window, more tokens) —
  demoted from "biggest lever" to "not the bottleneck" by the image null
  result. Don't invest until the floor moves.
- **CFG** — premise weakened: there's no missing condition-adherence to
  amplify when two condition sets give identical floors. Revisit only on real
  multi-drive data (where conditioning may genuinely bind).
- **Unfreeze encoder** — same demotion, same reason.
- **Capacity / NFE-for-accuracy** — tapped (oracle). Distillation for latency
  (reflow/consistency) remains relevant for deployment, unchanged.
- **B. recipe items** (per-channel loss weighting, action normalization,
  time-sampling tuning, parameterization ablations) — still plausible
  secondary wins, behind the bottleneck hunt.

## Top 3 to do next

1. **EMA run** — `EMAWeights` callback implemented (swaps in for val, shadow
   persisted in ckpt callback state; `finetune_overfit_flow` now uses
   `/trainer/callbacks: finetune_ema`, val at heun/32). Launch the overfit,
   read val `sample_l1` + `just fan '+fan.ema=true'` vs the 74%/0.040 field
   floor. This is the main lever for the remaining gap vs oracle 0.017.
2. **Post-EMA integrator re-sweep** (cheap, `just fan`) — tests the
   prediction that a smoother field lifts the heun plateau and shrinks the
   coarse-Euler contraction bonus.
3. **Maneuver-L1 metric** (L1 on `|GT| > 0.5` frames, spike vs flat) as a
   first-class val metric — the scoreboard that actually favors getting
   maneuvers right, for the baseline comparison and all future runs.

## Let evaluation pick the lever

1. **best-of-N curve** — measured: small gap on the overfit (expected,
   unimodal). Re-measure on multi-drive data before concluding anything about
   flow's multimodality premise.
2. **Spike vs flat concentration** (`just fan`) — the new workhorse: spike
   hit@0.05 is the "covers → commits" number (currently ~36% vs flat ~55%).
3. **Closed-loop (Tier 4)** — still the decisive metric.

Decision criterion (from `todo.md`): ship the action expert only if it (a)
matches/beats the horizon-matched Gaussian baseline on first-step steering
RMSE (E1.2), (b) shows a measurable best-of-N gain (E3.1) — *amended: or a
maneuver-L1 win, which the fan results suggest is flow's real edge* — and (c)
is non-inferior closed-loop (E4.1) at an affordable inference NFE (E2.2).
