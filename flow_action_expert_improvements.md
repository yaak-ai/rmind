# Flow Action Expert — Improvement Backlog

Ways to improve the flow-matching action generator (`FlowPolicyObjective` +
`FlowActionDecoder`), ranked by expected leverage. Grounded in the oracle
experiments and the real-data diagnostics below, not generic flow-matching
advice.

## Setup — genuine single-drive overfit, via policy finetune

Every real-data run below is a genuine **overfit** run: the flow policy head is
**finetuned** on the single drive `Niro096-HQ/2023-01-11--13-47-36` from the
frozen pretrained encoder artifact `model-e61ycirr:v9` (train == val == that one
drive). So every result here concerns the *head's* ability to memorize one
drive's condition→action map — not generalization. (An earlier worry that the
`yaak/overfit` datamodule had silently expanded to the full `yaak/train` corpus
was investigated and does **not** apply: these runs trained on the single
overfit drive as intended. The datamodule reference was hardened defensively so
it stays that way.)

The core puzzle this sets up: even asked to memorize one drive, **bo32 stalls at
0.042 ≫ oracle 0.017** (diagnostics below). The decoder collapses a clean
*synthetic* task to 0.017, but the real single-drive conditional map underfits,
and more epochs don't move the floor. Explaining that gap — on a true overfit —
is the whole hunt.

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
      instant replots, NaN-aware stats. Also emits the **horizon view**
      (`*_horizon.html`): per-h mean traces aligned to target frames +
      frame×horizon error heatmap + per-h stats — the planning-vs-reacting
      diagnostic. Knobs: `+fan.sampling_steps/sampling_method/ema/data`.
      `just fan inference=yaak/control_transformer/policy model.artifact=… \
        '+fan.out=….html' ['+fan.legacy_condition=true'] ['+fan.sampling_steps=N'] \
        ['+fan.data=….npz']`
      `legacy_condition` is REQUIRED for ckpts trained before image tokens —
      `POLICY_CONDITION_TOKENS` is baked into code, not the ckpt, so loading
      an old ckpt with new code silently changes its conditioning.
- [x] Env-gated NaN locator (`RMIND_NAN_DEBUG`) (debug-only; strip when done).
- [x] **RoPE in decoder slot self-attention** (`RotarySelfAttention`;
      `rope` flag threaded through the decoder, default-on for finetune via
      `flow_decoder_rope`). Constructor default off → old ckpts rebuild
      byte-identical.
- [x] **Within-chunk delta loss** (`chunk_delta_weight`) — see chunk section.
- [x] **PE diagnostics**: `pe_drift` val metric (relative drift of slot PE
      from a non-persistent reference; ~0 = slot identity unlearned) + the slot
      PE logged every val epoch as a **6×6 cosine-similarity matrix** image
      (`WandbImageParamLogger` with `torchmetrics.pairwise_cosine_similarity`).
- [x] **`EMAWeights` callback** kept in `finetune_flow` callbacks (decay 0.999,
      shadow swapped in for val, persisted in ckpt callback state; `+fan.ema`).
      Null on this data but rides along for free; re-test at multi-drive scale.
- [x] **`just field` — velocity-field / trajectory viz**
      (`flow_field_viz.py`): exact full-trajectory integration projected to
      (flow-time, steering@slot) bundles for straight→maneuver conditions,
      plus a (steering, gas) **quiver cross-section** over flow-time
      (`+field.quiver=true`). Finding (`n3bfdrrl:v399`): **single basin even at
      sharp maneuvers** (unimodal — expected on this data), steering-dominated
      convergence that sharpens late in `t` (why coarse Euler loses accuracy),
      model endpoint slightly *undershoots* GT (conservative mode-pull).
      Marks GT action + model endpoint; other dims frozen at bundle mean
      (cross-section caveat).
- [x] **Hardened `yaak/overfit` datamodule** to reference its own
      `/dataset/yaak/overfit` for both splits (defensive: prevents silent
      scope-creep into the full `yaak/train` corpus if that drive list grows).
      Verified: resolves to one drive (`Niro096-HQ/2023-01-11--13-47-36`).
- [x] **NaN guards**: drop non-finite rows in the flow loss (per-sample safe)
      with culprit-field attribution in the warning; auto-disable delta loss for
      `action_horizon < 2` (empty `diff` → NaN); `flow_mse` always logged
      (stable curve whether or not the delta term is active).
- [~] **Action standardization (linear, per-channel)** — implemented with
      non-persistent buffers (backward-compatible loads) → **tested → reverted**:
      flat noise ↓ but spikes ↑ (suspect 5). Replacement = nonlinear transform,
      not in tree.

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
- **Overfit/debug datamodules must reference their *own* dataset, never a
  shared one.** `yaak/overfit` borrowing `yaak/train` would silently scope-creep
  into the full corpus if that drive list grew — hardened defensively so it
  can't. Check `just train ... | grep "built dataframe"` drive variety if a run
  materializes longer than expected.
- **Never run two rbyte jobs against the same cache concurrently.** Launching
  the horizon=1 job while the full-ds run trained collided on the shared
  `${paths.rbyte.cache}/yaak/train/samples` dir (cache keyed by hostname +
  dataset, NOT by `sequence_length`) → `BrokenProcessPool` in the new job +
  `ConnectionResetError` killed the full-ds run. Mitigation: give each
  concurrent run an isolated `paths.rbyte.cache=...` dir.

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
2. **Training noise (no EMA)** — **RESOLVED: NULL** (`model-n3bfdrrl`, 400
   epochs, decay 0.999, raw-vs-EMA fan from the same ckpt at heun/32):
   EMA consistently ~2-3pt *worse* (flat hit@0.05 86.3% raw vs 84.6% EMA;
   flat L1 0.0280 vs 0.0297). No SGD orbit to average — the weights are in
   *steady slow descent*, so the trailing average lags. Consistent with the
   tiny-dataset regime (18 steps/epoch, near-smooth gradients). Re-test EMA
   at multi-drive scale where gradients are actually noisy; keep the
   implementation (`EMAWeights`, `finetune_ema.yaml`, `+fan.ema=true`).
   For overfit diagnostics, prefer plain `finetune` callbacks (EMA-swapped
   val metrics understate a still-descending model).
3. **Training duration** — **CONFIRMED as the hidden lever** by the same run:
   raw @400 epochs = flat 86%/0.028, spike 56%/0.096 vs ~100-epoch ckpts'
   73%/0.040, 42%/0.12. The "converged by epoch 40" plateau was an illusion:
   heun/8 integration noise (~0.015) fogged the val curves while the field
   kept improving underneath. Second half of the run improved at floor LR
   (`lr_training_steps: 3000` expired ~epoch 170) → a longer/stretched LR
   schedule is the cheapest open win. Check the wandb slope at epoch 400.
4. **Irreducible target noise** — real driver steering is noisy; if
   near-identical conditions carry different GT actions, no model closes the
   gap. → aliasing check: distribution of GT-action differences across
   condition-space near-duplicates. (Flat best-of-32 = 0.0022 argues this
   isn't the whole story; now bounded above by flat mean-draw 0.028.)
5. **Action scale / SNR** — **TESTED linear standardization → REVERTED.**
   The oracle standardizes its targets (zero-mean, σ≈0.5, matched to the
   `N(0,1)` flow noise); the real pipeline fed raw `*_normalized` actions
   straight in. Measured stats (one drive): gas μ+.15 σ.09 ∈[0,.47], brake
   μ.01 σ.06 ∈[0,.30], steering μ−.02 σ.20 ∈[−.94,.98]. So targets are
   5–20× smaller-std than the noise (poor SNR), nonzero-mean, and 4×
   unequal across channels — the decoder *internals* are fine (LayerNorm
   throughout; condition tokens are encoder-LN'd), the gap is purely this
   action/target boundary. Per-channel `(x−μ)/σ` was implemented (buffers,
   un-standardize at sampling, raw-space metrics) and **empirically: flat got
   less noisy (SNR fix worked) but spikes got WORSE.** Mechanism: a σ set by
   the near-zero bulk maps a 0.98 steering spike to ~5σ, an extreme tail the
   Gaussian-centric flow under-reaches → amplified undershoot on exactly the
   maneuvers that matter. Linear standardization *relocates* the problem (bulk
   precision bought with tail fidelity), so it was reverted.
   → **The fix is a NONLINEAR, invertible, tail-compressing/bulk-expanding
   transform**, not linear scaling:
   - **μ-law companding** (`MuLawEncoding` already exists in `components.norm`,
     used by the discrete `*_diff` tokenizers): spreads near-zero mass (flat
     resolution ↑) and compresses large magnitudes (spikes no longer 5σ).
   - **quantile / Gaussianize** (empirical CDF → `N(0,1)`): principled — makes
     the marginal exactly Gaussian so bulk *and* tails are well-conditioned.
   - cheap probe: scale by max-abs (spikes→±1) instead of σ to confirm the
     tail-reach mechanism before committing.
   Also compounds with the **gas/brake → signed longitudinal merge**
   (`long = gas − brake ∈ [−1,1]`): gas/brake are one-sided with a point-mass
   at 0 (brake is ~always 0) — pathological for a continuous Gaussian flow
   (can't represent a delta, leaks negative). They're physically mutually
   exclusive, so merging to one zero-centred symmetric channel is ~lossless
   and Gaussian-friendly; steering is already symmetric. Neither built yet.

## Chunk temporal structure — the chunk is blurred, anchored mid-horizon

Horizon diagnostic (`just fan` now also emits `*_horizon.html`: per-h traces
aligned to *target* frames + frame×horizon error heatmap + per-h stats;
calibrated against a synthetic copycat, whose signature is hit@0.1 collapsing
100%→9% and L1 growing ~6× across the chunk). Measured on `n3bfdrrl:v100`:

- **Per-h table is flat** (hit@0.1 ≈ 89-92% for all h, U-shaped with mid-chunk
  best) — NOT copycat. But not six sharp predictions either:
- **Lag analysis (mean prediction vs own target, at spikes): `lag = h − 3`,
  perfectly linear.** Every slot's content best matches the target of slot ~3;
  shift-corrected errors are identical (0.045-0.046) for all six slots →
  **the slots are interchangeable**: the model predicts ≈ one smoothed action
  anchored at the chunk middle and writes it into all slots.
- Within-chunk variation of draws is pure noise: the *mean* chunk's internal
  std at spikes is 0.011 vs GT 0.054 (~5× flatter), and the chunk-internal
  slope has **zero correlation with GT slope (ratio 0.00)**. The chunk is
  literally constant in expectation — one t+3 estimate copied into all slots.
- Visual signature (user-spotted): in the horizon view the h-traces are
  near-copies marching right by one step each.

Interpretation: the model **genuinely anticipates ~3 steps** (mid-chunk
anchor — a present-copycat would anchor at 0), but does **not resolve
within-chunk dynamics**. The flat per-h table is the blur's signature, not
chunk quality.

**Mechanistic root cause — found by weight inspection (`model-n3bfdrrl:v100`):**
the slot **position embeddings were never trained**. After 100 epochs their
row norms are 0.30-0.33 (= the `trunc_normal_ std=0.02` init scale, `0.02·√256
≈ 0.32`) and pairwise cosine ≈ 0 (random/orthogonal — unchanged from init).
Contrast: the `adaLN_modulation` linear (zero-init) trained to `|W| ≈
0.76-0.83` per layer and its scale/shift swing ~10-15% across flow-time `t`.
Same blocks, same optimizer, same loss — the network learned exactly what the
loss pays for. **Flow time has enormous gradient pressure** (the velocity
target's stats change drastically with `t` every batch); **slot identity has
~4×10⁻⁴ of the loss** (the within-chunk AC signal), so its only dedicated
channel (the additive PE) sat at init. The constant chunk is the **DC-optimal
fixed point**, and nothing pushed off it. Not a plumbing bug (adaLN proves the
conditioning path works) — an **incentive** problem.

Consequences:
- Receding-horizon execution of slot 1 applies an action ~2 steps (~0.7 s)
  **early** at maneuver onsets — a systematic anticipation bias, not noise.
- Temporal ensembling (ACT-style) does NOT fix this — averaging overlapping
  chunks re-centers but keeps the blur. The fix is training-side.

Fixes **implemented this session** (both attack the incentive/path; verify
together on a fresh overfit run carrying them):
- **Within-chunk delta loss** (`chunk_delta_weight`, default 10.0 in
  `finetune_overfit_flow`). Recovers the implied clean chunk via the
  interpolant identity `x̂₁ = x_t + (1−t)·v̂`, penalizes
  `MSE(diff(x̂₁), diff(x₁))` along the horizon. Differencing removes the DC
  component the model already nails → the term is **zero on flats, pure phase
  at maneuvers, and unsatisfiable by any constant chunk**. Verified: a
  constant chunk pays the full GT-delta-energy penalty; backprop puts
  norm-0.67 gradient on the previously-static PEs. Readouts: `pe_drift` lifts
  off 0, lag profile `h−3 → 0`, chunk slope ratio `0.00 → 1`.
- **RoPE in slot self-attention** (`rope`, default-on for finetune via
  `flow_decoder_rope`). Injects position at every attention layer (vs the
  additive PE added once at input). NB: RoPE alone *cannot* break slot
  symmetry — it rotates q/k (mixing weights) but identical slot *contents* mix
  to identical outputs (verified: permutation-equivariance gap 0.15 with RoPE
  vs 1e-7 without, but only because noise differs per slot). It complements
  the learned PE (the content-identity channel), doesn't replace it.
- Escalation if both stall (PE still static, lag staircase intact): **slot-
  conditioned adaLN** (widen modulation input from `t` to `(t, slot)` — the
  one channel proven to train), then per-slot output heads. Not yet built.
- Caveat: all measured on the overfit run *before* these fixes
  (`n3bfdrrl`, no delta loss / RoPE); re-run the horizon diagnostic on a fresh
  overfit carrying the fixes before trusting the fix verdicts.

**Observation: horizon=1 is notably WORSE than horizon=6** (despite h=6's
prediction being t+3-anchored). Two non-exclusive explanations, both plausible:
(a) *t+1 is intrinsically harder than t+3* — the immediate action is dominated
by high-frequency human reaction noise (aleatoric, scene-undetermined), while
the action ~3 steps out is the committed maneuver (scene-determined, smoother).
So the chunk anchoring at t+3 is partly adaptive, not purely a defect — a point
*for* chunking. (b) *More supervision* — 6 future targets give richer gradient
signal / regularization than 1. Confounds to clear before banking it: the
metric averages 6 steps for h=6 vs 1 for h=1 (compare h=6 slot-1 vs h=1 on the
same t+1 target); h=1 auto-disables the delta loss; and lower open-loop L1 on a
smoother target ≠ better closed-loop (h=6 slot-1 executes ~2-3 steps early).

## Strategic alternative: discrete action head (VQ-BeT-style)

The model already has a **trained action tokenizer**: per-channel
`UniformBinner` (gas/brake [0,1], steering [−1,1]) + `MuLawEncoding` for the
`*_diff` channels + a learned per-bin `Embedding` table, with categorical PT
objectives over the bins. (This is a *fixed* quantizer + learned embedding —
not VQ-BeT's *learned* residual-VQ codebook, but the same "discretize +
model categorically" family; ref VQ-BeT, arXiv 2403.03181.)

It does **not** plug into the continuous flow (paradigm mismatch; flowing in
the embedding space and snapping to codes is speculative). But it raises the
real strategic question: **flow vs. a discrete categorical head**, because the
discrete route natively solves the exact things flow struggles with here:
- **multimodality** — a softmax over bins is natively multimodal, in one
  forward pass; no N-sample + best-of-N, no sampling variance. (Turns the whole
  unanswered "is the data multimodal" question into "read the categorical.")
- **bounded + zero-spike** — a bin sits exactly at 0 (brake), bins are
  in-range (no out-of-bounds leakage), MuLaw gives extra resolution near 0.
- **~5× faster** (VQ-BeT headline) — no ODE integration; residual offset
  recovers sub-bin precision so quantization isn't a hard floor.
The VQ-BeT upgrade over the current *per-channel* binning is a **joint residual
VQ over the action chunk**, capturing cross-channel/cross-step correlation +
multimodality that independent per-channel bins miss. Worth standing up as a
competing baseline given the unresolved multimodality question and the
bounded-action pain. Near-term, even within flow, **borrow the MuLaw
companding** as the (nonlinear) action transform (see bottleneck suspect 5).

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
- **B. recipe items** — *action normalization*: linear standardization tested
  → reverted (flat↑ spike↓, suspect 5); the live form is the **nonlinear
  transform (μ-law/quantile) + gas-brake merge**, now a primary lever, not a
  secondary one. Per-channel loss weighting is subsumed by unit-σ normalization.
  Time-sampling tuning / parameterization ablations remain secondary.

## Top 3 to do next

1. **Re-run the overfit WITH the fixes** (delta loss + RoPE + PE diagnostics) —
   the overfit floor (bo32 ≈ 0.042) and the constant-chunk defect were both
   measured on the *pre-fix* run (`n3bfdrrl`). This is the first clean test of
   delta loss + RoPE + EMA together, and it answers the gating question: **do
   the fixes break the constant chunk and move the underfit floor toward oracle
   ~0.017?**
   - floor drops toward ~0.017 → the fixes work; the 0.042 stall was the
     untrained-PE / constant-chunk incentive problem, now solved.
   - stalls at ~0.042 with `pe_drift` still ~0 → escalate to slot-conditioned
     adaLN (the incentive fix didn't take); genuine capacity/conditioning limit.
   Watch `pe_drift` (off 0?), `chunk_delta_mse` (falling?), the cosine-PE image
   (banded structure?), and `sample_l1` vs oracle.
2. **Horizon-fan the resulting overfit ckpt** — does the lag profile flatten
   (`h−3 → 0`) and the velocity field stay unimodal / sharpen? Confirms whether
   delta loss + RoPE killed the constant chunk.
3. **Maneuver-L1 metric** (L1 on `|GT| > 0.5` frames, spike vs flat) as a
   first-class val metric — the scoreboard that actually favors getting
   maneuvers right, for the baseline comparison and all future runs. (Still
   not built.)

(Done since the last revision: EMA tested → null on the overfit, kept for
multi-drive; integrator sweep finalized → val on heun/32; delta loss + RoPE +
PE diagnostics + field viz implemented; overfit datamodule reference hardened
defensively (no contamination found). Also: linear
action standardization implemented → tested → **reverted** (flat↑ spike↓; the
nonlinear μ-law/quantile transform is the live form, suspect 5); horizon=1 vs 6
observation logged; VQ-BeT discrete-head alternative added as a strategic
option. NaN guards added: finite-row drop + culprit attribution, delta-loss
auto-disable for `action_horizon<2`, `flow_mse` always logged.)

## Let evaluation pick the lever

1. **best-of-N curve** — measured: small gap on the overfit (expected,
   unimodal). Re-measure on multi-drive data before concluding anything about
   flow's multimodality premise.
2. **Spike vs flat concentration** (`just fan`) — the new workhorse: spike
   hit@0.05 is the "covers → commits" number (best so far: 56% spike / 86%
   flat, `n3bfdrrl:v399` raw at heun/32 — up from 34%/53% where this hunt
   started).
3. **Horizon lag profile** (`*_horizon.html` + lag analysis) — the
   planning-vs-reacting readout; currently: anticipates mid-chunk, blurred
   within-chunk (`lag = h − 3`).
4. **Closed-loop (Tier 4)** — still the decisive metric.

Decision criterion (from `todo.md`): ship the action expert only if it (a)
matches/beats the horizon-matched Gaussian baseline on first-step steering
RMSE (E1.2), (b) shows a measurable best-of-N gain (E3.1) — *amended: or a
maneuver-L1 win, which the fan results suggest is flow's real edge* — and (c)
is non-inferior closed-loop (E4.1) at an affordable inference NFE (E2.2).
