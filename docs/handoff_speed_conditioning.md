# Handoff: speed-limit conditioning for the causal-decoder architecture

**Audience:** an agent working on the causal decoder, not on patch policy.
**Status:** the patch-policy attempt is finished and is a documented negative result. What
transfers is the problem framing, the data, the evaluation instruments, and a diagnosis of
*why* it failed. Do not re-run the patch-policy experiment.

Repo/branch: `yaak-ai/rmind` `feat/map-context` (pushed). Everything referenced below is
in that branch under `diag_results/` and `src/rmind/scripts/map_gt/`. Serving counterpart:
`yaak-ai/drivr` `feat/max-speed-input`.

---

## 1. The idea and the goals

**Goal.** Make the driving policy respect traffic rules: (1) speed limits, (2) traffic
lights and signs, (3) general environment awareness (German defaults — city 50, rural 100,
autobahn unlimited). Plus a near-term product need: on the private test ground (no legal
limit) we must be able to tell the model *"drive as if this were a 30 zone / a 10 zone /
Schrittgeschwindigkeit"* — as a model **input**, not as an external speed limiter.

**The framing that survived contact with reality.** The design space is three independent
axes, and most confusion comes from conflating them:

1. **Source** — where rule information comes from at inference: map (stale sometimes, and
   structurally cannot provide traffic-light *state*) vs vision (resolution-limited) vs both.
2. **Retention** — how a fact survives past the ~2 s observation window. A limit persists
   for minutes; context does not. This is a memory problem, not a window-size problem.
3. **Enforcement** — *why the model would obey*. This is the axis everyone forgets, and it
   is where the whole effort actually failed.

**The load-bearing insight, learned the expensive way.** Behaviour cloning provides almost
no gradient pressure on a rule input. The demonstrator's own speed already encodes the
limit, and where the limit changes, the sign is usually in the pixels. So
`I(action ; limit-token | vision, ego-speed) ≈ 0` on ~95 % of frames. **A conditioning
input that the objective does not need will be ignored regardless of how it is wired in.**
Everything in §3 follows from this.

## 2. What exists and transfers for free

**Ground truth — the expensive part, already done.**

- Per-frame map GT for **643 drives** (incl. all 5 val drives), ~34 M frames:
  `caches/map_gt/<Vehicle>/<drive>.parquet` — `frame_idx`, lat/lon, `max_speed_kmh`
  (NaN = unknown, −1 = explicitly unlimited), `road_class`, `env_class`
  (city|rural|motorway|private|unknown), `osm_way_id`, distance to next traffic light /
  stop sign. Builder: `src/rmind/scripts/map_gt/build_sidecar.py` (self-contained, no
  torch). Source is each drive's own `osm.mcap`; Overpass only enriches signal/stop nodes.
- Dataset wiring exists: the rbyte DAG joins the sidecars to produce
  `meta/MapContext/max_speed`. **Any cache built before that join silently yields
  all-UNKNOWN and trains fine with no error** — verify a built batch has non-NaN values for
  a sidecar-covered drive before spending GPU-hours.

**Two distributions you must design around** (`diag_results/map_gt/audit_report.md`):

- *Demonstrator compliance is already high*: over the limit only **5.6 %** of the time at
  +0 km/h tolerance, 2.3 % at +5 (city worst at 8.8 % while moving; rural 3.0 %). A
  compliance-conditioning token therefore partitions almost nothing — do not expect much
  from it.
- *Token frequencies are wildly unbalanced*: UNKNOWN 39 %, 30 km/h 13 %, 50 km/h 12.5 %,
  70/100/80/60 each 6–9 %, 120 at 2.6 %, and the classes the test ground needs —
  **WALK 0.15 %, 10 km/h 0.75 %, plus 130 at 0.22 % and UNLIMITED 0.19 %** — are
  vanishingly rare. Rare-class oversampling is not optional if the test-ground use case
  matters.

**Demonstrator reference behaviour** for limit transitions
(`diag_results/map_probe/transitions.md`, 4 548 debounced events): braking begins a median
**5–6.5 s before** a limit drop, 81–88 % are already compliant at the sign, median overshoot
0 km/h (small drops) to 2.2 (large); raises are exploited at t ≈ 0…+1 s. Worst case is a
drop *from* unlimited (median 29 km/h overshoot). These curves are what a conditioned
policy should be compared against.

**Vocabulary (keep it).** 13 semantic German classes, not numeric bins — the user was
explicit: *"I want to tell it: this is a 30 zone or a Schrittgeschwindigkeit zone, not
drive max 25."* `0 UNKNOWN, 1 UNLIMITED, 2 WALK (≤7 km/h), 3..12 = 10,20,30,50,60,70,80,
100,120,130`. Snap to nearest, exact ties **down** (40 → 30, conservative). Inputs are raw
float km/h; NaN = UNKNOWN, negative = UNLIMITED. Implementation:
`src/rmind/components/map_context.py`.

**Serving.** `drivr` `feat/max-speed-input` already carries it end to end: the engine's
optional `max_speed` binding is discovered at load (engines without it are fed nothing and
behave exactly as before), a `--max-speed <kmh|unlimited>` flag, a **zone dropdown in the
web UI** that applies from the next inference with no restart, and — added after the first
drive proved unattributable — `max_speed` + `max_speed_binding` logged **per prediction
row**. Keep the last one; without it a test drive cannot be analysed afterwards.

## 3. What was tried, and exactly how it failed

Two arms, warm-started from the dinov2_dinowm winner, 5 epochs each, on the full map-joined
data: **Arm M** (per-frame max-speed token + 30 % UNKNOWN dropout) = wandb `1n0ih44y`;
**Arm MV** (same + auxiliary head classifying the limit from the trunk) = `0nr1ydjm`.

Four independent measurements, all pointing the same way:

1. **Plumbing is provably fine.** Every override value produces pairwise-distinct logits,
   and `override=None` is **bitwise identical** to an all-NaN input. Not a wiring bug.
2. **Behaviourally inert, and epochs do not help.** The standing go/no-go probe — override
   30 vs 100 on frames above 50 km/h — is ~0 or *wrongly signed* at every checkpoint
   (epochs 1→5). Arm M: KL ≤ 8e-4, code flips ≤ 0.4 %. Arm MV is ~100× more responsive but
   only for WALK (KL 0.11 → 0.21 → 0.34 across epochs, 14 → 20 % code flips) and in the
   **wrong direction** (+0.017 gas under a walking-pace token — a regime association with
   parking/creep manoeuvres, not compliance).
3. **Guidance cannot rescue it.** A classifier-free-guidance sweep (the dropout training
   gives a clean cond/uncond pair) amplifies code flips monotonically with `w`, but the
   30-vs-100 contrast grows the *wrong* way and WALK's apparent brake rise past `w=5` is
   decode degeneracy (60–78 % code flips, gas-and-brake-both-high slots appearing). The
   median (sample, quantizer) slot needs **w ≈ 168** to flip at all.
   → `diag_results/eval_v0/cfg_sweep_mv_v1.md`
4. **The signal is numerically negligible** — measured on both finals over real val
   batches (`diag_results/salience/`): live max-speed token L2 **0.173** (M) / **0.239**
   (MV) against patch tokens at **15.98**, i.e. ~1 % of one patch token and ~0.005 % of the
   frame block's norm mass, with 256 patch tokens competing. For calibration, the speed
   token — which the policy demonstrably uses — sits at 0.79, **4.6× louder**. The 13 class
   rows *are* mutually separated (pairwise 0.20 / 0.32), so this is amplitude, not collapse.
   Arm MV's rows are 1.5× louder and 1.6× better separated than Arm M's — the aux gradient
   *is* the growth mechanism, just orders of magnitude short.

**No driving regression** from any of this (parent vs arms at n=6400: joint top-1 0.0616 vs
0.0602 — tied), and warm-start fidelity held (78.7 % code agreement under UNKNOWN). The
arms are good drivers that ignore the input.

**The on-car test was uninformative, separately.** No before/after pair (AI disengaged
during the one recoverable change), the value used was `10` — offline the most inert class,
distinct from WALK — and the drive resolved Δgas ≥ 0.025 against a predicted 0.0004, i.e.
**63× underpowered**, never exceeding 43 km/h. → `diag_results/oncar_max_speed/analysis.md`

## 4. Recommended method for the causal decoder

The reason to move this to the causal decoder is **portability**: conditioning applied at
the head is trunk-agnostic, whereas the patch-policy fusion site has no guaranteed
equivalent elsewhere. Design accordingly.

**Primary arm — FiLM conditioning at the head, on a frozen + cached trunk.**

- Embed the limit token (small, 16–32 dims is plenty for 13 classes) and use it to produce
  **scale/shift modulation** of the head's readout feature. Use FiLM, **not concat**:
  concat re-creates the amplitude failure of §3.4 (a small vector summed beside a large
  feature, requiring the model to learn amplification), while multiplicative modulation is
  loud by construction.
- **Condition both heads.** Decoded actions come from the code head *and* the offset head,
  and the offset head is teacher-forced on the chosen codes. Modulating only the code head
  moves codes without cleanly moving continuous outputs — our probes measure code flips, so
  this mistake is easy to miss.
- **Freeze the trunk and cache its readout.** One token per frame ≈ 6 × 512 floats ≈ 6 KB
  per sample → the whole train set fits in ~12 GB, turning an arm from ~6 h into minutes per
  epoch, which is what makes a real sweep (modulation form, gain init, dropout rate, loss
  weight) affordable. **Cache the readout, not patch tokens** — those are ~1.2 MB/sample
  (~2.4 TB). Note the `encoder_cache` machinery from the earlier head-FT study lives on the
  control-transformer lineage and is *not* on this branch.
- Keep the **UNKNOWN dropout** (~0.3). It keeps the missing-map path in-distribution and
  preserves a clean cond/uncond pair.

**Do not misapply the old verdict.** The head-side FT study concluded head changes are
"encoder-bounded". That was about asking an expressive head to extract *more* from an
existing feature; here we inject a **new input the trunk never carried**, so the bound does
not bind.

**Orthogonal and probably necessary — make the objective need the token.** Amplitude alone
may not suffice; nothing above changes the near-zero marginal information of §1. The three
levers, in the order I would try them:

1. **Informativeness weighting.** Upweight frames where the limit is *not* visually
   inferable (sign already passed) **and** ego speed ≠ limit — there the demonstrator's
   subsequent deceleration is predictable only from the token. Use the 4 548 transition
   windows as the seed set.
2. **Future-speed-profile auxiliary target** (mean speed over the next 5–10 s). The limit is
   genuinely predictive of that beyond the 2 s action horizon, which creates the gradient
   path limit → speed-plan → actions that plain BC lacks.
3. **Rare-class oversampling** for WALK / 10 / 20 — required for the test-ground use case
   regardless of anything else (§2).

Compliance conditioning (a compliant/non-compliant flag) is *not* recommended as a first
lever: at 94 % demonstrator compliance the partition is nearly degenerate.

**Fallback, only if head injection moves nothing:** early injection at the fusion site
(concatenate to the patch/goal stream). If you do this on an architecture that has such a
site, **normalize it** — the raw embedding sits at per-element RMS ≈ 0.008 against
LayerNormed patches at 1.0, a ~130× imbalance, worse than the 90× goal-swamping case that
motivated `fusion_norm`.

## 5. Evaluation

**The go/no-go probe (use this one, it is the standing instrument).** Override sweep over
{UNKNOWN, WALK, 10, 30, 50, 100, UNLIMITED} on identical real val batches, argmax decoding.
The headline number is **Δgas and Δbrake for override 30 vs override 100 on frames with ego
speed > 50 km/h**. Compliance-correct means *less gas / more brake under the lower limit*.
Report sign, not just magnitude — every failure so far had the wrong sign.
Tooling: `src/rmind/scripts/map_probe.py`, `src/rmind/scripts/eval_v0_dump.py` (has
`--micro-batch` so it fits on a busy shared GPU), analysis in `diag_results/eval_v0/`.

**Mandatory sanity anchors** (each has already caught something):

- `override=None` must be bitwise identical to an all-NaN input — proves plumbing.
- All override pairs pairwise distinct in logits — proves the input reaches the tensor.
- Salience: report the conditioning vector's norm against the features it modulates. A
  ratio of ~0.01 is what failure looks like.

**Driving-quality regression gate.** Compare against the parent on the *same* subset with
`patch_policy_eval` **argmax** columns (sampled decoding is entangled with entropy
calibration; never select on focal/perplexity — val loss rises while argmax recon improves,
a known calibration artifact). Conditioning that costs driving quality is not a win.

**Compliance metrics, once the probe shows a correct sign.** Speed error vs limit in ±10 s
windows around limit transitions, compared to the demonstrator curves in §2 (t_adapt,
overshoot, %-of-window over limit); %-time-over-limit per env class at +0/+3/+5 tolerance.
Steady-state compliance alone is confounded — the model can look compliant by copying its
own speed input, which is why the transition windows and the counterfactual override are
the real instruments.

**Aux-head metrics, if you use one.** Train accuracy is **copy-inflated** — the GT token is
visible in the input on ~70 % of frames. Log **dropped-frame-only accuracy** and per-class
recall, or the number is meaningless. (Ours read 0.98–1.00 and meant nothing.)

**Closed-loop and on-car.** CARLA (`rsim`) natively simulates limits and lights and dodges
the copy-the-speed-token confound entirely; add %-time-over-limit and behaviour at limit
changes to the existing distance/collision/rdev metrics. For an on-car test, the protocol
that yesterday's drive violated on every count: toggle **while AI is engaged**, A/B/A/B
every 20–30 s on one loop, ≥10 alternations, using the **extreme pair (WALK vs 100)** —
`10` vs `100` is offline-inert — and get into the speed regime the probe cares about. At
~34 plans/arm the resolution is ~0.025 gas; resolving a 0.017 effect needs ~10× more plans
per arm (≈10–15 min engaged per arm).

**Ship the shield regardless.** Given four negative results on learned compliance, the
deterministic guard (cap commanded speed at the map limit) should be treated as the
mechanism that *makes* the car compliant, with the learned conditioning responsible for
smoothness and anticipation. Log every shield trigger; the trigger rate is then a direct
KPI of how much the learned side is actually contributing.

## 6. Pointers

| what | where |
|---|---|
| Plan of record | Notion "Traffic rules & environment awareness" |
| Offline eval + verdicts | `diag_results/eval_v0/synthesis_v0.md` |
| CFG sweep (rescue ruled out) | `diag_results/eval_v0/cfg_sweep_mv_v1.md` |
| Salience measurement | `diag_results/salience/README.md` |
| On-car analysis + drive protocol | `diag_results/oncar_max_speed/analysis.md` |
| Demonstrator compliance / transitions | `diag_results/map_gt/audit_report.md`, `diag_results/map_probe/transitions.md` |
| Vocabulary + tokenizer | `src/rmind/components/map_context.py` |
| GT builder | `src/rmind/scripts/map_gt/build_sidecar.py` |
| Serving | drivr `feat/max-speed-input`; runbook section 10 in Notion "Drivr Runbook" |
| Final checkpoints | wandb `yaak/rmind` `model-1n0ih44y:v4` (Arm M), `model-0nr1ydjm:v4` (Arm MV); TRT engines + parity archives under `/nasa/max/models/patch_policy/` |
