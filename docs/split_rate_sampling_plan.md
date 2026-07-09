# Throttle rebalance: split-rate action sampling + rare-case phase-offset oversampling — FINAL PLAN

Repos: `/home/max/Code/throttle/rmind` (branch `feat/throttle-rebalance`, base PR #248) and `/home/max/Code/throttle/rbyte` (reference only). **rbyte changes: NONE** — pinned PyPI rbyte 0.38.1 already ships per-column `gather_every: Mapping[str,int]` (verified in `.venv/.../rbyte/io/dataframe/groupby.py:35,54-58`), heterogeneous fixed-Array column widths through `to_torch`/TensorDict, and batch-level unique-frame stream-read dedup (#101). Everything lands in rmind: one ~5-line code change + templates/configs + the existing stats harness.

______________________________________________________________________

## 1. Architecture decision

### 1.1 Rate scheme — frame-snapped 6 Hz actions (verified constraint chain)

Actions are shipped as an additional high-frequency (HF) column set gathered at `action_sample_step: 5` frames = **6 Hz**, alongside images at `episode_step: 10` = 3 Hz. The FT `joint_actions` chunk becomes **12 samples @ 6 Hz over 2 s** (vs 6 @ 3 Hz today), so a 0.5–1.5 s gas ramp = 3–9 supervised samples instead of 1–4.

**Why not true 8 Hz (30 fps / 8 Hz = 3.75 frames, the non-integer problem):**

- `gather_every` strides are integers; a chunk for image step *t* must start exactly on image frame `10t`, so `action_sample_step` must **divide** `episode_step=10`. Valid rates: stride 5 = 6 Hz (default), stride 2 = 15 Hz, stride 1 = 30 Hz — all one config knob. Stride 4 (7.5 Hz) puts image steps at fractional action indices (0, 2.5, 5…); rejected.
- True timestamp-based 8 Hz would require: new rbyte resampler/interval-aggregator components + a release + pin bump, `t_rel` tensors through the model, the unwired RoPE `input_pos` path (float positions aren't even supported by the integer cache — new PE class), a changed ONNX input contract, and reproduction of ±16 ms frame-snap jitter at inference. That is maximal exposure to the Notion §3 train/inference-mismatch warning for marginal rate benefit over 6 Hz. **Rejected**; 15 Hz remains reachable later by config if 6 Hz proves insufficient.
- Bonus: 6 Hz divides the 3 Hz serving tick exactly (2 sub-actions per rsim/car tick) — no second non-integer problem at execution time.

**Backward-compat mechanism (the load-bearing trick, verified on DuckDB 1.5.4):** the canonical 3 Hz action columns keep their exact names and are **derived from the HF list by a stepped slice** `hf[1 : (clip_length-1)*rate_ratio + 1 : rate_ratio]` (1-based inclusive; for ratio=2, clip_length=11: `hf[1:21:2]` → HF indices 0,2,…,20 = frames 0,10,…,100 — **bit-identical values** to today's stride-10 gather). All existing SQL filters (Stage-6 clamp, conflict-drop, low-signal, at-rest predicates) evaluate on these canonical views → filter decisions unchanged; PT, the old FT recipe, `*_diff` semantics (1/3 s), and `RuleBasedCluster`/`PredictMetricsCallback` field reads are all untouched with **zero recalibration**. The PT/FT shared-dataset-build property is preserved.

### 1.2 Phase-offset oversampling — exact specification

Enumeration: `every: "${phase_stride}i"` with `phase_stride: 2` (default 10 = off) → windows at 5 phases per canonical 10-frame stride. **Single pass** (5× rows exist only at the transient pre-cast stage); the two-pass rare-anchor variant (would need rbyte's `offset` passthrough) is the documented fallback if the M2 dry run says build time is painful.

Definitions (verified: polars `group_by_dynamic(start_by="datapoint")` anchors at the drive's **first datapoint**, so phase must be drive-relative — a raw `frame_idx % 10` would be wrong). `samples_cast` sees one drive (mapspec `samples[i] -> samples_cast[i]`):

```sql
w0     := list_first("meta/ImageMetadata.cam_front_left/frame_idx")
drive0 := (SELECT min(list_first("...frame_idx")) FROM samples)   -- scalar subquery, per-drive grid origin
phase  := (w0 - drive0) % ${episode_stride}                       -- ∈ {0,2,4,6,8}
anchor := (w0 - drive0) // ${episode_stride}                      -- phase-invariant window identity
```

`phase = 0` reproduces today's `every=10i` window set exactly (same anchor origin). Stratified keep rule — ONE `CASE`, evaluated in order (R wins over H by design: a window that is both launch-ish and brake-hold-ish is kept, never capped):

```sql
CASE
  -- Stratum R (rare / launch-continuation): decision frame in the launch band,
  -- gas pressed within the decision-step 2 s chunk (HF list, 1-based 11..22)
  WHEN speed_c[${episode_length}] < ${rare_speed_max}                       -- km/h (confirmed via harness docstring)
       AND list_max(gas_hf[(${episode_length}-1)*${rate_ratio}+1 : ${hf_len}]) > ${rare_gas_min}
    THEN phase IN ${rare_phases_sql}                                        -- default {0,4,8}: 3x, 133 ms apart
  -- Stratum H (static brake-hold): existing cap predicate verbatim, on canonical views
  WHEN speed_c[${episode_length}] < 1.0 AND brake_c[${episode_length}] > 0.1
       AND list_max(gas_c) <= 0.02 AND list_max(speed_c) < 2.0
    THEN phase = 0 AND anchor % 100 < ${atrest_keep_pct}
  -- Stratum C (common)
  ELSE phase = 0
END
```

Properties: (1) deterministic, no `random()`, reproducible per drive; (2) the cap selector moves from `(list_last(frame_idx)//10)%100` to `anchor % 100` — phase-invariant (all siblings share one cap decision) and enumeration-density-invariant; this fixes the correlated-runs-of-5 failure the current selector would have at `every=2i`. Survivor set changes once vs today (distribution-identical, 1% granularity preserved) — documented, validated in M2; (3) commons and brake-holds stay phase-0 → common set == today's build; (4) `rare_phases` is the explicit diversity-vs-count knob (Notion §3a): default `[0,4,8]` (3×), widen to all 5 or narrow to `[0]` per the M2 diversity audit; (5) with rares ≈ 2% of windows, final rows grow ~ +4-6%.

### 1.3 Window geometry (clip_period unchanged)

`clip_period` stays **110i**: HF gather @5 over \[0,110) → frames 0,5,…,105 = `hf_len = (episode_length-1)*rate_ratio + action_horizon_hf = 22` elements; images @10 → 0,…,100 = 11 (today's `clip_length`). Chunk at image step *t* = `hf[2t : 2t+12]`, starting exactly on frame `10t`; last chunk (t=5) spans frames 50…105. The image-side contiguity check is unchanged. One known edge effect: the `len(gas_hf) = 22` completeness check requires data to `w0+105` vs today's `w0+100` → at most ~1 fewer tail window per drive (accepted; counted in M2 regression).

### 1.4 New Hydra globals (`config/experiment/yaak/control_transformer/finetune_throttle_hf.yaml`, `# @package _global_`)

```yaml
episode_length: 6
episode_step: 10                 # images 3 Hz — unchanged
episode_stride: 10               # canonical grid — unchanged
clip_horizon: 6
clip_length: "${eval:'${episode_length} + ${clip_horizon} - 1'}"        # 11
clip_period: "${eval:'${clip_length} * ${episode_step}'}i"              # 110i — unchanged
action_sample_step: 5            # 6 Hz; MUST divide episode_step (2 -> 15 Hz); 10 = legacy
rate_ratio: "${eval:'${episode_step} // ${action_sample_step}'}"        # 2
action_horizon_hf: 12            # 2 s @ 6 Hz
hf_len: "${eval:'(${episode_length} - 1) * ${rate_ratio} + ${action_horizon_hf}'}"   # 22
action_horizon: ${action_horizon_hf}
phase_stride: 2                  # window enumeration every (frames); 10 = off (today)
rare_phases: [0, 4, 8]           # diversity-vs-count knob
rare_speed_max: 10.0             # km/h — launch-continuation band
rare_gas_min: 0.05               # disjoint from brake-hold's gas<=0.02 by construction
atrest_keep_pct: 5
action_lowpass: false            # optional anti-alias, see M2
```

______________________________________________________________________

## 2. Milestones

### M1 — `ChunkFields.step` (0.5 d, zero behavior change)

**Files:** `src/rmind/components/nn.py` (+ tests under `tests/`).

```python
def __init__(self, *, episode_length, action_horizon, unfold_paths, dim=1, step: int = 1): ...
# forward, unfold branch (currently nn.py:241, hardcoded step=1 — verified):
return value.unfold(self.dim, self.action_horizon, self.step).narrow(self.dim, 0, self.episode_length)
```

Add validation: `(input_len - action_horizon) // step + 1 >= episode_length`. Geometry with step=2 on a 22-long axis: `unfold(1,12,2)` → 6 windows → `(B,6,12)`; PT semantics with step=1 unchanged.

**Tests:** (a) `step=1` bit-exact regression vs current output on random tensors; (b) synthetic 22-long HF axis: `chunk[..., 0]` at step=2 equals the stride-2 subsample (the canonical 3 Hz value at each image step); (c) length-validation raises.
**Go/no-go:** tests green; no config references changed yet — mergeable independently.

### M2 — Dataset template refactor + harness + dry runs (2 d)

**Files:** `config/_templates/dataset/yaak/{train,val,train_debug}.yaml` (ytt templates are the source of truth; rendered files regenerate via `just generate-config`), `src/rmind/scripts/atrest_window_stats.py`.

**2a. Group-by node (train.yaml:1000-1011):** `every: "${phase_stride}i"`; `gather_every` becomes a per-column Mapping — **every column must be mapped** (unmapped columns keep full 30 fps and bloat pre-cast rows; waypoints especially):

```yaml
gather_every:
  meta/ImageMetadata.cam_front_left/frame_idx: ${episode_step}
  meta/ImageMetadata.cam_front_left/time_stamp: ${episode_step}
  meta/VehicleMotion/speed: ${episode_step}
  meta/Gnss/xy: ${episode_step}
  waypoints/xy: ${episode_step}
  waypoints/xy_normalized: ${episode_step}
  meta/VehicleMotion/gas_pedal_normalized: ${action_sample_step}
  meta/VehicleMotion/brake_pedal_normalized: ${action_sample_step}
  meta/VehicleMotion/steering_angle_normalized: ${action_sample_step}
  meta/VehicleState/turn_signal: ${action_sample_step}
```

**2b. samples_cast (train.yaml:1013-1079):** CTE computes `phase`/`anchor` (§1.2) and canonical views `gas_c/brake_c/steer_c/turn_c/`(speed already 3 Hz) via the stepped slice (§1.1). SELECT emits:

```sql
-- observed-only image-side (decode + shm savings):
"...frame_idx"[1:${episode_length}]::INT32[${episode_length}],          -- 6-wide: JPEG decode 11->6
"...time_stamp"[1:${episode_length}]::TIMESTAMP[${episode_length}],
"meta/Gnss/xy"[1:${episode_length}]::FLOAT[2][${episode_length}],
"waypoints/xy(_normalized)"[1:${episode_length}]::FLOAT[2][10][${episode_length}],
-- canonical (names + widths unchanged -> PT, old FT, callbacks untouched):
"meta/VehicleMotion/speed"[1:${clip_length}]::FLOAT[${clip_length}],
gas_c::FLOAT[${clip_length}]  AS "meta/VehicleMotion/gas_pedal_normalized",   -- + brake, steering; turn_c::BIGINT
-- NEW HF columns (FT consumes these):
"meta/VehicleMotion/gas_pedal_normalized"[1:${hf_len}]::FLOAT[${hf_len}] AS ".../gas_pedal_normalized_hf",
-- + brake_hf, steering_hf FLOAT[22]; turn_signal_hf BIGINT[22]
```

WHERE: existing contiguity pair on the **pre-cast** frame_idx list (`len = ${clip_length}` AND span `= (${clip_length}-1)*${episode_step}` — unchanged, per-phase-valid); NEW `len(".../gas_pedal_normalized") >= ${hf_len}` (×4, catches ragged tail windows — verified polars emits short lists at drive tails); all existing Stage-6/conflict/low-signal/at-rest predicates rewritten to reference canonical views (bit-exact decisions); the stratified CASE from §1.2 replacing the current cap clause (train.yaml:1071-1079).

Notes: keep canonical `speed` at 11-wide (44 B) so `RuleBasedCluster` `last`/`last_diff` semantics ([:, -1] = horizon end) are untouched; waypoints shrink 11→6 — `WandbWaypointsLogger`'s `[..., -1]` then reads the last *observed* step (more correct for logging; note in PR). Audit PT `raw.yaml` Remapper paths for any consumer of horizon-step image/Gnss/waypoint entries before merging (none expected — ChunkFields narrows to 6).

**2c. Optional anti-alias (`action_lowpass: true`, ytt-gated):** forward mean over each `action_sample_step`-frame interval applied to the 30 fps `filtered` df **before** group-by (port of rsim's replay-validated `action_lowpass`, RMS speed err 3.79→0.73 km/h). If on, the canonical views derive from the filtered HF list so `chunk[...,0]` == canonical token stays invariant. Default off; decide via M3 tokenizer recon + M5 offline eval.

**2d. Cache keys (mandatory — current key silently serves stale builds):** train run_folder (train.yaml:709) → `.../yaak/train/clip${clip_length}/as${action_sample_step}/ah${action_horizon_hf}/ph${phase_stride}/rp${rare_phases_key}/lp${action_lowpass}/cap${atrest_keep_pct}/samples`; same-style keys for `val.yaml:59` and `train_debug.yaml:57` (currently fixed folders → guaranteed stale-serve). Delete orphaned `clip11/cap*` caches after validation.

**2e. val / train_debug / predict:** `val.yaml` gets the gather mapping + dual-column cast (model needs 22-wide HF actions; loss comparability) but **keeps `every: ${episode_stride}i`, no CASE clause, no cap** — natural distribution. `train_debug.yaml` fully mirrors train. `predict.yaml` + `inference/onnx|tensorrt.yaml` literals (6×10i @3 Hz observed-only) **unchanged** — they mirror deployment.

**2f. Harness extension (`atrest_window_stats.py` — exists, structure confirmed):** parameterize the interpolation context (add `phase_stride, rare_phases, action_sample_step, action_horizon_hf, rare_speed_max, rare_gas_min` as CLI args); recompute `phase/anchor/stratum` in polars exactly mirroring the SQL; report per-stratum × phase window histograms, anchor multiplicity, achieved at-rest keep %, rare-window uplift (target ≈ len(rare_phases)×), per-phase decision-frame gas/speed histograms in the 0–10 km/h band (proves phases land targets between canonical grid points), and a **sibling-diversity audit** (per sibling group: mean |Δ| decision-frame gas/speed, distinct launch events).

**Validation / go-no-go:**

1. **Bit-exact regression:** build 30 drives at `phase_stride=10, rare_phases=[0]` and diff canonical columns + window keys `(drive, w0)` vs the current-schema build. Must match except (a) documented cap-survivor re-selection (anchor-based; rate must stay ≈ atrest_keep_pct) and (b) ≤1 HF-completeness tail window per drive. Launch windows never dropped (reuse `_launch_keys`).
1. **1-drive `every=2i` timing** pins the rebuild estimate (extrapolated 6–12 s/drive; abort to the two-pass fallback if ≫).
1. **Diversity gate:** sibling audit must show distinct decision-frame states across phases; if near-duplicate, narrow `rare_phases` or turn on `action_lowpass` before proceeding.

### M3 — RVQ action tokenizer retrain @ 6 Hz (1 d engineering + compute)

**Files:** `config/experiment/yaak/action_tokenizer/pretrain_6hz.yaml` (new: `action_clip: 12, action_step: 5, action_stride: 10` ⇒ `action_dim` auto-scales 24→48, `clip_period` 60i, run_folder auto-rekeys `t12d5s10`), `config/_templates/dataset/yaak/action_train.yaml` (+`action_val.yaml`).

- **Parity fix (do it now):** add the Stage-6 brake clamp + conflict-drop filters to `action_train.yaml` — the tokenizer currently trains on unclamped data, inconsistent with the policy dataset.
- Keep G=4 / C=16 initially; capacity screen `{(4,16),(6,16),(4,32)}` only if the gate fails.
- Publish new W&B artifact; **consolidate the triple-hardcoded refs** (`model/yaak/control_transformer/raw.yaml:204`, `raw.yaml:382`, `policy_finetune.yaml:49`) onto `${action_tokenizer_artifact}` while touching them.

**Go/no-go (before any FT spend):** held-out **per-channel reconstruction L1**, especially gas in the 0–10 km/h launch band, ≤ the 3 Hz tokenizer's. If it degrades: escalate capacity; if still failing: reconsider H=12 (this gates the whole design).

### M4 — FT wiring + smoke (1 d)

**Files:** `config/model/yaak/control_transformer/policy_finetune.yaml` (hparams_jq), `src/rmind/components/objectives/joint_policy.py`, `config/trainer/callbacks/finetune_unfrozen.yaml`, optionally `src/rmind/components/episode.py`.

1. **hparams_jq:** (a) repoint the 4 action paths in the Remapper to the `*_hf` batch columns (`*_diff` paths stay canonical — 1/3 s diff semantics preserved); (b) extend the existing ChunkFields patch: `.action_horizon = ${action_horizon_hf}` **and** `.step = ${rate_ratio}`; (c) `SliceFields` unchanged — `chunk[..., 0:1]` on the HF chunk = the action at the image frame = identical value to today's 3 Hz token → per-timestep tokens, UniformBinner tokenizers, S-layout (530), masks, RoPE, EpisodeBuilder all untouched. StackFields → `joint_actions (B,6,12,4)`.
1. **Heads:** `offset_head` out `1536 → ${eval:'4*16*4*${action_horizon_hf}'}` = 3072 (G·C·action_space·H); `code_head` 64 unchanged. `JointPolicyObjective`/`ActionTokenizer` code is H-invariant.
1. **Fix the found gap:** implement `score_l1/score_signed_error/prediction_std` in `JointPolicyObjective.predict` — at HEAD the `finetune_unfrozen.yaml` cluster metrics (at-rest/launch slices) silently log nothing; the M5 eval gates need them. (First confirm what run eyn6xhtl actually logged.)
1. Callbacks need **zero recalibration** (they read canonical columns). `LogitBiasSetter` remains a no-op in this FT.
1. Optional hardening (cheap, from Design B): shape-check `(t,s)` on the EpisodeBuilder attention-mask cache (episode.py:105-110,172-196) — latent stale-mask bug; not triggered by this design (T=6, S=530 unchanged) but worth closing.

**Go/no-go (train_debug smoke):** joint_actions `(B,6,12,4)`; offset_head 3072; per-timestep action-token values equal the canonical columns on a fixture batch; cluster metrics actually log; loss curves sane for ~100 steps.

### M5 — Full build + FT run (1.5 d attended + GPU time)

Full 655-drive build at `phase_stride=2` (~10–25 min with the process pool per M2 timing; transient samples stage ~8.5 GB). First-run checks: shm TensorDict ≈ 3.3 GB, decode throughput (expect ~0.57× decodes), LogitBiasSetter RSS ~GB-scale. Then FT: rerun the eyn6xhtl recipe (cap5 + unfrozen) on `finetune_throttle_hf`.

**Go/no-go:** val NLL/L1 not degraded overall; **launch-band predict-L1** (5–10 km/h, gas>0 cluster) improved vs eyn6xhtl; for 3 Hz-baseline comparability, decimate the 12@6 Hz predicted chunk (every 2nd) to point-L1 @3 Hz.

### M6 — Offline eval + CARLA closed-loop (1 d attended; rsim adapter is a separate workstream)

CARLA 8-spawn creep benchmark via rsim (memory: point-L1 alone doesn't transfer — closed-loop is the arbiter). Success = ≥8/8 exceed creep AND improved time-to-10 km/h / gas-ramp continuity vs eyn6xhtl. Document the run in rsim docs per convention.

**Total effort ≈ 7 engineering days** (M1 0.5, M2 2, M3 1, M4 1, M5 1.5, M6 1), excluding GPU time (tokenizer retrain, FT, CARLA).

______________________________________________________________________

## 3. Efficiency budget (numbers from the recon digests, verified where possible)

| Item                       | Today                          | After                                                                          | Delta                                            |
| -------------------------- | ------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| shm bytes/row              | 2,244 B (waypoints 78%)        | ≈ 1,780 B (+440 HF cols; −800 waypoints 11→6; −100 frame_idx/ts/Gnss 11→6)     | **−21%**                                         |
| Rows (cap5)                | ~1.58 M                        | ~1.65 M (rares ≈2% × 3 phases → +4-6%)                                         | +4-6%                                            |
| shm TensorDict             | ~4.0 GB (cap100)               | ~3.3 GB                                                                        | **smaller despite densification + oversampling** |
| JPEG decode/sample         | 11 (5 wasted post-ChunkFields) | 6 (frame_idx cast 6-wide; get_batch decodes len(frame_idx))                    | **−45%**; net ≈0.57× with row growth             |
| Rebuild (655 drives)       | ~35 min (3.2 s/drive measured) | est. 6–12 s/drive at every=2i → ~10–25 min pooled (≤2 h sequential worst case) | pinned by M2 1-drive run                         |
| Transient pre-cast storage | ~1.7 GB-scale                  | ~8.5 GB (5× windows, no pixels)                                                | acceptable, transient                            |
| Model compute              | —                              | unchanged (T=6, S=530); offset_head +~1.6 M params                             | negligible                                       |

Overlapping phase-sibling windows share JPEG files on disk (rows carry only frame_idx) and share decodes when co-batched (rbyte 0.38.1 #101 dedup) — bonus, not load-bearing. A decoded-frame LRU in `PathTensorSource` is optional future rbyte work only.

## 4. Serving implications (rsim / car — stated, out of implementation scope)

- **Input contract unchanged:** observation windows stay 6 × 3 Hz; executed-action history stays `(1,6,1)` with 3 Hz values (SliceFields index-0 semantics preserved). `predict.yaml` + `inference/onnx|tensorrt.yaml` untouched on the input side.
- **Output chunk changes:** `(B,6,4)` @3 Hz → `(B,12,4)` @6 Hz ⇒ ONNX re-export required (offset_head resize); export arg shapes for joint_actions in `config/export/yaak/control_transformer/finetuned.yaml`.
- rsim/car tick at dt=1/3 s executing `chunk[0]` + re-plan per tick keeps working (chunk[0] ≈ today's action). Proper use of the added resolution = 2 sub-actions per tick with zero-order-hold at the CAN layer — 6 Hz was chosen so this divides evenly. rsim `agents/rmind.py` `_build_batch` frame-padding and the chunk-decode path need the new shape (separate rsim workstream).
- Phase-offset training windows need **no** inference change (window-relative layout identical).

## 5. Risks and mitigations

1. **RVQ under-reconstruction of launch-band gas ramps** (action_dim 24→48 at constant 4×16 capacity) — the FT offset-L1 can mask degraded targets. Mitigation: hard M3 gate on per-channel launch-band reconstruction L1 before any FT spend; capacity screen as fallback.
1. **Phase-sibling near-duplication** (Notion §3a): siblings share ~94% content; enumeration can multiply the same few launch events instead of adding diversity. Mitigation: anchor-keyed cap, `rare_phases=[0,4,8]` default with the M2 diversity audit as go/no-go, `action_lowpass` anti-alias knob.
1. **Cache-key mistakes → stale builds:** run_folder currently keys only clip/cap. Mitigation: key extension lands in M2 *before* any grid change; M2 regression gate catches collisions; delete orphaned caches.
1. **Train→serve chunk-rate semantics:** model supervised on 6 Hz chunks, car executes at 3 Hz — chunk[0]-hold is safe but discards resolution; genuine closed-loop gains partly depend on the out-of-scope rsim/car execution change. CARLA (M6) is the arbiter either way.
1. **Cap-survivor re-selection:** moving the selector to `anchor%100` changes which at-rest windows survive (once, distribution-identical). Validated in M2; not silently mixed with other changes.
1. **Build-time estimate is extrapolated** (6–12 s/drive at 2i): pinned by the M2 1-drive run; two-pass rare-anchor group-by (1.1–1.3×, needs rbyte `offset` passthrough) is the fallback.
1. **Hidden horizon-step consumers** of the shrunk image-side columns (waypoints/Gnss/time_stamp 11→6): PT Remapper audit in M2; WandbWaypointsLogger `[...,-1]` semantics shift noted.

## 6. Open questions for Max

1. **Rate default:** 6 Hz/H=12 (this plan) vs 15 Hz/H=30 (stride 2; action_dim 120, ~2.5× tokenizer DoF, bigger offset head)? 6 Hz is the recommended start; 15 Hz is one knob away after the pipeline lands.
1. **`rare_phases` default:** `[0,4,8]` (3×, 133 ms apart, conservative) vs all 5 (`[0,2,4,6,8]`)? The M2 diversity audit will inform, but the prior matters.
1. **`action_lowpass`** on by default for the HF columns (rsim replay evidence is strong) or off for raw-signal fidelity? It changes both chunk targets and (via the derivation invariant) the canonical token values.
1. **Rare predicate params:** `rare_speed_max=10 km/h`, `rare_gas_min=0.05` — confirm the launch-continuation band definition against the Notion plan before freezing.
1. **eyn6xhtl cluster metrics:** did it actually log `predict/atrest_*` scores? (Determines whether the `JointPolicyObjective.predict` score-keys fix in M4 is a regression fix or net-new.)
1. **PT migration:** PT stays on canonical 3 Hz columns indefinitely under this design — is an eventual PT move to HF chunks (joint inverse dynamics at 6 Hz) wanted as a follow-up?
1. **Cache cleanup:** OK to delete the orphaned `clip11/cap*` run_folders after M2 validation?
