# Phase 3 task brief: auxiliary trajectory head for `PatchPolicy`

Self-contained task brief. Written for an agent with **no prior context** on
the conversation that produced it — everything needed to execute is here or
behind the file:line references. Parent document:
`docs/action_tokenizer_repair_plan.md` Phase 3 (steps 15-18) — that section now
also carries a "Caution added 2026-09-08" note this brief supersedes with a
concrete design. Sibling result document (do not modify, read-only reference):
`/nasa/alex/docs/round6_1cam_causal_results.md`.

## 1. Why this work exists

`docs/action_tokenizer_repair_plan.md` Phase 3 calls for an auxiliary
trajectory-prediction head on `PatchPolicy`, supervised by a dead-reckoned (not
GPS) future-ego-path target, at `clip_horizon: 6` / 2s, to see whether it helps
the policy the way similar auxiliaries do in related work.

## 2. Decisions already made — do not revisit

- **Port, don't rewrite from scratch.** A nearly-complete, tested
  implementation of this exact feature already exists on sibling branch
  `feat/patch-policy-decoder-causal-traj`, commit `af4fcf1` (originating from
  `feat/drivor`'s DrivoR reference, arXiv:2601.05083). It is not on this
  branch and needs porting/adapting — see §3 for what to reuse verbatim vs.
  what to change and why.
- **That same branch produced checkpoint `v9y58oei`**, which tournament round
  6 measured trending **negative** in closed-loop route completion with more
  of its own training budget (A4→A5: −0.081 rc over 1.67 epochs, vs. a
  budget-only expectation of +0.164 — see
  `/nasa/alex/docs/round6_1cam_causal_results.md` §3). That implementation
  adds `losses["trajectory"]` into the total loss **unweighted** — no
  per-term weight knob exists in it at all. An unweighted trajectory loss
  competing at full strength against the action losses is a concrete,
  testable candidate explanation for the regression, independent of any
  question about target quality (the target there is already dead-reckoned,
  not raw GPS — so "it's just a bad target" does not explain it away).
- **Add a weight knob defaulting to off (`0.0`)**, and enforce a gate
  discipline before spending real training budget: weight=0.0 must reproduce
  the pre-existing baseline bit-for-bit, only low weights get swept, and a
  closed-loop check is required before scaling further. This mirrors the LFG
  branch's own documented gate order ("weights 0.0 first, must reproduce
  baseline curve bit-for-bit, only then sweep {0.03, 0.1, 0.3}").
- **Scope is 1-camera only.** The Phase-3 target config, `dinov3.yaml`, uses
  the 1-camera datamodule (`/datamodule: yaak/train`). The 3-camera dataset
  templates (`train_3cam.yaml`/`val_3cam.yaml`) are deliberately **not**
  touched by this brief even though the current branch is named
  `feat/patch-policy-3cam-traj` — defer that schema edit until a
  3cam+trajectory arm is actually being built (project-owner decision).
- **The `mode_head` distillation extension is out of scope.** `af4fcf1`'s
  later commits `411f964`/`02a5583` add a classifier distilling the
  winner-takes-all oracle's `best_index`, plus decoder/export wiring
  (`PatchPolicyDecoder`/`PatchPolicyDecoderStep`, ONNX/TRT). Do not port
  either. This head is training-time-only — it never feeds `joint_actions` or
  any served/exported output — and training a mode head on top of a
  trajectory head that hasn't cleared its own closed-loop gate would compound
  risk rather than isolate it.

## 3. Established facts (verified against current code, cite instead of re-deriving)

- **`dead_reckon_future_trajectory`** already exists, tested, and correct at
  `src/rmind/components/dead_reckoning.py:11-68` — verbatim-ported from
  `feat/drivor` (see the file's own header comment: commit `920030a`, "fix:
  correct heading-convention mismatch"). Signature:
  `dead_reckon_future_trajectory(*, speed_kmh, heading_deg, time_stamp_s,
  reference_index=0) -> (position, heading)`. Inputs `(*batch, T)`; output
  `position (*batch, P, 2)` (meters, ego-centric, `P = T-1-t0`) --
  **superseded 2026-09-10**: this brief's `/100`-normalized was dropped, see
  `dead_reckoning.py`'s docstring; the loss balances xy (meters) against
  heading in degrees instead,
  `heading (*batch, P)` (radians, wrapped to `[-pi,pi]`). **Axis convention,
  do not re-derive**: `heading_deg` is a compass bearing (0°=north), so
  forward in the rotated ego frame is local **+y**, lateral is **+x** — the
  exact justification (matched against `ST_Rotate` production SQL) is in the
  code comment at `dead_reckoning.py:51-62`; this bug has reportedly been
  rediscovered twice already, use the tested function, do not reimplement the
  trig.
- **`time_stamp_s` must be float64 seconds**, not float32. Real Unix-epoch
  microsecond timestamps cast to float32 silently zero every `dt` (float32's
  ~7 significant figures can't hold sub-second resolution at that magnitude).
  Regression test: `tests/test_dead_reckoning.py::test_dead_reckon_survives_float32_epoch_timestamps`.
- **`af4fcf1`'s `rolling_dead_reckoned_trajectory`** (in a new file,
  `src/rmind/components/trajectory.py`, on that branch) reimplements the same
  physics as a second, independent "vectorized closed-form" function instead
  of calling `dead_reckon_future_trajectory` — a second place the
  axis-convention bug could reappear. **Do not port this file as-is**; see §4
  step 1 for the recommended alternative (a thin loop wrapper around the
  existing tested primitive).
- **`_features`** (`src/rmind/models/patch_policy.py:611-681`) has a stable
  2-tuple return `(features, chunk)` and already uses two out-parameters
  (`token_norms`, `turn_signal_chunk`) for auxiliary data that rides along
  without changing that signature — this is the established convention in
  this file for exactly the reason a new head needs it. ~9 existing call
  sites (production + tests) rely on the 2-tuple. `af4fcf1` widens this to a
  3-tuple, breaking all of them — **do not port that widening**; add a third
  out-parameter instead (§4 step 2).
- **`turn_signal_head`** (`patch_policy.py:210-304`, docstring at `:192-206`)
  is the precedent to mirror in every respect except blast radius: opt-in
  (`None` default), eager `__init__`-time `ValueError` if
  `losses["turn_signal"]` is missing (`:293-299`), reads the same per-frame
  readout feature `_features` already produces, loss/metrics added inline
  inside the existing `"policy"` top-level key so `_step` needs no changes.
- **`_step`** (`patch_policy.py:1030-1062`) sums every key under
  `policy.loss.*` generically (`metrics.select(*((k, "loss") for k in
  metrics.keys()))`) — no hardcoded key list. A new loss term added inside
  `"policy"` requires **no edit** to `_step`, `training_step`, or
  `validation_step`'s sanity-check branch (`:1070-1073`, which hardcodes
  `["policy","loss"]` — only relevant if a *separate* top-level key like
  LFG's `"aux"` were used, which this brief deliberately avoids).
- **The LFG `aux_heads`/`aux_weights` dict machinery** (multi-term weighted
  loss group, lives on a third, unrelated branch,
  `feat/patch-policy-decoder-causal-lfg`) has **zero references** anywhere in
  this repo's current `src/`/`config/` — only `commands.sh`'s CLI overrides
  for a different experiment family reference `aux_weights`, and that
  experiment config doesn't exist in this tree. Building that machinery here
  for one consumer is unjustified; a single scalar weight is sufficient (§4
  step 3).
- **Dataset schema**: `headings_denoised/heading` is already computed in the
  1-camera dataset SQL (feeds the waypoint rotation) then explicitly dropped
  via `* EXCLUDE (...)` in `config/_templates/dataset/yaak/{train,val,train_debug}.yaml`.
  `time_stamp` (`meta/ImageMetadata.cam_front_left/time_stamp`) is already
  selected in every template's cast list — no SQL change needed for it, only
  a new `Remapper` path.
- **Cache invalidation risk**: the rbyte `run_folder` path is keyed only on
  `clip_length`/`episode_stride`/`episode_offset`
  (`config/_templates/dataset/yaak/train.yaml:~707-717`), not on column
  schema. Since `clip_horizon`/`clip_length` stay at `6`/`11` (unchanged by
  this work), adding the `heading` column does not change `run_folder` — a
  stale on-disk samples cache from a prior run risks being served without the
  new column. A second `pipefunc` disk-cache layer's key composition was not
  verified either way.
- **`dinov3.yaml`** (`config/experiment/yaak/patch_policy/dinov3.yaml`):
  `episode_length: 6`, `clip_horizon: 6` → `clip_length: 11`, `/datamodule:
  yaak/train` (1-camera), `action_tokenizer_artifact:
  yaak/rmind/model-q6ocue9a:v9` (the Phase 2a 3-feature artifact).

## 4. Work steps, in dependency order

### Step 1 — dead-reckoning rollup, `src/rmind/components/dead_reckoning.py`

Add `rolling_dead_reckoned_trajectory(*, speed_kmh, heading_deg,
time_stamp_us, episode_length, num_poses) -> Tensor`, shape `(*batch,
episode_length, num_poses, 3)` (last dim `x, y, theta`). Implement as a loop
over anchors, **calling the existing tested `dead_reckon_future_trajectory`**,
not a reimplementation:

```python
def rolling_dead_reckoned_trajectory(
    *, speed_kmh: Tensor, heading_deg: Tensor, time_stamp_us: Tensor,
    episode_length: int, num_poses: int,
) -> Tensor:
    *_batch, t = speed_kmh.shape
    needed = episode_length + num_poses
    if t < needed:
        msg = f"need {needed} steps (episode_length + num_poses), got {t}"
        raise ValueError(msg)

    time_stamp_s = time_stamp_us.double() / 1e6  # see step's float64 note above
    poses = []
    for t0 in range(episode_length):
        position, heading = dead_reckon_future_trajectory(
            speed_kmh=speed_kmh, heading_deg=heading_deg,
            time_stamp_s=time_stamp_s, reference_index=t0,
        )
        poses.append(torch.cat(
            [position[..., :num_poses, :], heading[..., :num_poses, None]], dim=-1
        ).float())
    return torch.stack(poses, dim=-3)  # (*batch, episode_length, num_poses, 3)
```

Sizing check: for anchor `t0`, `dead_reckon_future_trajectory` returns `P =
T-1-t0` poses; since `T = needed`, `P >= num_poses` for every `t0 <
episode_length`, so the slice is always safe. At `episode_length=6` this is 6
cheap calls on ≤11-element tensors — negligible next to one ViT forward pass;
there is no real efficiency case for a separate closed-form rewrite at this
scale.

**Tests** (extend `tests/test_dead_reckoning.py`):
`test_rolling_trajectory_shape`,
`test_rolling_trajectory_raises_when_not_enough_future_context`,
`test_rolling_trajectory_heading_is_wrapped` (the `-179°`/`179°` case),
`test_rolling_trajectory_batched`, and
`test_rolling_trajectory_matches_dead_reckon_future_trajectory_at_every_anchor`
(a plumbing check against the already-proven primitive, not a fresh algebra
proof).

### Step 2 — loss module, `src/rmind/components/loss.py`

Port from `af4fcf1` verbatim (dtype/shape-agnostic, no changes needed):
- `_per_candidate_pose_errors(input, target) -> (xy_err, heading_err)`
- `winner_takes_all_pose_l1(input, target, *, heading_weight=0.1,
  reduction="mean") -> (loss, best_index, per_candidate_loss)` — `torch.min`
  over `Q` hypotheses is natively differentiable; only the winner gets
  gradient.
- `winner_takes_all_pose_l1_components(...)` — same, plus separate xy/heading
  terms for logging.
- `WinnerTakesAllPoseLoss(Module)` — 2-arg `forward(input, target)`, matching
  this file's existing `losses[...]` convention.

**Tests** (new file, e.g. `tests/test_loss.py` — check none already covers
this module before creating): winner-takes-all gradient reaches only the
winning hypothesis; `WinnerTakesAllPoseLoss` matches the plain function;
`winner_takes_all_pose_l1_components`'s `loss` output matches the plain
function's `loss`.

### Step 3 — trajectory target module, `src/rmind/components/nn.py`

Add `TrajectoryTarget(Module)`, ported from `af4fcf1` verbatim except its
internal call imports `rolling_dead_reckoned_trajectory` from
`rmind.components.dead_reckoning` (per step 1 — do not create a separate
`trajectory.py` module). Constructor: `speed`, `heading`, `time_stamp` (input
paths), `out` (write path), `episode_length`, `num_poses`. Wired into
`input_transform` **before** `ChunkFields` (needs the full un-truncated time
axis, which `ChunkFields` truncates). Emits `None` if any input is missing.

### Step 4 — `PatchPolicy` wiring, `src/rmind/models/patch_policy.py`

1. Constructor additions, placed immediately after the existing
   `turn_signal_head` block (`:290-299`):
   ```python
   trajectory_head: HydraConfig[Module] | InstanceOf[Module] | None = None,
   ...
   trajectory_target: Path = ("context", "trajectory_target"),
   num_trajectory_hypotheses: int = 5,
   trajectory_weight: float = 0.0,
   ```
   ```python
   self.trajectory_head: Module | None = init_hydra_param(
       hparams, "trajectory_head", trajectory_head
   )
   if self.trajectory_head is not None and "trajectory" not in self.losses:
       msg = (
           "trajectory_head requires losses['trajectory'] (e.g. "
           "rmind.components.loss.WinnerTakesAllPoseLoss) to supervise it"
       )
       raise ValueError(msg)
   self.trajectory_target: Path = trajectory_target
   self.num_trajectory_hypotheses = num_trajectory_hypotheses
   self.trajectory_weight = trajectory_weight
   ```
   (`af4fcf1` lacks this eager validation gate — closing that gap is
   deliberate, matching `turn_signal_head`'s precedent. Also add
   `trajectory_target`/`num_trajectory_hypotheses`/`trajectory_weight` to
   `hparams` so they round-trip through checkpoints.)

2. `_features`: add a third out-parameter,
   `trajectory_target_out: dict[str, Tensor] | None = None`, filled under key
   `"trajectory_target"` exactly like `turn_signal_chunk` is filled under
   `"turn_signal"` (`:644-647`). Return signature stays `(features, chunk)` —
   **no existing call site changes**.

3. New method:
   ```python
   def _predict_trajectory(self, features: Tensor) -> Tensor:
       pred = self.trajectory_head(features)
       return rearrange(
           pred, "... (q p c) -> ... q p c",
           q=self.num_trajectory_hypotheses, c=3,
       )
   ```

4. `_compute_metrics`, right after the `offset` loss:
   ```python
   trajectory_pred: Tensor | None = None
   trajectory_target: Tensor | None = None
   if self.trajectory_head is not None:
       trajectory_target = trajectory_target_out["trajectory_target"]
       trajectory_pred = self._predict_trajectory(features)
       losses["trajectory"] = self.trajectory_weight * self.losses["trajectory"](
           trajectory_pred, trajectory_target
       )
   ```
   Extend the gradient-free metrics block (`_readout_metrics`) with
   `trajectory_loss_xy`, `trajectory_loss_heading`,
   `trajectory_best_index_unique_frac` via
   `winner_takes_all_pose_l1_components` — keep these **unweighted** (raw
   units) regardless of `trajectory_weight`, since they're diagnostic and
   should stay comparable across a weight sweep.

5. `forward`: no change — already ignores `_features`'s second return value.

6. `predict_step`: add a new top-level `"trajectory"` result branch
   (`prediction`, `best_prediction`, `best_index`, `per_candidate_loss`,
   `ground_truth`), parallel to the existing `"policy"` branch — see
   `af4fcf1`'s version for the exact `winner_takes_all_pose_l1` + `gather`
   shape logic, adapted to call `_features(..., trajectory_target_out={})`
   instead of unpacking a 3-tuple.

7. Class docstring: add a `trajectory_head` paragraph mirroring
   `turn_signal_head`'s (`:192-206`) — opt-in, `trajectory_weight=0.0`
   reproduces old behavior exactly, cites the round-6 caution and gate
   discipline, notes it never feeds `joint_actions`.

**Tests** (extend `tests/test_patch_policy.py`, mirroring the
`turn_signal_head` test group): `_make_model`/`_make_batch` gain
`with_trajectory_head`/`with_trajectory_target` kwargs. Add:
- `test_trajectory_head_requires_a_trajectory_loss` (mirrors
  `test_turn_signal_head_requires_a_turn_signal_loss`) — the eager-gate test
  `af4fcf1` lacks.
- `test_trajectory_head_absent_by_default`
- `test_trajectory_head_metrics_and_gradients`
- `test_trajectory_head_predict_step`
- **`test_trajectory_weight_zero_reproduces_baseline_bit_for_bit`** — the
  actual gate this brief adds beyond `af4fcf1`: build two models (one with
  `trajectory_head=None`, one with it set but `trajectory_weight=0.0`), fill
  parameters deterministically (reuse
  `tests/test_training_step_snapshot.py`'s RNG-free `_fill_deterministic`
  pattern rather than relying on matched seeds across separate constructions),
  and assert every pre-existing loss/metric matches with `rtol=0, atol=0`,
  plus `losses["trajectory"].item() == 0.0`. This turns "weight 0.0 must
  reproduce baseline bit-for-bit" into a cheap, exact, CI-checkable unit test
  instead of a real-training-run curve diff.

No existing `_features(...)` call site (production or test) needs updating —
the out-parameter choice in step 4.2 is what buys this.

### Step 5 — config, `config/model/yaak/patch_policy/raw.yaml`

- Add to the `Remapper`'s `context` block:
  ```yaml
  heading: [data, headings_denoised/heading]
  time_stamp: [data, "meta/ImageMetadata.cam_front_left/time_stamp"]
  ```
- Add a `TrajectoryTarget` stage to `input_transform`, immediately before
  `ChunkFields`.
- Add `trajectory_head: null` (architecture pieces defined here, but the real
  MLP only gets instantiated in `dinov3.yaml` — see §2's blast-radius
  decision) and `num_trajectory_hypotheses: ${num_trajectory_hypotheses}`.
- Add to `losses`: `trajectory: {_target_:
  rmind.components.loss.WinnerTakesAllPoseLoss, heading_weight:
  ${trajectory_heading_weight}}` (safe to always define — the validation gate
  only fires when `trajectory_head is not None`).

### Step 6 — config, `config/experiment/yaak/patch_policy/dinov3.yaml`

```yaml
# auxiliary trajectory head (docs/phase3_trajectory_head_plan.md).
# trajectory_weight=0.0 is the SAFE default: the head is constructed and
# exercised end-to-end every step, but contributes exactly zero loss/gradient,
# so this arm is provably identical to the no-trajectory-head baseline (see
# tests/test_patch_policy.py::test_trajectory_weight_zero_reproduces_baseline_bit_for_bit).
# Only raise this after that gate passes AND a closed-loop (rsim) check clears
# -- round 6 (/nasa/alex/docs/round6_1cam_causal_results.md) found an
# UNWEIGHTED version of this same idea trending NEGATIVE in route completion.
trajectory_horizon: 5   # episode_length + trajectory_horizon must fit clip_length (11);
                        # needs one MORE future step than action_horizon's chunk
num_trajectory_hypotheses: 5
trajectory_heading_weight: 0.1
trajectory_weight: 0.0

model:
  trajectory_head:
    _target_: torchvision.ops.MLP
    in_channels: ${policy_embedding_dim}
    hidden_channels:
      [1024, 1024, "${eval:'${num_trajectory_hypotheses} * ${trajectory_horizon} * 3'}"]
  trajectory_weight: ${trajectory_weight}
```
`trajectory_weight` is a top-level experiment var (not just
`model.trajectory_weight`) specifically so it can be swept with a single CLI
override (`trajectory_weight=0.03`) — no experiment-file fork needed.

### Step 7 — dataset schema (1-camera only)

Edit templates, not generated files (`just generate-config` regenerates
`config/dataset/` from these):
`config/_templates/dataset/yaak/{train,val,train_debug}.yaml`. For each:
1. Remove `"headings_denoised/heading"` from the `* EXCLUDE (...)` list.
2. Add it to the `samples_cast` slice-and-cast SELECT, next to `speed`:
   ```sql
   "headings_denoised/heading"[1:${clip_length}]::FLOAT[${clip_length}] AS "headings_denoised/heading",
   ```
Confirm exact line numbers after `just generate-config` — they may have
shifted since this brief was written.

Do **not** touch `train_3cam.yaml`/`val_3cam.yaml` (§2 scope decision).

### Step 8 — cache invalidation

Point this work at a **fresh** `paths.rbyte.cache` directory (e.g.
`.rbyte_cache_traj`) rather than trying to verify whether the existing cache
layers would detect the schema change — matches existing precedent
(`.rbyte_cache_causal32_lfgaux`, `.rbyte_cache_32step`, etc.). Do not pass
`resume=true` on the first run against it, and do not run two launches
against the same new cache dir concurrently (single-writer caveat — see
`[[project_rbyte_cache_contamination_risk]]`).

## 5. Verification sequence

1. `just test tests/test_dead_reckoning.py tests/test_patch_policy.py` plus
   the new loss test file, then `just test` for the full suite, then `just
   lint && just format && just typecheck`.
2. Hydra compose smoke check: confirm `dinov3.yaml` resolves
   `trajectory_head` to the real MLP and `trajectory_weight: 0.0`, and that
   other patch_policy arms still resolve `trajectory_head: null` unchanged.
3. `just train-debug experiment=yaak/patch_policy/dinov3
   paths.rbyte.cache=.rbyte_cache_traj_debug` — once at the default
   (`trajectory_weight=0.0`), once with `++model.trajectory_head=null` (the
   true pre-Phase-3 baseline). Confirm no crash and identical non-trajectory
   loss/metric keys in both — this layer catches dataset-plumbing bugs (a
   missing column, a path typo) the unit test can't.
4. One short real training run at `trajectory_weight=0.0` against the fresh
   cache dir, loss curves compared against the current baseline
   (`do8m9ot8`/`v4mma4th`, or `dinov3.yaml`'s own current baseline). Narrower
   purpose than step 1's exact unit test (GPU non-determinism makes literal
   bit-for-bit unrealistic across separate launches) — this catches
   interactions the unit test's synthetic shapes can't (`torch.compile`,
   mixed precision, the real tokenizer artifact).
5. Low-weight sweep only after 1-4 pass: `trajectory_weight ∈ {0.03, 0.1}` —
   the repair plan's own range; hold off on `0.3` given round 6's caution
   applies more directly here than it did to LFG.
6. **Closed-loop (rsim) check before any further budget** —
   `benchmark.n_runs=3` (never `1`; round 3's `n_runs=1` estimates were off by
   up to 0.095), paired per scenario. Non-negotiable: round 6's regression was
   only visible in closed-loop route completion, never in a loss curve — a
   good offline loss on the trajectory head must not be read as evidence the
   policy head is unharmed.

## 6. Risk register

| risk | where it bites | mitigation |
| --- | --- | --- |
| Stale rbyte/pipefunc cache serves pre-schema data | dataset load, silent | §4 step 8: fresh cache dir, no `resume=true` on first run |
| Axis-convention regression reintroduced | dead-reckoning math | §4 step 1: call the tested primitive, don't reimplement trig |
| `time_stamp` float32 truncation zeros `dt` | dead-reckoning math | §4 step 1: `time_stamp_us.double() / 1e6` before calling the primitive |
| Trajectory loss dominates shared-trunk gradient | training dynamics, closed-loop only | `trajectory_weight` knob, default 0.0, gated sweep (§5 steps 4-6) |
| Good offline trajectory loss masks a closed-loop regression | judging the arm | §5 step 6 is mandatory before any budget increase, per round 6 |
| `_features` tuple-arity churn from future heads | test/prod maintenance | §4 step 4.2: out-parameter convention, not tuple widening |
| Universal rollout blocked on dataset schema for every arm | every patch_policy experiment | §4 step 5: `trajectory_head: null` default in `raw.yaml`, only on in `dinov3.yaml` |

## 7. Out of scope (explicitly, not silently dropped)

- `mode_head` distillation (`af4fcf1`'s `411f964`/`02a5583`).
- Any export/decoder wiring (`PatchPolicyDecoder`/`PatchPolicyDecoderStep`,
  ONNX/TRT) — this head never serves.
- 3-camera dataset templates — revisit when a 3cam+trajectory arm is actually
  being built.
