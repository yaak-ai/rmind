# Serving max_speed: how the speed-limit token reaches the model in drivr

Status: design + unapplied patch sketch (`docs/serving_max_speed.patch`).
Scope: the on-car runtime `drivr` (repo `yaak-ai/drivr`, analyzed at branch
`feat/configurable-decode-strategies`, HEAD `ae755b4`) serving the
map-context PatchPolicy exports from this branch (`feat/map-context`).

---

## 1. Today: how PatchPolicy engines are fed

drivr (`src/drivr/app/drivr.py`) runs a **single static-shape TRT engine**
over a rolling 6-frame history window (`NUM_TIMESTEPS = 6`, one frame per
~333 ms camera tick).

### Episode batch layout

`DriveApp._build_episode_batch()` produces a dict keyed by the **canonical
dotted names** (`drivr/engine/model_io.py::CANONICAL_INPUT_KEYS`):

| canonical batch key                              | shape           | dtype   | source |
|--------------------------------------------------|-----------------|---------|--------|
| `data.cam_front_left`                            | (1, 6, 3, H, W) | float32 | camera, `preprocess_image` (norm per `--image-norm`) |
| `data.meta/VehicleMotion/speed`                  | (1, 6, 1)       | float32 | CAN |
| `data.meta/VehicleMotion/gas_pedal_normalized`   | (1, 6, 1)       | float32 | CAN |
| `data.meta/VehicleMotion/brake_pedal_normalized` | (1, 6, 1)       | float32 | CAN |
| `data.meta/VehicleMotion/steering_angle_normalized` | (1, 6, 1)    | float32 | CAN |
| `data.meta/VehicleState/turn_signal`             | (1, 6, 1)       | int32   | CAN |
| `data.waypoints/xy_normalized`                   | (1, 6, 10, 2)   | float32 | route grid window, ego-normalized |

### Binding

`TRTEngine.__init__` reads the engine's IO tensor list and dtypes **from the
engine itself** and resolves each engine input name to a canonical batch key
via `resolve_input_names` (exact match -> `data.`-prefix strip -> flattened
form -> last-segment alias). This is the PR#27 philosophy end to end:
*derived rather than configured* — the engine declares what it needs, the
host adapts, and an engine input that cannot be resolved fails **loudly at
load time** (`KeyError`), never silently feeds garbage.

The actual rmind PatchPolicy ONNX exports (verified on
`/nasa/max/models/patch_policy/patch_policy_dinov2_dinowm-ifuusvwq-v5.onnx`)
name their graph inputs in the **flattened** form the dynamo exporter
produces:

```
batch_data_cam_front_left
batch_data_meta_vehiclemotion_speed
batch_data_waypoints_xy_normalized
```

which drivr's `_flat()` resolution step (strip `batch_`/`data_`, separators
to `_`) maps back onto the dotted canonical keys.

`TRTEngine.run` binds by name (static shapes, no `set_input_shape`), casts
each array to the engine's declared dtype, executes, and collapses the
output. The PatchPolicy `finetuned` exports have a single output
`policy.joint_actions` `(1, 6, 4)` — recognized as an **action plan**
(`extract_action_plan`), so drivr executes the ~2 s horizon between
inferences instead of re-predicting every frame.

---

## 2. Export side: exactly what the maxspeed export consumes

Model code: `src/rmind/models/patch_policy.py` (+
`src/rmind/components/map_context.py`), branch `feat/map-context`.

- **Extra graph input**: yes — one new input, per-frame, alongside the
  legacy inputs. Batch path `[data][meta/MapContext/max_speed]`, so the
  exported graph input name will be
  **`batch_data_meta_mapcontext_max_speed`** (flattened dynamo form; the
  canonical dotted name is `data.meta/MapContext/max_speed`).
- **Shape**: `(1, 6, 1)` — **per-frame**, one value per history timestep
  (matches every other scalar channel). Not a scalar.
- **Dtype**: `float32`, raw **km/h**. Sentinels: `NaN` = unknown, any
  negative value (convention `-1.0`) = explicitly unlimited (autobahn
  `maxspeed=none`), any value `<= 7` = WALK / Schrittgeschwindigkeit.
- **Tokenization is IN-GRAPH** — the host sends float km/h, *not* token
  ids. `MaxSpeedTokenizer` (isnan/where/argmin, ONNX-traceable by design)
  snaps to the 13-class German vocabulary inside the graph: 0 UNKNOWN,
  1 UNLIMITED, 2 WALK, 3..12 = [10,20,30,50,60,70,80,100,120,130] km/h,
  nearest-with-tie-down (40 -> 30), > 130 clamps to 130. The host never
  needs the vocabulary.
- **Absent input** (models the `context.max_speed` Remapper path yielding
  `None`) falls back to an all-UNKNOWN token **in eager rmind only**. In an
  ONNX/TRT export the input is part of the traced graph, so serving cannot
  "omit" it — sending **all-NaN is the equivalent** (NaN -> UNKNOWN
  in-graph) and is exactly the distribution the model saw under the 0.3
  train-time dropout.
- **Baked-constant alternative**: `+model.max_speed_override=<kmh>` at
  export time traces `torch.full(...)` into the graph as a constant; the
  resulting engine has **no** max-speed input at all and ignores any feed
  (useful for a fixed-zone test engine, useless for live limits).

### Export config status (round 1): correct, one nit

`config/export/yaak/patch_policy/finetuned_maxspeed.yaml` layers the extra
input over `finetuned.yaml`:

```yaml
input:
  data:
    meta/MapContext/max_speed:
      _target_: torch.testing.make_tensor
      _args_: [1, 6, 1]
      dtype: torch.float32
      low: 5.0
      high: 130.0
```

Checked against the export pipeline (`config/export_onnx.yaml` passes
`args: [${input}]` into `rmind-export-onnx`; hydra merges the `input.data`
dicts by key, so all four legacy inputs survive): shape, dtype, nesting and
the defaults chain are all right. **Not broken.** Two things to know:

1. The sample tensor is only a tracing example — `make_tensor` can't emit
   NaN/-1, but that doesn't matter: the NaN/negative branches are tensor
   ops (`where`/`isnan`), not Python control flow, so they are traced
   regardless. No fix needed.
2. **Verification item for the first real export** (not a config change):
   confirm the built TRT engine preserves the sentinel semantics —
   `IsNaN` must survive the ONNX->TRT conversion (TRT supports the op, but
   fp16 builder passes and `--builderOptimizationLevel=5` should be
   spot-checked by feeding NaN / -1 / 30 / 130 and comparing token-level
   behavior against the ONNX runtime). A quiet `NaN -> garbage class`
   regression would silently un-train the missing-map behavior.

No maxspeed ONNX artifact exists yet in `/nasa/max/models/patch_policy/`
(round-1 checkpoint still training); the interface above is fixed by the
config + model code, so the serving work can proceed in parallel.

---

## 3. Proposal v1 — test ground: constant zone limit via `--max-speed`

Goal: on the private test ground (no legal limit applies) drive the same
engine "as if" it were a 30 km/h zone, a 10 km/h zone, or a parking lot
(Schrittgeschwindigkeit), and compare behavior.

Design (see `docs/serving_max_speed.patch`, unapplied):

1. **Optional binding, discovered from the engine** (PR#27 pattern).
   - `model_io.py`: add `MAX_SPEED_KEY = "data.meta/MapContext/max_speed"`
     to `CANONICAL_INPUT_KEYS` + a `max_speed` last-segment alias. All
     four resolution stages then cover the exported name
     (`batch_data_meta_mapcontext_max_speed` flattens to
     `meta_mapcontext_max_speed`, which matches the canonical key's flat
     form exactly).
   - `TRTEngine` exposes `wants_max_speed = MAX_SPEED_KEY in
     self.input_map.values()` (and logs it at load). Engines without the
     binding are legacy: resolution is driven by the **engine's** input
     list, so nothing changes for them — no flag, no config.
2. **`--max-speed <kmh|unlimited>` CLI** -> `DriveApp.max_speed:
   float | None`. Parsing: float km/h; `unlimited` -> `-1.0`; values <= 7
   (e.g. `5`) mean Schrittgeschwindigkeit; **absent -> `None`**.
3. **Feed**: `_build_episode_batch` always adds
   `data.meta/MapContext/max_speed` = `np.full((1, 6, 1), fill, float32)`
   where `fill = max_speed` or `NaN` when unset. The key is simply present
   in the host batch; `TRTEngine.run` only binds keys the engine asked for,
   so legacy engines never see it. Constant across the 6 history frames and
   re-fed every inference — the zone doesn't change mid-window on the test
   ground, and per-frame granularity is already in the interface for v2.

Semantics summary (one line to remember): **float km/h; `-1` unlimited;
`NaN`/unset -> model-internal UNKNOWN** — identical to the training
contract, no serving-side vocabulary.

Mismatch handling: `--max-speed` against a legacy engine logs a warning
(flag ignored); a maxspeed engine without `--max-speed` logs that it is
running in the trained missing-map (UNKNOWN) mode. Both are one-line
`_load_model_async` checks in the patch.

Deferred (deliberately not in the sketch): a live `/api/max_speed` endpoint
+ sidebar control mirroring the `/api/decode` pattern, so the zone can be
switched without restarting. Trivial to add once v1 is validated.

## 4. Proposal v2 — road: live limit lookup

drivr already carries the two ingredients:

- **GPS position per step** — `_gnss_fix` / the per-timestep
  `EpisodeTimestep.latitude/longitude` (RTK-grade with NTRIP).
- **A road network + snapping** — `drivr/io/map_match.py::RoadNetwork`
  (loaded from GeoJSON via `--map-match-file`), today used to snap
  hand-drawn waypoints (`_matched_waypoints`, 50 m gate).

v2 extends the network payload: bake `max_speed_kmh` per LineString into
the map-match GeoJSON (the map-GT sidecar pipeline
`src/rmind/scripts/map_gt/` already produces exactly this mapping from
per-drive OSM data — way geometry -> maxspeed float with the same NaN/-1
conventions). Each inference step:

1. Snap the current GNSS fix to the nearest segment (`RoadNetwork.snap`,
   same ≤ 50 m gate).
2. Take that segment's `max_speed_kmh` as this step's value; each of the 6
   history frames keeps the value active when its frame was captured (the
   `(1, 6, 1)` interface is per-frame precisely so a zone change rolls
   through the window instead of teleporting).
3. **Staleness/fallback -> UNKNOWN (NaN), never hold**: no fix, fix older
   than ~2 s, snap distance > gate, or way without a tagged limit all feed
   NaN for that frame. UNKNOWN is in-distribution by construction (0.3
   training dropout), so degradation is graceful rather than a frozen wrong
   limit. Precedence: `--max-speed` override (test ground) > live lookup >
   NaN.

### The deterministic shield sits BESIDE the model, not inside it

Conditioning is a behavioral prior, not a guarantee. The hard guarantee
belongs in the actuation path, after the model:

```
camera/CAN/GPS -> episode batch -> TRT engine -> plan/action
                                                    |
             map limit (same lookup) -> shield: cap commanded speed
                                                    |
                                            MCM setpoints
```

Concretely: in `_run_inference_core`, between plan/action selection and
`_mcm_controller.send_setpoints(...)`, a `SpeedShield` clamps gas (and, if
needed, injects brake) whenever CAN speed exceeds the active map limit + a
small margin. It reuses the *same* lookup value that fed the model, but is
pure host code: auditable, testable, unaffected by training. When the limit
is UNKNOWN the shield falls back to a configured test-ground ceiling rather
than being disabled. The shield is out of scope for the v1 patch (test
ground has a human + MCM override chain) but the insertion point is fixed
by this design.

---

## 5. Failure modes

| failure | behavior | verdict |
|---|---|---|
| wrong dtype/shape fed to the binding | host casts to the engine's declared dtype; a genuinely wrong shape fails **loudly** as a TRT error at load/execute — same as any other binding | good (fail-fast) |
| engine input name unresolvable (naming drift in a future export) | `resolve_input_names` raises `KeyError` at **load** | good (fail-fast). Note this cuts both ways: TODAY's drivr cannot even load a maxspeed engine — the patch is a prerequisite, not an optimization |
| maxspeed engine, no map / no `--max-speed` | all-NaN -> in-graph UNKNOWN every frame — the exact training-dropout condition; model drives on visual evidence alone | by design; logged at load |
| UNKNOWN-flood mid-drive (v2: GPS outage, off-network) | per-frame NaN rolls through the 6-frame window; behavior degrades to the no-map policy, no latching of stale limits | by design; shield keeps the hard cap |
| legacy engine + `--max-speed` | value ignored (engine has no binding); warning logged | acceptable |
| NaN/-1 sentinel mangled by TRT builder (fp16/optimization) | silent wrong-class tokens | **must be verified once per engine build** (section 2); add a 4-value probe (NaN/-1/30/130) to the engine-build checklist |
| zeros accidentally fed instead of NaN | 0 km/h <= 7 -> WALK token, i.e. "parking lot everywhere" — plausible and wrong | that is why the fallback fill is NaN, never `np.zeros`; do not "shape-safe zero-fill" this input |

The last row is the one real trap: for every other channel drivr's
zero-fill fallback is shape-safe *and* semantically safe; for max_speed,
zero is a *strong* semantic claim (walking speed). NaN is the neutral
element of this vocabulary.

---

## 6. Patch sketch

`docs/serving_max_speed.patch` — unified diff against `yaak-ai/drivr` @
`ae755b4` (`feat/configurable-decode-strategies`), verified with
`git apply --check`. **Not applied anywhere.** Touches:

- `src/drivr/engine/model_io.py`: `MAX_SPEED_KEY`, canonical key list,
  segment alias.
- `src/drivr/app/drivr.py`: `TRTEngine.wants_max_speed` discovery + load
  log; `DriveApp.max_speed` field; NaN-or-constant feed in
  `_build_episode_batch`; load-time mismatch logs; `_parse_max_speed` +
  `--max-speed` CLI; constructor pass-through.
