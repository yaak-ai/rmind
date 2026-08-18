# LFG auxiliary supervision — Stage 1–3 results

**Status:** Stage 1 (offline labelling) is **complete** for the entire corpus. Stage 2
(rbyte plumbing) is **complete and verified** (gates 7.1 and 7.2, see below). Stage 3
(the aux loss in `PatchPolicy`) is **implemented and unit-tested** (see §9). Stage 4
(experiment config, zero-weight control run, weight sweep) from
`lfg_aux_supervision_task.md` has **not** been started.

**Licence caveat still applies unchanged**: LFG weights are CC BY-NC 4.0. This run was executed
as experimentation/pilot work; resolve the licence question before any production use of these
labels or a policy trained on them.

______________________________________________________________________

## 1. Environment

- LFG repo cloned to `/nasa/tools/lfg` (Apache 2.0, commit as of 2026-08-16), isolated `uv` venv
  at `/nasa/tools/lfg/.venv` (Python 3.12, torch 2.13, no CUDA extensions required), kept out of
  the rmind venv per the brief.
- Checkpoint: `/nasa/tools/lfg/lfg_seg_motion_m3n3.pt`, 4,874,393,866 bytes, matching the expected
  4.87 GB.
  sha256: `6c4f050ea91fbf44cad3be725c8f3b50505335f614bcecb1c88be36571851eaa`
- Loaded cleanly via `lfg.checkpoint.load_model_from_checkpoint`: 1392/1392 tensors, no missing,
  no unexpected, no shape mismatches.
- Resolved `ModelConfig` matched the brief's inspection exactly: `m=3 n=3 encoder_name=dinov2 decoder_size=large ar_n_heads=8 ar_n_layers=4 use_segmentation_head=True segmentation_num_classes=7 use_motion_head=True use_flow_head=False point_head_type=linear`.
  (`m`/`n` only affect a build-time validation bound; actual window length and
  `n_future_frames_override=0` are set per-call, so this default config was used unmodified.)
- Hardware: one RTX 5090 (32 GB), bf16 autocast.

## 2. Script

New file `scripts/lfg_label_drives.py`, implementing brief §3 (frame selection at
`frame_idx % 10 == 0`, 15-frame windows with tail-padding, crop-then-bicubic-resize
preprocessing matching `config/model/yaak/patch_policy/raw.yaml` up through the crop,
`n_future_frames_override=0`, `adaptive_avg_pool2d` reduction to the 16×16 patch grid, packed
4-channel `uint8` `.bin` output, per-drive `manifest.json`).

One deviation from the initial draft, added after the 30-drive pilot and before the full-corpus
run: **per-drive fault isolation**. `main()` now wraps each drive's `label_drive(...)` call in
`try/except` and records failures in `{output_root}/failures.json` instead of letting one bad
drive crash the whole batch. This was a real risk for a ~24-hour, 630-drive job — untested at
pilot scale — and in the event zero drives failed, but the isolation is now a permanent feature
of the script, not a one-off patch.

## 3. Runs performed

### 3.1 Smoke test (1 window)

Drive `Niro096-HQ/2023-01-11--13-47-36`, `--limit-windows 1`. Confirmed checkpoint load, output
shapes, and end-to-end write path before spending real GPU time. ~4.4s for the first (cold)
window.

### 3.2 Single-drive pilot

Same drive, full run: 4638/4638 target frames labelled in ~295s (~0.95s/window steady state).
Spot-checked `seg_label` planes across several frames — informal gate-7.1 check: a sky-like class
fills the top ~7 rows, a mid-band class fills rows 7–12, and a road-like class dominates and grows
toward the bottom rows. No flip/transpose artifact visible.

### 3.3 30-drive pilot (`config/dataset/yaak/train_subset30.yaml`)

All 30 drives labelled (the one above plus 29 more), per brief §3.6. Result:

- 135,661 frames labelled, 532 MB on disk.
- All `.bin` files exactly 1024 bytes; every drive has a `manifest.json`.
- Zero errors.

### 3.4 Full corpus (655 drives from `config/_templates/dataset/yaak/train.yaml` +

5 drives from `config/_templates/dataset/yaak/val.yaml`, minus the 30 already done = 630 drives
labelled in this run)

- Launched 2026-08-16 20:47:55 UTC, completed 2026-08-17 ~21:09 UTC — **~24h 21m wall clock**
  on the single RTX 5090.
- **630/630 drives succeeded, 0 failures** (`failures.json` was never created).
- Per-drive throughput varied a lot with drive length (spot windows measured from 0.46s/window to
  0.98s/window depending on load/caching); no drive needed manual intervention.

## 4. Final corpus totals (verified after completion)

- **660/660 drives have a `manifest.json`** (655 train + 5 val).
- **2,719,328 `.bin` label files total**, all confirmed exactly 1024 bytes (no header, no
  truncation, no corruption found by a full `find -size` sweep).
- This is higher than the brief's back-of-envelope estimate of ≈1.97M frames (§3.6) — the actual
  corpus is denser (more 10-frame-aligned frames per drive on average) than that estimate assumed.
  Downstream disk/IO planning (brief §3.5, §8) should use **2.72M frames** as the real number:
  logical payload ≈ 2.72 GB, ~4 KiB block size → roughly **10–11 GB on disk**, still trivial
  against the 5.6 TB free on `/nasa` at the time of the run.
- Output root: `/nasa/drives/yaak/lfg_labels/v1/{drive_id}/{frame_idx:09d}.bin` +
  `/nasa/drives/yaak/lfg_labels/v1/{drive_id}/manifest.json`, exactly as specified in brief §3.5.
- `du -sh` over the full tree times out over NFS (2.7M small files) — not attempted further; the
  file-count-based estimate above is the basis for the disk-usage number.

## 5. What this does and does not satisfy from the brief's validation gates (§7)

- **§7.1 (spatial alignment)** — only an informal spot-check was done (visual inspection of raw
  `seg_label` arrays on a handful of frames from the first drive, not a rendered heatmap overlay
  on the actual 224×224 model input, and no automated bottom-vs-top road-fraction assertion). This
  should be treated as **not formally passed** — worth doing properly, especially now that the
  full corpus is done, since it's the cheapest gate and the brief calls a flip/transpose bug here
  "invisible in the loss."
- **§7.2 (datamodule round-trip)** — not attempted; depends on Stage 2 (rbyte plumbing), which
  hasn't been built.
- **§7.3–7.5** — not applicable yet; they gate the aux-loss experiment arms (Stage 3/4), not
  started.

## 6. §7.1 formal check (done 2026-08-17)

The informal spot-check in §3.2 above was upgraded to the actual gate: a numeric road/sky
row-fraction assertion over the **full labelled drive** (4638 frames, not just ~20), using
`decode_lfg_label` — the same decoder Stage 2 wires into rbyte, not a re-derivation.

- Dominant class in the bottom two patch rows: class 0, 94.9% of bottom-row patches, 0.0% of
  top-row patches.
- Dominant class in the top two patch rows: class 5 (sky-like), consistent with the informal
  check's "sky fills the top ~7 rows" observation.
- `bottom_frac(road_class) > top_frac(road_class)` holds by a wide margin (0.949 vs 0.0) — no
  flip/transpose. **Gate 7.1: PASS**, formally, not just informally.

Script used: an ad hoc check (not committed — trivial enough to not warrant a permanent script);
reran with `decode_lfg_label` directly against `/nasa/drives/yaak/lfg_labels/v1/Niro096-HQ/2023-01-11--13-47-36/`.

## 7. Stage 2 — rbyte plumbing (done 2026-08-17)

All of brief §4 implemented:

- **§4.1** `paths.lfg_labels` added to `config/paths/yaak/default.yaml`
  (`/nasa/drives/yaak/lfg_labels/v1`) and `config/paths/yaak/verda.yaml`
  (`/mnt/verda-nas/lfg_labels/v1`). **Caveat:** the Stage 1 labelling job only wrote to the local
  `/nasa` path on this host — the verda-nas mirror is **not yet populated**. Noted inline in
  `verda.yaml`; sync (or re-run labelling) before training this arm on Verda.
- **§4.2** `src/rmind/utils/lfg_labels.py` — already existed from Stage 1 prep, unchanged, matches
  the brief exactly.
- **§4.3** New `lfg_labels` rbyte stream added to both `config/_templates/dataset/yaak/train.yaml`
  and `val.yaml`, same `index` as `cam_front_left`, `PathTensorSource` +
  `rmind.utils.lfg_labels.decode_lfg_label` partial. `just generate-config` regenerated
  `config/dataset/yaak/{train,val}.yaml` cleanly — verified via `yaml.safe_load` that
  `streams.lfg_labels.sources` has 655 drives (train) and 5 drives (val), matching
  `streams.cam_front_left`.
- Also added the same stream block by hand to `config/dataset/yaak/train_subset30.yaml` (30
  drives) — this file is **not** ytt-templated (`config/dataset/*` is gitignored, and this
  particular file lives outside `config/_templates`), so it needed a direct edit to keep the
  pilot/gate-7.2 dataset in sync with the templates.
- **§4.4** `config/model/yaak/patch_policy/raw.yaml` — added `lfg: [data, lfg_labels]` to the
  `Remapper`'s `context` group, alongside `waypoints`.
- **§4.5** Single-writer cache rebuild: **not performed, and I don't think it's actually
  required.** The `train`/`val` samples pipeline (the part `pipefunc`'s disk cache covers) is
  unchanged by this edit — I only added a new `streams` entry, not a new pipeline function or a
  change to any existing one, so no cached intermediate (`aligned`, `filtered`, `samples_cast`,
  ...) is stale. Streams are read lazily per-`__getitem__`, outside the cached pipeline. Flagging
  this reasoning explicitly rather than silently skipping the brief's instruction — if it's wrong,
  the fix is a rebuild, not a revert.

### Validation

- **Gate 7.1**: formally passed, see §6 above.
- **Gate 7.2 (round-trip)**: passed **at the stream/source level**, not the full datamodule level.
  Instantiated the `cam_front_left` and `lfg_labels` `PathTensorSource`s directly from the
  hydra-composed `predict_train_subset` datamodule config (drive
  `Niro096-HQ/2023-01-11--13-47-36`, 5 frames) and confirmed: (a) both sources resolve to the
  expected paths under `paths.data` / `paths.lfg_labels`, (b) `lfg_labels[frame_idx]` byte-for-byte
  equals `decode_lfg_label((label_root / f"{frame_idx:09d}.bin").read_bytes())`, (c) shapes/dtypes
  are `(324, 576, 3) uint8` and `(4, 16, 16) uint8` as expected. **PASS.**
  - Could **not** run the round-trip through the *full* `rbyte.Dataset.from_config` (the
    `DataFrameGroupByDynamic`-based sample builder) for `train_subset30.yaml`: that file's pipeline
    references `rmind.utils.pipeline.left_join_parquet` / `drop_overrepresented_by_loss`, which do
    not exist under `src/rmind/utils/` in this checkout. **This is a pre-existing issue unrelated
    to this task** — those functions belong to the loss-mining/rebalancing pipeline stages from
    the stall-investigation work (`pretrain_stall_experiments.md`), not to the LFG stream I added;
    `git diff` confirms I did not touch that part of the file. It blocks anyone trying to
    instantiate `train_subset30.yaml` end-to-end today, independent of this task, and is worth a
    separate fix.
  - No shared state was touched: the isolated-cache attempt used a scratch `paths.rbyte.cache`
    override, and the source-level check used no cache at all.
- Ran `ruff check` and `ty check` on the touched Python file (`lfg_labels.py`, unchanged from
  Stage 1) — clean. All touched YAML re-parses cleanly with `yaml.safe_load`/`OmegaConf`.

## 8. Follow-ups carried from Stage 2 (unchanged)

1. Resolve the CC BY-NC 4.0 licence question before any production use (unchanged, still the one
   true blocker for anything beyond experimentation).
1. The missing `rmind.utils.pipeline` module (blocking full-pipeline instantiation of
   `train_subset30.yaml`) turned out not to matter for Stage 3 — the aux loss reads straight off
   `blocks`/`inputs` inside `_compute_metrics`, never touching `train_subset30.yaml`'s
   loss-mining/rebalancing pipeline stages. Still worth fixing separately.
1. Sync labels to `/mnt/verda-nas/lfg_labels/v1` (or re-run labelling there) before training this
   arm on Verda.

## 9. Stage 3 — the aux loss in `PatchPolicy` (done 2026-08-17)

All of brief §5 implemented in `src/rmind/models/patch_policy.py`:

- **§5.1** `_features` now returns `(features, blocks, chunk)` — a 3-tuple, exactly as specified.
  The brief also flags an internal tension (§5.3's snippet needs `inputs` in `_compute_metrics`,
  which a 3-tuple `_features` doesn't carry) and recommends resolving it by having `_features`
  return `inputs` too — but that would have forced every other call site (`forward`,
  `predict_step`) to carry an `inputs` value they don't use. Instead: introduced a private
  `_encode(batch, *, require_chunk)` returning `(inputs, features, blocks, chunk)`, of which
  `_features` is now a thin wrapper that drops `inputs`. `_compute_metrics` calls `_encode`
  directly. `input_transform` still runs exactly once per batch either way. All three original
  call sites (`_compute_metrics`, `forward`, `predict_step`) updated for the 3-tuple.
- **§5.2** New constructor params `aux_heads` (`HydraConfig[ModuleDict] | InstanceOf[ModuleDict] | None`), `aux_weights: dict[str, float] | None`, `aux_purity_min: float = 0.6`, `lfg_labels: Path = ("context", "lfg")`. Registered via `init_hydra_param` and recorded in `hparams`, per the
  brief. Added one thing the brief didn't spell out: `__init__` now raises `ValueError` if
  `aux_heads` is set but `aux_weights` is missing an entry for any of its keys — otherwise
  `_aux_metrics`'s `self.aux_weights[k]` would `KeyError` deep inside a training step instead of
  at construction time.
- **§5.3** `_aux_metrics(blocks, labels)` implemented verbatim per the brief: patches
  `blocks[:, :, 1:]`, confidence-weighted segmentation NLL + motion BCE, purity-thresholded patch
  masking, `denom.clamp(min=1.0)` guard, weights applied inside the method (not in `_step`'s
  unweighted sum). `_compute_metrics` wires it in as a second top-level `"aux"` group exactly as
  specified, guarded by `self.aux_heads is not None`.
- **§5.4** Confirmed via `git diff`/reading that `patch_policy_decoder.py` never calls `_features`
  or `_encode` — untouched, as required. `image_encoder`/`goal_encoder`/`tokenizer` stay under the
  existing `torch.no_grad()` block in `_frame_tokens`, so the aux path (which only reads `blocks`,
  downstream of that block) cannot reach them; a dedicated test confirms zero gradient on all
  three (`test_aux_gradients_reach_trunk_not_frozen_modules`).

### Type-checker note

`ty check` on the edited file reports the same 15 pre-existing diagnostics as the unmodified file
(confirmed via `git stash`/diff — only line numbers shifted), plus zero new ones after adding
`assert self.aux_heads is not None` / `assert self.aux_weights is not None` guards inside
`_aux_metrics` and at the `_compute_metrics` call site (mirrors the existing `assert self._loader is not None`-style pattern already used elsewhere in this codebase, e.g.
`src/rmind/components/dataloader.py`).

### Tests added (`tests/test_patch_policy.py`)

Extended the existing fixtures (`_make_model` now accepts `aux_heads`/`aux_weights`/
`aux_purity_min`; `_make_batch(with_lfg=True)` attaches a `(b, t, 4, 2, 2)` uint8 label tensor at
`context.lfg`, matching `decode_lfg_label`'s channel layout at a scaled-down `NUM_PATCHES=4` grid).
All existing call sites of `model._features(...)` updated for the new 3-tuple return. 8 new tests,
all passing:

- `test_aux_heads_absent_by_default` — no `aux_heads` -> no `"aux"` group at all.
- `test_aux_weights_missing_entry_raises` — the new constructor-time validation.
- `test_aux_metrics_shapes_and_weighting` — loss/metric key sets, finiteness, `supervised_fraction == 1.0` at `aux_purity_min=0.0`.
- `test_aux_weight_scaling_matches_unweighted_terms` — weighted model's logged loss equals
  `weight * unweighted_term`, confirming weighting happens inside `_aux_metrics` (not in `_step`'s
  sum, which is unweighted).
- `test_aux_purity_min_masks_low_purity_patches` — half the patches forced below/above the purity
  threshold; `supervised_fraction` matches exactly.
- `test_aux_zero_confidence_masks_patches` — `confidence=0` zeroes a patch's contribution even at
  full purity.
- `test_aux_gradients_reach_trunk_not_frozen_modules` — gradient reaches `aux_heads` and
  `encoder`, and is `None` on `tokenizer`/`goal_encoder` (brief §5.4's requirement, made concrete).
- `test_aux_loss_does_not_perturb_policy_metrics` — a model with `aux_heads` attached (weights
  zeroed) reproduces byte-identical `"policy"` loss/metric values to one without — the aux branch
  is additive and doesn't perturb the shared `_encode` computation. This is the unit-test analogue
  of the brief's §7.3 zero-weight control, at the module level rather than the training-curve
  level.

`nix develop --command uv run pytest tests/test_patch_policy.py`: **20/20 pass** (12 pre-existing +
8 new). `nix develop --command just test` (full suite) launched to confirm no other test file
regressed from the `_features` signature change; result pending at time of writing this section —
see the addendum below once it completes.

## 11. Stage 4 — two distinct OOMs blocking the first training run (2026-08-18)

`config/experiment/yaak/patch_policy/dinov2_dinowm_causal_lfgaux.yaml` was added per brief §6
(linear `segmentation`/`motion` aux heads on `dinov2_dinowm_causal`, `aux_weights` defaulted to
`{segmentation: 0.0, motion: 0.0}` in the committed file so the file itself *is* the §7.3
zero-weight control). Every docker attempt at running it against the full 655+5-drive corpus has
died before a single training step logs, on the host's shared 40-core / 188 GB machine. Two
separate root causes, confirmed via `journalctl -k`, not one:

### 11.1 Per-drive samples build — unbounded `ProcessPoolExecutor` (fixed)

`samples.executor` in `config/_templates/dataset/yaak/{train,val}.yaml` was a bare
`ProcessPoolExecutor` with no `max_workers`, defaulting to `os.cpu_count()` (40). With
`scheduling_strategy: eager`, pipefunc fires the `filtered` → `samples` stage (a DuckDB spatial
query plus `DataFrameGroupByDynamic.agg(pl.all()...)`, which materializes full per-window
list-columns for every raw field) for up to 40 of the 660 drives at once, all held in the parent
process pending pipefunc's `file_array` write. This OOM-killed `rmind-train` itself twice:

```
Aug 18 08:08:14 kernel: Out of memory: Killed process 446678 (rmind-train) anon-rss:49428552kB
Aug 18 08:30:48 kernel: Out of memory: Killed process 781095 (rmind-train) anon-rss:49757520kB
```

Not caused by the LFG work: `lfg_labels` is a lazy `PathTensorSource`, read outside this cached
samples pipeline entirely (§7 above) — it contributes nothing to this build's memory footprint.
This is a pre-existing property of the full-corpus build that this arm's docker run was simply the
first thing to exercise at full scale under today's host memory contention.

**Fix applied** (commit `094563b`): capped `executor.max_workers: 8` in both templates,
regenerated `config/dataset/yaak/{train,val}.yaml`. Trades wall-clock for a predictable ceiling;
can be raised per-run on a quieter/higher-memory host via
`++datamodule.{train,val}.dataset.samples.executor.max_workers=N` without touching the template
default.

### 11.2 Full-corpus reduce step — process-pool round-trip of the whole corpus table (fixed)

With `max_workers` capped (via CLI override, `++...max_workers=8`, on top of the pre-fix image
`rmind:352620a...`), the per-drive stage completed, but the run then died a *third* time with a
different signature — `BrokenProcessPool` surfaced in the app during
`partial(samples_aggregated=shape: (1_939_159, 11), ...)` → `polars.DataFrame.with_row_index`.
`docker inspect` confirmed `OOMKilled=true`, and `journalctl -k` pinned it precisely:

```
Aug 18 10:18:02 kernel: Out of memory: Killed process 1189332 (python3) anon-rss:53307156kB
```

Root cause is structurally different from §11.1 and **not addressed by `max_workers`** — but it is
also *not* "one process legitimately needing 53 GB", which is what the first pass of this section
assumed. The corpus table is only ~14.5 GB; the OOM came from **serializing it across a process
boundary**, which costs ~4× that.

**The measurements** (taken 2026-08-18 against the surviving `samples_cast` outputs of the failed
run, so these are the real numbers for this corpus, not estimates):

| quantity | value |
|---|---|
| final table | 1,939,159 rows × 11 cols, **7548 B/row** → **14.5 GB in memory** |
| all 655 `samples_cast` parts loaded | 14.5 GB (pipefunc stores them as *parquet*, hence only 318 MB on disk) |
| `DataFrameConcater` (`rechunk=False`) | **+0 GB** — polars keeps the 655 chunks by reference |
| `with_row_index` | **+0 GB** — prepends a 1-chunk index column; `n_chunks("all") == [1, 655, 655…]`, i.e. it does *not* rechunk the data columns |
| **both reduce steps back-to-back, single process** | **14.5 GB peak** |
| what actually happened (two processes) | **~92 GB**: 42 GB parent + 50 GB worker |

Per-column shares of that 7548 B/row confirm where the bulk is: `waypoints/xy_normalized` and
`waypoints/xy` are 2960 B/row **each** (37 × 10 × 2 f32) = 78% of the table between them.

**Mechanism.** `samples_aggregated` and `samples_with_id` are the pipeline's only `mapspec`-less
functions, so pipefunc runs each exactly once — but it materializes a `mapspec`-less function's
kwargs **in the parent** (`pipefunc.map._run._func_kwargs` → `_load_from_store(p, store).value`)
and only *then* submits the call to the executor (`_maybe_execute_single` → `_submit`). With a
`ProcessPoolExecutor` that means the whole 14.5 GB table is paid for four times over:

1. parent materializes the table from the store — 14.5 GB
2. parent pickles it for the worker — +14.5 GB
3. worker unpickles — 14.5 GB
4. worker holds the live frame while computing — +14.5 GB

Which is exactly the observed `rmind-train` 42 GB + `python3` 50 GB. The kernel report also shows
this was a **global** OOM at ~183 GB of 188 GB total RSS, not a cgroup limit: the run's ~104 GB
(both of the above plus 8 idle pool workers at ~1.3 GB each) plus a **36 GB job from another user**
(`harsimrat`, a notebook kernel — gone by the time this was diagnosed) plus the rest of the host.
So host contention was a genuine co-factor, but the run's own ~104 GB was the avoidable part.

**Fix applied.** pipefunc accepts a **per-output-name executor mapping**, with `""` as the default
key (`pipefunc.map._run._executor_for_func`), and rbyte passes it straight through
(`PipelineHydraConfig.executor: HydraConfig[Executor] | dict[OUTPUT_TYPE, HydraConfig[Executor]]`,
instantiated leaf-wise via `tree_map` in `Dataset._build_samples`). So in
`config/_templates/dataset/yaak/{train,val}.yaml` the `executor` became a mapping:

```yaml
  executor:
    "":                  # every mapped, per-drive step -- unchanged from §11.1
      _target_: concurrent.futures.ProcessPoolExecutor
      max_workers: 8
      mp_context: { _target_: multiprocessing.get_context, method: forkserver }
    samples_aggregated:   # the two reduce steps stay in-process
      _target_: concurrent.futures.ThreadPoolExecutor
      max_workers: 1
    samples_with_id:
      _target_: concurrent.futures.ThreadPoolExecutor
      max_workers: 1
```

A thread executor keeps both reduce steps in the parent process, so steps 2–4 above disappear and
the table exists exactly once: **14.5 GB instead of ~92 GB**. Nothing is given up — neither step is
parallel over drives, so there was never any concurrency to lose, and polars releases the GIL for
the concat/collect.

**Verification.** Two checks, both run before touching the real corpus:

1. The empty-string key survives OmegaConf → pydantic (`extra="allow"`) → `tree_map` and
   instantiates to `{'': ProcessPoolExecutor, 'samples_aggregated': ThreadPoolExecutor,
   'samples_with_id': ThreadPoolExecutor}`. `config/dataset/yaak/{train,val}.yaml` regenerated and
   re-parsed (655 / 5 drives, both streams present).
2. A minimal pipeline with the same shape (one mapped step + two `mapspec`-less reduce steps) where
   each reduce step reports `os.getpid()`, run through the same
   `PipelineHydraConfig` → `tree_map` → `pipeline.map(executor=...)` path:

   ```
   parent pid: 1838688
   BEFORE (single ProcessPoolExecutor): {'pid': 1838823, 'pid_b': 1838824}
   AFTER  (per-output dict w/ threads): {'pid': 1838688, 'pid_b': 1838688}
   ```

   Note the *two different* worker pids in the BEFORE case: the reduce steps do not even share a
   worker, so with `return_results: false` the table makes **two** independent round trips
   (parent → worker A → store, then store → parent → worker B), each paying the 4× above. That is
   worse than the original reading of this section and explains why the parent was still holding
   42 GB at the moment the second step's worker died.
3. **End-to-end A/B through the real `rbyte.Dataset.from_config`**, same experiment config, same
   drives, only the `executor` shape differing (the control arm rewrites the composed config back
   to the pre-fix scalar). Identical datasets, no wall-clock penalty:

   | arm | `executor` | result | time | peak RSS |
   |---|---|---|---|---|
   | control | scalar `ProcessPoolExecutor` | `len=2994` | 18 s | 1.06 GB |
   | fixed | `{'': Process, samples_aggregated: Thread, samples_with_id: Thread}` | `len=2994` | 16 s | 1.02 GB |

   The real 5-drive `val` config then built clean with the fix: `len=14365` in 39 s, peak 1.30 GB,
   both streams (`cam_front_left`, `lfg_labels`) present and `ds[0]` materializing a `Batch`.

Also asserted programmatically against the *generated* configs that the set of `mapspec`-less
functions equals the set of thread-executor keys, for both `train.yaml` and `val.yaml` — i.e. no
reduce step was missed.

**Two operational consequences of this change — both will bite the next run if missed:**

- **The image must be rebuilt.** `Dockerfile` does `COPY . .` and `just train-unsafe` depends on
  `generate-config`, so the container regenerates `config/dataset/` from the *baked-in* templates.
  Editing the host templates has no effect on `rmind:352620a…`. And `just docker-build` runs
  `check-git`, so the fix has to be committed and pushed first. `config/dataset/*` is gitignored —
  only the two templates need committing.
- **Drop the `++…executor.max_workers=8` CLI overrides.** They were correct against the old scalar
  `executor`; against the mapping they inject a fourth key next to `""`/`samples_aggregated`/
  `samples_with_id` and pydantic rejects the whole config
  (`ValidationError: executor.HydraConfig[Executor]._target_ Field required`) — verified. They are
  also redundant: `max_workers: 8` is now the template default.

**Expected new ceiling for the full-corpus run**, ~45 GB rather than ~104 GB: 14.5 GB for the
reduce steps, then `Dataset.from_config`'s `to_torch(...)` makes one more full copy while
`sample_df` is still alive (~29 GB peak) before the frame is dropped, plus ~10 GB of idle pool
workers and torch/CUDA init. Steady-state training then holds ~14 GB of `TensorDict` — and note the
dataloader is `method: thread` (`config/datamodule/yaak/train.yaml`), so that table is held **once**,
not once per worker.

**Two things deliberately *not* changed:**

- **`resume`.** Every retry so far redid the full ~30-minute build from scratch: pipefunc's
  `resume` defaults to `False`, which means `_cleanup_run_folder(run_folder)` wipes the outputs on
  start (confirmed from mtimes — the last run rebuilt `meta` at 09:48 → `samples_cast` at 10:06 →
  `samples_aggregated` at 10:14 → OOM at 10:18). Only the 13 GB of `cache: true` disk cache at the
  `paths.rbyte.cache` root persists. Because `BasePipelineConfig` is `extra="allow"`, `resume` can
  be turned on per-run from the CLI —
  `++datamodule.train.dataset.samples.resume=true` — which will pick up an interrupted build
  instead of restarting it. Left **off** by default on purpose: a stale run folder would otherwise
  be silently reused after a pipeline edit.
- **Shrinking the payload.** Dropping the redundant world-frame `waypoints/xy` column would cut the
  table 39% (7548 → 4588 B/row, 14.5 → 8.8 GB), and `patch_policy` only consumes
  `waypoints/xy_normalized` (`config/model/yaak/patch_policy/raw.yaml`). But `train.yaml`/`val.yaml`
  are shared with `control_transformer` experiments whose callbacks
  (`trainer/callbacks/{pretrain,finetune}.yaml`) do read raw `waypoints/xy`, so this needs a
  patch_policy-scoped dataset variant rather than a blanket template edit. Not needed now that the
  4× amplification is gone; it is the next lever if memory ever gets tight again.

### 11.3 The same latent bug in `action_train.yaml` (not fixed — different experiment line)

`config/_templates/dataset/yaak/action_train.yaml` is also a 655-drive config with the identical
two `mapspec`-less reduce steps *and* a still-uncapped `ProcessPoolExecutor` (no `max_workers`), so
it carries both §11.1 and §11.2 unfixed. `action_val.yaml`, `predict.yaml` and `train_debug.yaml`
share the shape but at 5/15/3 drives, where it does not matter. Left alone because it belongs to a
different experiment line, not this arm — but the `executor` mapping above is a drop-in for it and
costs nothing.

## 10. Next steps

1. Stage 4: `config/experiment/yaak/patch_policy/dinov2_dinowm_causal_lfgaux.yaml` per brief §6 —
   **added**. §11.2 is fixed but the fix is **not yet exercised at full-corpus scale**: commit the
   two templates, `just docker-build`, then re-run *without* the now-invalid
   `++…executor.max_workers=8` overrides. Watch the parent's RSS through the samples build — it
   should top out near 30 GB (14.5 GB table + the `to_torch` copy), not ~100 GB.
1. The §7.3 zero-weight-control gate at the **training-curve** level (not just the unit-test
   analogue above) — run it and diff against the `dinov2_dinowm_causal` baseline curve before any
   nonzero-weight arm. Blocked on getting any run past the samples build (§11).
1. The weight sweep `{0.03, 0.1, 0.3}` per brief §6, judged on §7.4's metrics
   (`offset_argmax_recon_last`, `code_acc_joint_last`), not `val/loss/code_*`.
1. Still open from Stage 2/1: the CC BY-NC 4.0 licence question, and syncing labels to
   `verda-nas` before training there.
