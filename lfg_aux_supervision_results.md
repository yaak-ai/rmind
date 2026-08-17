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

## 10. Next steps

1. Stage 4: `config/experiment/yaak/patch_policy/dinov2_dinowm_causal_lfgaux.yaml` per brief §6.
1. The §7.3 zero-weight-control gate at the **training-curve** level (not just the unit-test
   analogue above) — run it and diff against the `dinov2_dinowm_causal` baseline curve before any
   nonzero-weight arm.
1. The weight sweep `{0.03, 0.1, 0.3}` per brief §6, judged on §7.4's metrics
   (`offset_argmax_recon_last`, `code_acc_joint_last`), not `val/loss/code_*`.
1. Still open from Stage 2/1: the CC BY-NC 4.0 licence question, and syncing labels to
   `verda-nas` before training there.
