# Handoff: finish the LFG `v2` exact-label run

Instructions for whoever (agent or human) picks this up. Read `lfg_relabel_v2_plan.md` first — it
has the measured resource budget, the reasoning behind each choice, and the execution log. This file
is the operational checklist only.

**Background:** `lfg_aux_supervision_results.md` §12 explains why the `v1` label root covers the
wrong frames and how the `v1_nearest` hardlink tree works around it at ~1.5% label cost. `v2`
replaces that approximation with labels computed at the frames the pipeline actually requests.

______________________________________________________________________

## 0. State as of 2026-08-19 ~19:15 UTC

| | |
|---|---|
| label root being built | `/nasa/drives/yaak/lfg_labels/v2` |
| frame list (do not regenerate) | `/nasa/drives/yaak/lfg_labels/required_frames_clip37.json` — 660 drives, 2,154,455 frames |
| pilot job | 35 drives (`train_subset30` + the 5 val drives), 121,193 frames, log in the session scratchpad `pilot2.log`; was 2/35 done, 0 failures, ~81k frames/h |
| full run | **not launched yet** — §3 below |
| competing GPU job | PID 2216274, `dinov2_dinowm_causal_lfgaux` with `aux_weights=0.03`, started 2026-08-18 18:20:49, reads `v1_nearest`. Owner accepted sharing the GPU |
| gates passed | equivalence (byte-identical, §9.2 of the plan), §7.1 spatial on `v2`, drive-level resume |
| gates outstanding | §7.2 round-trip, 1024-byte sweep, `verify`, `v1_nearest`-vs-`v2` agreement |

If the pilot process is gone, just run the §3 full-run command — `--resume-drives` skips whatever
completed and relabels any partial drive from scratch.

## 1. Environment invariants

- **Labelling runs in the LFG venv**, never the rmind venv:
  `/nasa/tools/lfg/.venv/bin/python scripts/lfg_label_drives.py ...`
- **Gate scripts run in the rmind venv via the flake's wrapped uv**:
  `nix develop --command uv run python scripts/lfg_validate_labels.py ...`.
  Plain `python` inside `nix develop` fails on libstdc++ — use `uv run`.
- LFG weights are CC BY-NC 4.0. Unresolved licence question; this is pilot/experimentation work.

## 2. Hard don'ts

1. **Do not repoint `paths.lfg_labels` at `v2`** in `config/paths/yaak/{default,verda}.yaml` while
   PID 2216274 (or any run reading `v1_nearest`) is alive, and not before every gate in §4 passes.
2. **Do not run any gate against `paths.rbyte.cache=.rbyte_cache_causal32_lfgaux`** — the live run
   uses it. pipefunc's `resume=False` default wipes a run folder on start and filter-node outputs
   are keyed by output name only, so a dataset instantiation there can wipe or cross-contaminate
   the live run's cache. Use a throwaway cache dir instead (§4.2).
3. **Do not use `--skip-existing`.** It skips individual frames, which changes which frames share a
   15-frame window and therefore the labels themselves. Use `--resume-drives`.
4. **Do not regenerate `required_frames_clip37.json`** unless the dataset config changed (the
   `filtered` query, `clip_length`, `episode_stride`/`episode_step`, or the drive list). If it did,
   re-extract with `scripts/lfg_required_frames.py extract` and start `v2` over — a stale frame list
   is exactly what broke the first full-corpus attempt.
5. **Do not delete `v1` or `v1_nearest`.** `v1_nearest` is hardlinks into `v1`; the live run reads it.

## 3. Launch the full run

Wait for the pilot to finish (or just launch this — it is the same command with the full drive list,
and resume makes the overlap free):

```bash
mkdir -p ~/lfg_v2_logs
cd ~/rmind
nohup /nasa/tools/lfg/.venv/bin/python scripts/lfg_label_drives.py \
    --lfg-repo /nasa/tools/lfg \
    --checkpoint /nasa/tools/lfg/lfg_seg_motion_m3n3.pt \
    --output-root /nasa/drives/yaak/lfg_labels/v2 \
    --frames-from /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \
    --resume-drives \
    > ~/lfg_v2_logs/full.log 2>&1 &
```

Defaults are the intended settings: `--gap-split 20`, `--loader-threads 4`, labels-only forward.
Omitting `--drive`/`--drives-from` labels all 660 drives in the JSON.

**Expect ~26 h** while sharing the GPU with PID 2216274, ~11–12 h if that run ends
(0.63–0.74 s/window contended vs 0.28 uncontended). Progress lines carry a running `k frames/h`, so
ETA = `(2,154,455 − frames_done) / rate`.

Monitoring — per-drive completions and anything that looks like a failure:

```bash
grep -E "drives, .*elapsed" ~/lfg_v2_logs/full.log | tail -3
grep -E "FAILED|Traceback|out of memory|Killed" ~/lfg_v2_logs/full.log
cat /nasa/drives/yaak/lfg_labels/v2/failures.json 2>/dev/null   # absent means zero failures
```

A drive that raises is recorded in `failures.json` and the run continues. Re-running the same command
retries only the drives that are not already complete. Investigate any drive that fails twice —
the likely cause is a missing JPEG, which the script now reports per drive up front.

## 4. Gates — all four must pass before `v2` is used

### 4.1 Coverage and integrity

```bash
cd ~/rmind
nix develop --command uv run python scripts/lfg_required_frames.py verify \
    --required /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \
    --label-root /nasa/drives/yaak/lfg_labels/v2          # must exit 0

# every blob exactly 1024 bytes (no header, no truncation)
find /nasa/drives/yaak/lfg_labels/v2 -name '*.bin' ! -size -1024c ! -size +1024c | wc -l  # == 2154455
find /nasa/drives/yaak/lfg_labels/v2 -name '*.bin' \( -size -1024c -o -size +1024c \) | head  # empty
```

### 4.2 §7.2 dataloader round-trip (isolated cache — see §2.2)

```bash
nix develop --command uv run python scripts/lfg_validate_labels.py roundtrip \
    --label-root /nasa/drives/yaak/lfg_labels/v2 \
    2>&1 | tail -20
```

This composes `train.yaml` with the `dinov2_dinowm_causal_lfgaux` experiment, instantiates
`datamodule.val.dataset`, and asserts the delivered `(37, 4, 16, 16)` uint8 tensor equals
`decode_lfg_label` of the on-disk bytes for the first/middle/last sample of **every** val drive.
It defaults to `paths.rbyte.cache=.rbyte_cache_lfg_gate` (override with `--rbyte-cache`) so it can
never touch a live run's cache; the first invocation therefore builds val samples for 5 drives, which
takes a few minutes. **This subcommand has not been run yet** — it is the one piece of new code
without a successful execution behind it, so expect to debug it (most likely the `input_id` /
`frame_idx` key paths in `_sample_drive_id` and `cmd_roundtrip`). The original v1 check only fetched
`ds[0]`, which is why it missed the §12 phase bug on 4 of 5 drives — do not weaken it back to one
sample.

### 4.3 §7.1 spatial alignment

```bash
nix develop --command uv run python scripts/lfg_validate_labels.py spatial \
    --label-root /nasa/drives/yaak/lfg_labels/v2 \
    --drive Niro096-HQ/2023-01-11--13-47-36
```

Already PASS on that drive (road class 0: 92.9% of bottom-row patches vs 0.0% of top-row; sky class
5 at 81.1% of top rows). Re-run on a couple more drives after the full run for breadth.

### 4.4 Divergence from `v1_nearest` (the sanity check on the whole exercise)

```bash
nix develop --command uv run python scripts/lfg_validate_labels.py agree \
    --a /nasa/drives/yaak/lfg_labels/v1_nearest \
    --b /nasa/drives/yaak/lfg_labels/v2 \
    --drives 30 --frames 3000
```

**Expect** per-patch seg agreement ~0.98–0.99 over supervised patches (`purity >= 0.6`,
`confidence > 0`) and motion MAE ~0.01–0.02, reproducing §12.4's per-drive numbers at corpus scale.
Two ways this can be informative:

- Agreement much *lower* (say < 0.95) — something other than the temporal offset changed. Suspect
  `--gap-split` recomposition (expected to move labels near discontinuities, which is the point) or a
  preprocessing drift. Investigate before training on `v2`; do not just accept it.
- Agreement ~1.0 everywhere — then `v1_nearest` was already good enough and `v2` buys nothing, which
  is itself a useful result to write down.

## 5. After the gates pass

1. Repoint `paths.lfg_labels` to `.../v2` in `config/paths/yaak/default.yaml` (and `verda.yaml` once
   the labels are synced there — `verda-nas` is mounted **read-only**, so `v2` must be produced
   locally and copied; `v2` is real files, not hardlinks, so plain `rsync` is correct, but it is
   2.15M small files, so budget hours).
1. Rebuild the training image — it bakes `config/`.
1. Re-run §4.1 `verify` as the launch gate, then relaunch the aux arm on `v2`.
1. Per the experiment config's own gate order, the **zero-weight control** (`aux_weights` both 0.0)
   at the training-curve level still has never run, and it should be diffed against the
   `dinov2_dinowm_causal` baseline before any nonzero-weight arm is judged. The currently running
   0.03 arm is ahead of that gate.
1. Append results to `lfg_relabel_v2_plan.md` §9 and update `lfg_aux_supervision_results.md` §12.5,
   which currently says the exact run is optional and un-started.

## 6. Known residues, deliberately not fixed

- **1,657 frames (0.08%)** sit in windows with <=3 real frames, where tail-padding repeats the last
  frame and the window is near-static — mildly bad for the motion head. The fix, if it ever matters,
  is to pad short windows with real neighbouring JPEGs and write labels only for required frames.
- `config/_templates/dataset/yaak/action_train.yaml` still carries the §11.1/§11.2 OOM bugs
  (uncapped `ProcessPoolExecutor`, `mapspec`-less reduce steps). Different experiment line.
- The `--stride` labelling mode still exists as an exploratory escape hatch and warns that its output
  must not be trained on.
