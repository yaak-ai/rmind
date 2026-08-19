# Plan: exact LFG labels for the required frame set (`v2`)

**Goal.** Replace the `v1_nearest` hardlink tree (§12.4 of `lfg_aux_supervision_results.md`, mean
temporal offset 2.96 frames / 99 ms) with labels computed *at* the frames the training pipeline
actually requests, so the aux-supervision arm trains on exact targets.

**Status of this document:** plan + measured resource budget. Nothing has been launched. All timing
numbers below were measured on this host (single RTX 5090, 32 GB) on 2026-08-18, not extrapolated
from the v1 run.

______________________________________________________________________

## 1. Is this worth doing?

§12.4 already measured the cost of the nearest-label substitution at 1.3–1.9% of supervised patches
(about a third of one sample step) for a `weight: 0.1` auxiliary regularizer. That is the reason
`v1_nearest` was built, and it remains the right call for the coarse seg/motion aux loss.

Do the exact run when one of these holds:

- the geometry heads (brief §9: point maps, poses) get used — a fixed 99 ms offset is a real error
  there, not a rounding one;
- the aux weight sweep shows sensitivity to label noise at the 1–2% level;
- you want the label-noise variable eliminated before interpreting a null result from the sweep.

It is **not** a prerequisite for the §7.3 zero-weight control or the first `{0.03, 0.1, 0.3}` sweep.
Note the scheduling conflict in §6: this job occupies the only GPU for ~11 h, so it competes
directly with those runs.

## 2. Scope (measured, not estimated)

Against `/nasa/drives/yaak/lfg_labels/required_frames_clip37.json` (660 drives, 2,154,455 frames,
generated 2026-08-18 12:54, run folders still present, dataset templates unchanged since):

| | frames | windows (15/window) |
|---|---|---|
| required total | 2,154,455 | 143,936 |
| already exact in `v1` | 122,863 | — |
| missing from `v1` | 2,031,592 | 135,735 |
| **recommended: full relabel** | **2,154,455** | **145,531** (with §4 gap-split) |

**Relabel everything; do not label only the delta.** LFG's `decode()` alternates per-frame attention
with attention across all frames of the window (`Pi3/pi3/models/pi3.py`, `i % 2` → `hidden.reshape(B,
N*hw, -1)`), so **every frame's label depends on the other 14 frames in its window**. Labelling only
the 2,031,592 missing frames builds windows out of the missing-frames list, which is a different
composition from both `v1`'s stride-10 windows and a clean required-set run — the resulting tree
would mix three provenances. The delta-only shortcut saves 6% of the work (≈0.6 GPU-h out of 11);
it is not worth a heterogeneous label tree.

The same finding means **`v2` will not be bit-identical to `v1` even at offset 0**, wherever the
required set has gaps relative to the stride-10 set. §12.4's offset-0 control reproduced `v1` exactly
only because that drive's first 150 required frames happened to be contiguous in `v1`'s target list.

## 3. Measured GPU cost

Per 15-frame window, `n_future_frames_override=0`, bf16 autocast, batch 1:

| variant | ms/window | ms/frame | corpus (145,531 windows) |
|---|---|---|---|
| current script, typical drive (serial load→GPU→write) | 450 | 30.0 | **19.6 h** |
| current script, slow-NFS drive | 900–950 | 60–63 | up to 38 h |
| + threaded prefetch/write, full forward | 322 | 21.5 | 13.0 h |
| **+ threaded prefetch/write, labels-only forward** | **271–293** | **18.0–19.5** | **11.0–11.8 h** |
| pure GPU compute, no I/O (floor) | 263 | 17.5 | 10.6 h |

Stage breakdown of the current script on a cold drive (`Niro102-HQ/2023-01-23--13-13-49`, 40 windows):
**load 108.5 ms + forward/pool/D2H 340.5 ms + write 1.5 ms = 450.5 ms/window** — matching v1's actual
0.485 s/window average (2,719,328 frames in 24.4 h). The 0.9 s/window seen on
`Niro104-HQ/2023-04-06--08-38-02` is NFS read variance, not compute: writes are negligible (0.29 ms
per 1 KiB file on `/nasa`), reads range 108–560 ms/window across drives.

**Batching does not help.** bs=1/2/4 all run at 19.2 ms/frame — 15 frames already saturate the GPU.
The only effect of a larger batch is memory (5.55 → 8.47 GiB peak). Keep batch 1.

### Two optimizations worth implementing first (~2 h of work, saves ~8 GPU-h)

1. **Labels-only forward (1.13×).** The forward computes `point_decoder`, `camera_decoder`,
   `point_head`, `camera_head` and the `points`/`camera_poses` unprojection, none of which the
   labeller reads. Skipping them is **bit-identical** on `segmentation`, `motion` and `conf`
   (verified: `torch.equal` = True, max abs diff 0.0). Prototype:
   `scratchpad/lean.py` → `forward_labels_only()`.
1. **Overlap I/O with compute (1.4–1.7×).** A loader thread (JPEG decode + crop + bicubic resize +
   `pin_memory`) and a writer thread take the 108–560 ms read and the write off the critical path,
   leaving the job GPU-bound. Use **4 loader threads**, not 1: the slowest drives read at ~560 ms
   serially, which would starve a single-threaded prefetch (a 4-way pool brings it to ~140 ms, under
   the 271 ms compute budget).

**Verified equivalence:** the combined prototype (prefetch + writer + labels-only forward) was run
against `scripts/lfg_label_drives.py` on the same 600 frames of the same drive —
**600/600 `.bin` files byte-identical**. So these are pure throughput changes, not a new label
definition.

Deliberately not doing: `torch.compile` / fp16 (marginal on top of a GPU-bound pipeline, and both
perturb the numerics for no benefit), multiple processes per GPU (compute-bound already, and it
doubles the 4.5 GiB weight residency).

## 4. Fix the windowing while relabelling

`chunk_windows()` chunks the flat sorted target list by 15, ignoring temporal discontinuities, so a
window can span a filtered-out region and hand the video model an incoherent sequence. Measured on
the required set: 3,569 gaps larger than 20 frames (median 850, p90 3,080, max 42,634 frames), and
**2.3% of windows (3,296) straddle one**.

Splitting windows at gaps > 20 frames costs **+1,595 windows (+1.1%, ~7 min)** and removes the issue.
Splitting at *every* discontinuity is the wrong trade — the required set is riddled with 1–9 frame
phase shifts (8,522 gaps of 9, 8,343 of 1, …), and per-run chunking would cost +23.8% windows
(178,166) for windows that were temporally fine to begin with.

Record `window_size` and the gap-split threshold in each drive's manifest so the windowing rule is
part of the provenance.

## 5. Resume must be drive-level, not frame-level

`--skip-existing` skips individual frames, which changes the composition of every window after the
skip point — a crash-and-resume would silently produce labels under a different context than a clean
run. For `v2`, resume by **skipping drives whose manifest already covers the full required set for
that drive** and relabelling any incomplete drive from scratch. Per-drive cost is minutes, so the
wasted work on resume is bounded and window determinism is preserved.

## 6. Resource budget

| resource | requirement |
|---|---|
| **GPU time** | **11.0–11.8 h** with §3's two optimizations; 19.6 h with the script as-is (worse on slow-NFS drives) |
| **GPU memory** | 5.8 GiB peak reserved (4.54 GiB weights + 1.0–1.3 GiB activations) at batch 1 — 32 GB card is not a constraint; the two notebook processes currently holding 2.6 GiB can stay |
| **GPU count** | 1 (this box has exactly 1× RTX 5090). Sharding by drive is embarrassingly parallel — N GPUs ⇒ 11.3/N h — but `verda-nas` is mounted **read-only** (`config/paths/yaak/verda.yaml`), so a Verda run would need a writable label volume first |
| **CPU / RAM** | 4 loader threads (~5 ms/frame decode+resize each), < 2 GB RAM |
| **Disk** | 2,154,455 × 1024 B = 2.2 GB logical, ≈ 8.8 GB at 4 KiB NFS blocks, + 2.15M inodes. `/nasa` has 5.5 TB free (92% used). `v1` (2.72M files, ~11 GB) can be dropped once `v2` passes its gates — but only together with `v1_nearest`, whose entries are hardlinks into `v1` and free nothing on their own |
| **Wall clock** | ~12 h GPU-blocking. Sustained full-power run (575 W cap); v1's 24 h run completed without thermal trouble |
| **Sync to Verda** | `v2` is real files, not hardlinks, so plain `rsync` is correct there (no `-H` trap), but it is 2.15M small files — budget hours, and a writable mount |

## 7. Execution plan

1. **Implement §3's optimizations + §4 gap-split + §5 drive-level resume** in
   `scripts/lfg_label_drives.py` (behind flags; keep the existing serial path as the reference).
   Gate: re-run the byte-equivalence check of §3 on ≥600 frames of one drive — any diff means the
   optimization changed the label definition and must be fixed before proceeding.
1. **Pilot on `config/dataset/yaak/train_subset30.yaml`** — 30 drives, 103,415 frames, 6,994 windows,
   **≈33 min** (≈56 min unoptimized). Confirms throughput at scale and exercises the fault isolation.
1. **Full run** into `/nasa/drives/yaak/lfg_labels/v2`:
   ```bash
   /nasa/tools/lfg/.venv/bin/python scripts/lfg_label_drives.py \
       --lfg-repo /nasa/tools/lfg --checkpoint /nasa/tools/lfg/lfg_seg_motion_m3n3.pt \
       --output-root /nasa/drives/yaak/lfg_labels/v2 \
       --frames-from /nasa/drives/yaak/lfg_labels/required_frames_clip37.json \
       --resume-drives
   ```
   Run it detached (`nohup`/`tmux`) with stderr to a log; watch `failures.json`.
1. **Gates, in order:**
   - `lfg_required_frames.py verify --label-root .../v2` exits 0 (full coverage, 2,154,455 frames).
   - Every `.bin` is exactly 1024 bytes (`find -size` sweep, as in §4 of the results doc).
   - §7.1 road/sky row-fraction check on a full `v2` drive.
   - §7.2 round-trip: first/middle/last sample of every val drive, delivered `(37, 4, 16, 16)` tensor
     equals `decode_lfg_label` of the on-disk bytes (555 pairs, as re-run in §12.4).
   - **`v1_nearest` vs `v2` agreement diff** on a few thousand frames: expect ~98–99% per-patch
     dominant-class agreement and motion MAE ~0.01–0.02, i.e. §12.4's numbers reproduced at corpus
     scale. A materially larger divergence means something other than the temporal offset changed and
     should be understood before training on `v2`.
1. **Flip `paths.lfg_labels`** to `v2` in `config/paths/yaak/default.yaml` (and `verda.yaml` once
   synced), rebuild the training image (it bakes `config/`), re-run `verify` as the launch gate.

## 8. Open risks

- **The required set is config-coupled** (§12.3 caveat). It was extracted from the `clip37` run
  folders and is only valid for that `filtered` query / `clip_length` / `episode_stride` /
  `episode_step` / drive list. Re-extract and re-`verify` after any dataset-config change — a `v2`
  built against a stale JSON fails in the dataloader exactly the way `v1` did.
- **Licence.** LFG weights are CC BY-NC 4.0; unresolved, and a second labelling pass does not change
  that.
- **`v2` supersedes nothing until the gates pass.** Keep `v1_nearest` in place and the config
  pointing at it until §7's gate list is green.

______________________________________________________________________

## 9. Execution log

### 9.1 Script changes (done, 2026-08-18)

`scripts/lfg_label_drives.py`:

- `forward_labels_only()` — skips the point/camera decoders and heads.
- `--loader-threads` (default 4) + `LabelWriter` background writer thread.
- `--gap-split` (default 20 frames) in `chunk_windows()`; windows are now returned unpadded and
  padded at load time, which also fixes the old `n_real` bookkeeping.
- `--resume-drives` + `drive_is_complete()` — drive-level resume that refuses to reuse a manifest
  written under different windowing/forward settings.
- `--drives-from` — drive list from a file.
- Manifest now records `window_size`, `gap_split` and `forward`.

`scripts/lfg_validate_labels.py` (new) — the §7.1/§7.2 gates, which were ad hoc and uncommitted,
plus an `agree` subcommand for root-to-root divergence. All three exit 1 on failure.

### 9.2 Equivalence gate (PASS)

Both on `Niro104-HQ/2023-04-06--08-38-02`, first 600 required frames:

| comparison | result |
|---|---|
| new script `--loader-threads 0 --full-forward --gap-split 0` vs the original script | **600/600 byte-identical** |
| new script defaults minus gap-split (`--gap-split 0`) vs the reference above | **600/600 byte-identical** |
| new script defaults (`--gap-split 20`) vs reference | 275/575 differ — the intended recomposition (these 600 frames contain 2 gaps >20, max 610) |

So the throughput work provably does not change the label definition; only `--gap-split` does, by
design, and the manifest records it.

### 9.3 Degenerate-window check for `--gap-split 20`

Padding a short window repeats its last frame, which makes a very short segment a near-static
sequence — bad for the motion head in particular. Measured over the required set:

| chunking | windows | windows with <=3 real frames | frames affected |
|---|---|---|---|
| flat (v1 behaviour) | 143,936 | 133 | 249 (0.01%) |
| gap-split 20 | 145,531 | 796 | 1,657 (0.08%) |

0.08% of frames in near-static windows is a clearly better trade than 2.3% of windows spanning a
discontinuity of up to 42,634 frames. Not fixed further; if it ever matters, pad short windows with
real neighbouring JPEGs instead of repeating the last frame and write labels only for the required
frames — same 15-frame cost, real temporal context.

### 9.4 GPU contention (important for the schedule)

A training run was already on the card when the pilot started: PID 2216274,
`experiment=yaak/patch_policy/dinov2_dinowm_causal_lfgaux
model.aux_weights.{segmentation,motion}=0.03`, launched 2026-08-18 18:20:49 from the Docker image,
holding 19.1 GB with the GPU at 100%.

Sharing the card costs both jobs roughly 2×: the labeller runs at **0.63–0.74 s/window instead of
0.28**, i.e. **~26 h** for the corpus rather than ~11 h. Run acknowledged and accepted the slowdown.
The throughput figures in §3 were measured while the card was idle and remain the right numbers for
an uncontended run.

Note that run reads `v1_nearest`, so `paths.lfg_labels` must not be repointed at `v2` while it is
alive.

### 9.5 Gates so far

- **§7.1 spatial alignment on `v2`: PASS.** `Niro096-HQ/2023-01-11--13-47-36`, all 2,694
  required-set frames: dominant class in the bottom 2 patch rows is class 0 at 92.9% of patches vs
  0.0% in the top 2 rows; top rows are dominated by class 5 (sky-like) at 81.1%. Consistent with
  v1's 94.9%/0.0% on that drive's stride-10 frame set.
- Drive-level resume verified live: the pilot's relaunch skipped the already-complete drive with
  `[resumed]` and relabelled nothing.

**Caveat for §7.2:** run the round-trip gate against an **isolated** `paths.rbyte.cache`, not the
live run's `.rbyte_cache_causal32_lfgaux`. pipefunc's `resume=False` default wipes a run folder on
start, and filter-node outputs are keyed by output name only — so instantiating a dataset against a
live run's cache can wipe or cross-contaminate it.
