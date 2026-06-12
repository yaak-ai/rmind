# Autonomous Experiment Loop — Operating Instructions

Agreed with Nikita 2026-06-12. This file is the durable contract for the loop;
re-read it whenever context is unclear. Decisions settled: **stack-on-win**
(winners merge into the running best recipe; one replicate of each new stack),
**learned ranker (Phase D) runs regardless** of other phases' outcomes,
**keep going past 30** experiments if budget remains.

## Mission
>= 30 experiments improving the flow action expert, prioritizing held-out
maneuver accuracy. Baseline to beat: spike steering chunk-L1 **0.326**
(cached control 7c3dh7bs, meanK readout, 5 val drives). Known headroom:
oracle-mode 0.28, coverage floor 0.15. Full freedom to redirect the tree.

## Resources & constraints
- **GPU 0 only.** GPU 1 untouched. 2-3 concurrent jobs.
- Cached track (`artifacts/feature_cache/pilot/`) is the engine (~25 min/run).
  New dataset caches allowed (~40 min each).
- wandb project `action-flow`; run name `exp<NN>_<slug>`; lever tags
  (configs auto-tag) + `wandb.notes` = motivation/hypothesis/decision-rule.
- Train entry: `just train experiment=yaak/control_transformer/<exp>
  seed=N +wandb.name=exp<NN>_<slug> [overrides]` — NB just re-expands args
  through sh: NO parens/quotes in args; wandb.name needs the + prefix; set
  run NOTES post-launch via the wandb API (run.notes=...; run.update()).
  Launch each run as its own harness background task (nohup+& dies with the
  tool call).

## Per-experiment protocol
1. Launch (name/tags/notes).
2. Eval: cached `flow_meank_eval` (`+meank.cache=artifacts/feature_cache/pilot/val.pt
   +meank.cached_artifact=yaak/action-flow/model-<run>:v4 +meank.k=32`)
   -> panel: aggregate single/meanK L1, flat L1, spike L1 (single/meanK/mode),
   steering corr, gas/brake L1, bimodal %, selection regret, coverage floor.
3. Artifacts: `flow_cached_predict` parquet (mode readout; deterministic for
   regression) into `reports/parquets/exp<NN>_<slug>.parquet`; append run to
   `reports/loop_manifest.json`; rebuild `reports/loop_report.html` via
   `flow_report.py` (approved format).
4. Notion: one Experiments-DB row per run (Result enum, metrics, verdict,
   next-step pointer). Data source: collection://c8d2efb3-3cd8-45b4-9b30-55bb22b42865,
   Task relation -> https://app.notion.com/p/30bd658ccf87809c8941ca940bcca709.
5. Decide (rules below), record, launch next.

## Decision rules (pre-registered; amended after Phase A)
- Primary: held-out spike steering chunk-L1 (meanK readout, SEEDED eval).
- PAIRED DESIGN (Phase A finding): cross-seed sigma=0.014 but same-seed paired
  deltas are ~0.003 (5x tighter). All levers run at seed 1001 and compare
  against exp01 (control@1001 = 0.3555). Keep iff paired delta > 0.010
  (~3x paired noise); wins confirmed at a second seed before stacking.
- Guardrails: flat L1, aggregate L1, gas/brake L1 (no paired regression >0.010
  equivalent scale).
- Stack-on-win (after second-seed confirm); replicate each new stack once.

## Experiment tree (~35 slots, adaptive)
- A. Calibration (4): 3 control seed replicates (-> sigma); lr 0.5x / 2x.
- B. Loss-side (10-12): LDS alpha {0, 0.25, 1.0}; per-channel LDS
  (steering-only); LDS cap; chunk-delta weight {0, 1, 50}; histdrop p
  {0.25, 0.8}; beta-vs-logit flow-time (matched); logit_mean skew;
  waypoint-PE (1 slot).
- C. Decoder architecture (6-8): depth {2, 6, 8}; width {128, 512}; heads
  {2, 8}; hidden multiplier; condition handling variants (cond LayerNorm /
  projection / AdaLN-on-condition).
- D. Selection — the prize (5-6, MANDATORY): learned ranker on cached data
  (cond + K frozen-flow draws, GT-distance labels); regression /
  classification / contrastive objectives; plug into decomposition as
  selector; target oracle 0.28 -> coverage 0.15. ~1h new code.
- E. Dataset (3-4): stratified-30 pilot cache (fix 20-drives-one-car bias);
  re-run control + winners; optional 60-drive cache.
- F. Finale (2-3): stacked winner + replicate; ONE full-path confirmation
  overnight (~7.6h) — cached conclusions need a full-path anchor.

## Deliverables
1. `reports/loop_report.html` — all runs, cumulative (approved format).
2. Notion: row per run; PHASE-SUMMARY dev-log entries + final synthesis.
3. wandb: all runs named/tagged/noted.
4. Git commits for tooling/configs (no pushes).
5. Final recommended recipe + full-path confirmation numbers.

## Edge handling
- Diverged/NaN -> kill, tag crashed-early, note cause, continue.
- Tree-invalidating result -> rewrite remaining plan, log reasoning.
- OOM/contention -> serialize, never touch GPU 1.
- Never: push remote, modify GPU 1's run, delete others' files, change the
  val set mid-loop, alter logged results.

## State (update as the loop progresses)
- exp counter: 08 (01-03 control reps; 04-05 lr null -> keep 1.6e-4;
  06 lds_off + 07 lds_a025 training)
- current best recipe: finetune_flow_pilot_lds_cached (control)
- sigma: cross-seed 0.014 (controls: .3289/.3555/.3237/.3387); paired ~0.003
- paired baseline: exp01 (wvn1a4t6, seed 1001) spike-meanK 0.3555
