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

## Decision rules (pre-registered; amended after Phase A; RE-AMENDED after exp12)
- Primary: held-out spike steering chunk-L1 (meanK readout, SEEDED eval).
- exp12 BROKE the paired-design assumption: delta-off paired delta was -0.028
  at seed 1001 but +0.004 at seed 1002 — lever x seed interaction >> the 0.003
  paired noise measured on lr-nulls. Paired deltas from ONE seed are NOT
  trustworthy for stack decisions.
- CURRENT RULE: judge levers by the mean over >=2 seeds vs the control seed
  panel mean (exp01-03 + exp02: mean ~0.337, SEM ~0.007). Stack iff two-seed
  lever mean beats control mean by > 0.015. Single-seed results only triage
  what earns a second seed (promising if paired delta > 0.010).
- Guardrails unchanged: flat L1, aggregate L1, gas/brake L1.
- Stack-on-win (two-seed confirmed); replicate each new stack once.

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
- exp counter: 16 (Phase C in flight; Phase D ranker code next)
- verdicts: 01-03 sigma; 04-05 lr null; 06-07 LDS keep@0.5 (gas protection);
  08 delta-off "-0.028" DID NOT REPLICATE (see 12); 09 steer-only-LDS drop;
  10 delta=50 negative (dose-response 0->0.327, 10->0.356, 50->0.365);
  11 beta-time ambiguous-drop; 12 delta-off@1002 NULL (0.3276 vs 0.3237;
  two-seed mean 0.3275 vs ctrl ~0.337 -> below bar; side-finding: delta-off
  has near-zero seed variance — delta loss = gradient-noise source);
  13 delta-off+lds[1.0,0.5] NEGATIVE (gas 0.0942, steer 0.3327 — both worse)
- PHASE B CLOSED: nothing stacks. Recipe = control (LDS@0.5 already in it).
- in flight: exp14 depth2@1001 (5zlun6y0, trained, evaling — NB eval/predict
  need model.objective.decoder.num_layers=2 override to load the ckpt!),
  exp15 depth8@1001 (1z6154xi, training; same: num_layers=8 at eval)
- control panel (spike-meanK): 0.3555@1001, 0.3237@1002, 0.3289, 0.3387 ->
  mean ~0.337, sigma 0.014. Lever verdicts vs this mean, two seeds, bar 0.015.
- Phase C queue: width {128,512}, heads {2,8}, hidden mult 4, cond-AdaLN.
  Phase D (ranker, MANDATORY) — write code during training windows.
