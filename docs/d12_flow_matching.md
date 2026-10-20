# D12 direct-action flow matching

This experiment starts from `palleter/patch-policy` (`d17f8cbe`), which carries
the D12 ingestion and endpoint intention work from
`feat/palletjack-intention-goals`, plus the causal patch trunk and action chunks.
It predicts traction, steering and fork1 commands directly. Pickup and dropoff
positions remain observation tokens; there is no predicted subgoal or planner.
The DINOv3 image encoder remains frozen, as in the existing D12 policies.

## Handover: 2026-10-15 truck test

Companion serving branch: `yaak-ai/coppernic`, `feat/d12-flow-serving`.
The operator has now run the fork-only `ptktynvw` step-35000 model through the
actuation app on `root@172.30.0.169`. Earlier shadow-only status in this document
is historical; supervised control validation is not complete.

The target FP32 TensorRT 10.16.2 engine and required manifest are in
`/data/d12-fm-validation/fork-only-step35000-20261015/`:
`model-orin-trt10.16.2-fp32.engine` and `manifest.json`. The matching archive is
`/nasa/max/d12/flow_matching/2026-10-14-h50-1cam/export-final/validation-d12-169`.
Training reads `data_for_training.mcap`, not raw `data.mcap`; this preserves the
steering-sign correction for the August 25 and September 10 recordings.

Live test recording: `/data/2026-10-15--09-33-42/`. Repeated releases in
`sensor--85.mcap` and `sensor--86.mcap` align with 1.589 s, 7.010 s and 3.201 s
gaps in pallet UWB position. Override drops about one second after the last fix,
matching the stale-input guard. Camera and vehicle-pose updates continue.
Predictions resume when tracking returns, but engagement remains released.
Investigate pallet tracking rather than relaxing freshness limits; add explicit
release-reason logging to the FM setpoint path. See Coppernic's
`crates/copper/cu-infer/D12_FLOW.md` for deployment details and remaining work.

## Flow head provenance

[PR 286](https://github.com/yaak-ai/rmind/pull/286) ports the study's
`fm-toy-verified-v1.1` head (`472e3a7`) byte for byte into Nero. This branch
reuses that exact file and tests its SHA256 and external parity fixtures.
[PR 287](https://github.com/yaak-ai/rmind/pull/287) adds explicit host-drawn
noise and unrolled Euler steps to Nero's serving graph. It does not add D12
serving; this branch implements a separate D12 streaming decoder and exporter.

The [bimodality study](https://app.notion.com/p/3f6d658ccf8781bd8e17c9b9734447dc)
uses a velocity MLP with two 1024-wide ReLU layers, a 64-dimensional sinusoidal
time embedding, Gaussian start noise and uniform flow time. The target is
`action - noise` at the linear interpolant. The trunk runs once per inference.
Four sampled Euler steps score 1.94/2 on BlockPush W1 and 1.55/2 on W10;
10 steps score 1.63/2 on W10. Ant prefers one step (3.17/4), with four steps
at 3.08/4. Four is a robust starting point, not a universal optimum.
Zero-noise starts remove sampling diversity. Long open-loop execution fails
even with flow matching, and validation L1 alone does not measure mode coverage.

## D12 recipe

- Three cameras at 256x144: fork, left-forward and right-forward.
- Six observations spaced six raw frames apart: one second of history at 5 Hz.
- Actions are spaced one raw reference frame apart, approximately 30 Hz.
- H50 predicts 1.67 seconds; `action_horizon=100` predicts 3.33 seconds.
- `near_horizon=6` scores the first 0.2 seconds for 5 Hz replanning.
- Frozen DINOv3-small patches, trainable 8-layer/8-head/512-wide causal trunk.
- Joint flow head predicts all Hx3 values. Action means and population standard
  deviations are fitted on usable training rows only, shared across horizons.
- AdamW, lr 1e-4, 1000 warmup steps, cosine to 35000 optimizer steps,
  bf16, gradient clipping 1.0. Effective batch 8 via accumulation.
- Sampled Gaussian noise, four Euler steps, fp32 integration state.
- Every target chunk is fully observed. No hold padding or action pad mask is
  needed; incomplete windows and windows crossing removed frame rows are excluded.
- All 38 jobs are considered. Whole-job validation holds out the last job from
  each date: August 25, September 10, October 2 and October 12; 34 training jobs.

Height is the recorded observer estimate, not a direct continuous sensor:
`fork/height` in October 12 recordings, `fork/height_retrofit` in October 2
sidecars. The token is `[height_m/2.3, sigma_m/2.3, valid]`. Joins use the most
recent past sample with a 250 ms freshness limit. Missing, stale or invalid
height becomes `[0, 0, 0]`, retaining older recordings without pretending that
the fork is down. The existing discrete height-switch token remains available.
The per-run audit records actual coverage after frame alignment.

## Reproduction

Generate the dataset configs and audit the source data before training:

```sh
uv sync --locked --extra train --group test --no-default-groups
just generate-config
.venv/bin/python -m rmind.scripts.prepare_d12_fm --output /path/to/run
.venv/bin/pytest tests/test_d12_flow_matching.py -q
```

`action_stats.yaml` and `audit.json` are written to the run directory. Pass
their `action_mean` and `action_std` arrays as Hydra overrides to
`experiment=palletjack/patch_policy/d12_fm`. Checkpoints retain these statistics.
Training should write to the run directory on the NAS and log to
`yaak/rmind-palleter`. Preserve the resolved config, source snapshot, audit,
training log and all periodic checkpoints. A run's launch script contains the
exact command. H100 uses the same head and observation settings with the horizon
override; rerun the audit because the set of complete windows changes.

## Streaming export and serving

`python -m rmind.scripts.export_d12_fm --checkpoint model.ckpt --config resolved.yaml --output export/` exports the one-frame decoder. It gates
full-window versus cached inference and ONNX Runtime parity using a held-out
real recording, then writes `manifest.json` and named float32 replay fixtures.
The graph includes ImageNet normalization, four Euler steps and conversion back
to raw action units. Feed unit RGB images, not pre-normalized images. Feed fresh
Gaussian `fm_x0` explicitly; zero noise is not the trained sampling recipe.

Inputs contain three camera frames, speed, the existing switch, ego-relative
pickup/dropoff endpoints, masked height, five frames of KV cache, cache validity
bias, RoPE cosine/sine and noise. Outputs are `[1,H,3]` actions and one frame's
new KV. The host advances cache and computes float64 RoPE before casting to
float32. Reset all state on engage/disengage, manual override, intention changes
and observation discontinuities. Long-running cached inference retains older
context indirectly through KV; six-frame parity does not prove sliding-window
equivalence after eviction.

Build TensorRT on the deployment GPU/runtime, and replay the saved fixtures
through Coppernic's `d12-flow-parity` executable. An RTX 4090/TRT11 engine is a
local validation artifact, **not** a portable Jetson engine. Keep FP32 I/O and
disable TF32 for the reference build; only introduce reduced precision after
action and cache parity tests pass. The companion Coppernic worktree supplies
contract validation, reference cache/RoPE state, height masking, latency-aware
30 Hz chunk lookup and a full-array TensorRT parity runner. These are not yet
connected to Palleter's live three-camera graph or actuator task.

Replan at 5 Hz while playing timestamp-aligned actions at 30 Hz; do not execute
all 50 actions open-loop. Retain the existing engagement, manual-override and
sensor-freshness interlocks. The reference plan expires 400 ms after the source
camera timestamp, including inference latency. Actual target timing and replay
must pass before robot deployment; nothing here starts a robot service.

## Interim Orin validation (2026-10-14)

The step-10000 checkpoint of run `zxpkh06j` was exported and replayed on
`max@172.30.0.203` (`delta-dev2`, Ubuntu 22.04, Orin, CUDA 12.6, TensorRT 10.3).
FP32/no-TF32 build passed: maximum absolute action error 2.38e-7 and KV error
2.22e-5 against the saved PyTorch fixture. Median engine-only latency was
58.48 ms over ten warmed runs; this excludes preprocessing, transfers and
control dispatch. The engine and report are under
`/home/max/d12-fm-validation/step10000/` on the kit and
`/nasa/max/d12/flow_matching/2026-10-14-h50/export-step10000/validation-kit/`
on the NAS. This is an intermediate checkpoint, not the final trained model.

`verify_d12_trt.py` requires only system TensorRT Python bindings, NumPy and
the CUDA runtime, so the kit does not need the training environment. The final
export watcher is also configured to build/replay on `.203` after training.
The intended deployment target remains `.169`. Berghain's yaak-eos 2.0 SDK
uses CUDA 13 and TensorRT 10.16.2; its Coppernic parity runner cross-build passed,
but that binary is not intended for the Ubuntu/CUDA12 `.203` kit. Rebuild the
engine for `.169`'s runtime when available; do not copy the `.203` engine as a
deployment shortcut.

## Matched fork-camera ablation

Run `ptktynvw` (`zesty-plant-65`) uses `--camera-count 1` with the same H50/30 Hz
targets, 5 Hz history, height mask, split, seed, batch 2 x accumulation 4,
learning-rate schedule and 35000-step budget as `zxpkh06j`. The audited data
tables, windows and action statistics are reused unchanged. Dataset alignment
still uses the original three-camera eligibility rules, so the ablation does
not silently gain different samples; only the model's consumed image stream
changes to `cam_fork`. The trunk has 149 rather than 437 tokens per frame.

The run directory is `/nasa/max/d12/flow_matching/2026-10-14-h50-1cam`.
Both variants use the same exporter and named camera bindings. The final-export
watchers keep validation-kit outputs in separate run directories. Training is
bf16-mixed; the reference exports and their KV I/O are FP32, with both PyTorch
matmul/cuDNN TF32 and TensorRT TF32 disabled for parity.

## Open-loop checkpoint evaluation

`python -m rmind.scripts.eval_d12_fm --checkpoint model.ckpt --config resolved.yaml --output eval/ --wandb-parent-run zxpkh06j` replays the usable portions of one
training recording (October 12, 15:40:43) and three held-out recordings
(September 10, October 2, October 12). The jobs are selected before inspecting
predictions and cover missing, retrofitted and live height. It compares ordinary
six-frame inference with continuous five-frame KV-cache inference, using the
same eight Gaussian noise draws per context and four Euler steps. Cache resets
on discontinuities; warm-cache/window parity is checked at each reset.

Outputs include per-job videos, timelines, CSV, full sampled chunks in NPZ,
JSON metrics and an HTML report. Metrics average individual sampled predictions,
not their ensemble mean; the separately labelled best-of-eight is optimistic
and is not the primary score. Near-horizon scoring uses the next six commands
(0.2 s); full-horizon scoring uses all 50. Predictions are not fed back into
recorded observations, and inference latency is not simulated. The command
deadbands are diagnostic thresholds, not calibrated physical motion thresholds.

The step-20000 three-camera checkpoint evaluation is logged as W&B run
`g8ylegzb`, with its artifact under
`/nasa/max/d12/flow_matching/2026-10-14-h50/openloop-step20000`.
KV near-horizon MAE versus literal zero-command baseline:

| Recording            | Split | FM MAE | Zero MAE |
| -------------------- | ----- | -----: | -------: |
| 2026-10-12--15-40-43 | train | 0.1033 |   0.1974 |
| 2026-09-10--15-17-38 | val   | 0.1671 |   0.3168 |
| 2026-10-02--15-49-44 | val   | 0.1485 |   0.2840 |
| 2026-10-12--15-59-15 | val   | 0.1139 |   0.2251 |

Across 714 validation contexts, near MAE is 0.1432 versus 0.2753 for zero,
and full-chunk MAE is 0.1451. All three predicted commands fall within +/-0.05
only 0.73% of the time, versus 6.61% in the recordings. Given an active recorded
command (magnitude >0.05), traction is missed 1.15% and fork actuation 2.41%
of the time. This is not evidence of no-action collapse. The opposite concern
is unwanted small commands while idle: predicted fork activity is 40.7% versus
18.5% recorded, and traction exceeds the 0.05 threshold in 70.7% of recorded
traction-idle cases. Steering has weaker correlation (0.546) and wrong sign in
17.3% of active cases. Plots show some weak/delayed lift transitions. These
numbers are open-loop diagnostics, not permission or evidence for safe deployment.

## Repeatable research and Rerun

Install with `uv sync --locked --extra train --extra export --extra eval --group test --group check --no-default-groups`. Video rendering also needs `ffmpeg`.
All commands run from this checkout; they never connect to robot actuation.

```sh
python -m rmind.scripts.eval_d12_fm \
  --checkpoint /path/to/model.ckpt --config /path/to/resolved.yaml \
  --output /path/to/eval --samples 8 --fm-steps 4 --rerun
rerun /path/to/eval/2026-10-12--15-59-15/openloop.rrd

python -m rmind.scripts.research_d12_fm \
  --checkpoint /path/to/model.ckpt --config /path/to/resolved.yaml \
  --output /path/to/research --rerun

python -m rmind.scripts.probe_d12_fm \
  --checkpoint /path/to/model.ckpt --config /path/to/resolved.yaml \
  --output /path/to/sensitivity --contexts-per-job 12
```

The evaluator accepts `--jobs`, `--samples`, `--seed`, `--fm-steps`,
`--noise-mode fresh|fixed|shift`, and `--no-video`. `--render-only --rerun`
rebuilds visualization from the output's existing NPZ/report without GPU
inference. `--reuse-features /path/to/eval` reuses saved window/KV features,
checking the exact checkpoint hash and resolved config, to isolate head sampling.
Seeds are stable per job, so changing the job list does not change its noise.
These seeds differ from the original `g8ylegzb` evaluation; compare variants
within the new suite, not tiny numerical differences between the two evaluations.

MP4s show all sampled action chunks and a moving red cursor on the recording
timeline. Rerun adds synchronized camera views, recording seconds/frame number,
recorded vehicle XY/heading and pickup/dropoff positions, and individual colored
sampled chunks. Scrub `recording_seconds`. Colors identify Gaussian samples,
**not persistent semantic modes**. The map shows recorded poses, not a simulated
trajectory obtained by integrating commands. Evaluation remains recorded-state
open loop, not a closed-loop success or safety test.

`research_d12_fm` evaluates 4/8/16 Euler steps using the same noise and features,
then fixed and horizon-shifted noise at four steps. Fixed noise is reused until
a cache reset. Shifted noise advances six action slots, adding six fresh Gaussian
slots. Replan diagnostics compare new action 0 against old action 6, plus the
actual old-5/new-0 command boundary, excluding discontinuities. Hold diagnostics
use separate actuator deadbands; holding traction/lift does not imply centered
steering. Batch-head timings exclude vision, trunk, transfers, and robot dispatch.

`probe_d12_fm` selects up to 12 contexts per job, stratifying hold, lift/lower,
driving with low forks and driving with high forks. Every intervention uses the
same eight Gaussian starts as its baseline. Image probes include each-camera and
all-camera blackouts, brightness changes, blur, stale history, and another time
in the same recording. Pickup and dropoff locations are shifted separately by
0.05/0.1/0.25/0.5/1/2 m in four world directions, consistently transformed into
each historical frame's ego coordinates. It reports paired action differences,
drive/lift magnitude, and collapse to idle conditional on baseline activity,
split by train/validation and phase. These probes test sensitivity and brittle
neutral outputs, **not whether a counterfactual action is correct**. Blackouts
and mismatched images are intentionally out of distribution.

`audit_d12_actions --output /path/to/alignment.json` reports raw vs processed
steering signs, source rates, and nearest-signal timestamp offsets. The tested
steering logs are approximately 10 Hz, despite 30 Hz output chunks. September's
training MCAP reverses raw steering; October's agrees with raw steering. This
does not independently establish physical calibration. Nearest joins sometimes
use a future signal (roughly up to 50 ms for steering); any future command-history
conditioning must use latest-past measurements and avoid expert-history leakage.

## October 14 research results and new training arms

The original three-camera step-20000 checkpoint, one train and three validation
jobs, eight samples, produces these validation near-horizon results:

| Sampling         |    MAE | Traction overlap revision | Steering overlap revision | Fork overlap revision |
| ---------------- | -----: | ------------------------: | ------------------------: | --------------------: |
| 4 steps, fresh   | 0.1445 |                    0.1168 |                    0.0947 |                0.1054 |
| 8 steps, fresh   | 0.1506 |                    0.1306 |                    0.1064 |                0.1190 |
| 16 steps, fresh  | 0.1540 |                    0.1382 |                    0.1130 |                0.1259 |
| 4 steps, fixed   | 0.1442 |                    0.0665 |                    0.0379 |                0.0543 |
| 4 steps, shifted | 0.1428 |                    0.0643 |                    0.0543 |                0.0552 |

More integration steps do not fix the hold issue on this checkpoint: unwanted
idle action magnitude rises. Coupled noise reduces boundary inconsistency but
does not prove semantic mode persistence, task success, or correct adaptation.
Files: `/nasa/max/d12/flow_matching/2026-10-14-h50/research-step20000`.

Sensitivity probes cover 48 contexts (12 train, 36 validation). Shifting dropoff
by 0.5/1/2 m changes near actions by 0.0261/0.0423/0.0733 on average, with
0.56%/1.28%/3.13% of previously active drive/lift samples becoming idle. At 2 m,
mean drive/lift magnitude decreases from 0.2257 to 0.1825. There is no abrupt
"only works at the exact training location" collapse in these probes; this
does not establish generalization to new pallet placements. All-camera blackout
changes actions by 0.0794; dimming by 30% changes them by 0.0051. Blackout does
not reliably produce a safe stop, so camera-health interlocks remain necessary.
Files: `/nasa/max/d12/flow_matching/2026-10-14-h50/sensitivity-step20000`.

The updated MP4s and Rerun recordings are published under W&B run `085qn9yf`.
The complete solver/noise comparison, sensitivity table and NPZ/source artifacts
are published under `dstv52z8`, in the same `yaak/rmind-palleter` project.

New runs on tresor (`192.168.207.242`):

- GPU 0: `3lgykvqc`, `graceful-galaxy-68`, fork + left/right **backward** cameras.
- GPU 1: `elqw5g88`, `fancy-music-67`, original forward triple, balanced sampling.

Both retain H50/30 Hz, six 5 Hz observations, continuous height/mask, the same
split/seed/optimizer budget, and frozen image encoder. Backward cameras are an
additive dataset overlay at `/nasa/max/d12/datasets/backward`; original data is
not overwritten. All 38 jobs pass the original pairing guard, and audits have
identical sample counts and action statistics: 54,799 train / 944 validation.

To reproduce preparation/training:

```sh
python -m rmind.scripts.prepare_d12_cameras --source /nasa/data/d12/yaak \
  --output /path/to/backward-overlay --audit /path/to/original/audit.json
python -m rmind.scripts.prepare_d12_fm --output /path/to/backward-run \
  --camera-layout backward --data-root /path/to/backward-overlay
python -m rmind.scripts.train_d12_fm --run-dir /path/to/backward-run \
  --camera-layout backward
python -m rmind.scripts.train_d12_fm --run-dir /path/to/balanced-run \
  --sampling balanced
```

The balanced run reuses an independently copied original audit/statistics pair.
Sampling is 50% uniform plus 50% uniform across nonempty exclusive categories:
ordinary, sustained hold, lift/lower, steering change, start/stop. Categories use
the newest frame's first 12 target commands; transitions take precedence.
Validation is unchanged, and the FM loss is unchanged. `--steering-weight 2`
supports a separate unlaunched loss-weight ablation through the vendored head's
existing weighted-mask loss. It is not mixed into the balanced run.

`watch_d12_fm --run-dirs RUN... --step 20000 --status /path/to/status.json`
serially evaluates matching steps as checkpoints appear. Missing checkpoints
wait; failed evaluations are explicitly recorded. Training/export watchers also
support remote training hosts and explicit SSH identities. Export still requires
target-specific rebuilding and live Coppernic integration before robot use.

## Replanning does not guarantee mode consistency

Current ONNX/TRT graphs take explicit `fm_x0` and return one H50 chunk. The eight
variants are an evaluation choice (batched Gaussian starts), not eight outputs
from the deployed single-sample graph. KV state carries observation context,
not the previously chosen action chunk. Fixed or shifted noise is an ablation,
not a mode identity or a guarantee. The live D12 FM task has not yet been wired
into Coppernic.

For production continuity, evaluate conditioning/inpainting on the unexecuted
previous chunk, preserving actions committed during inference latency while
allowing the future plan to change. Real-time chunking is a relevant reference:
https://arxiv.org/abs/2506.07339. Neither that algorithm nor a neutral gate is
implemented in serving here; no mode-sticking or robot-safety claim is made.

## Final comparison (2026-10-15)

Published metrics, action-separated sensitivity plots and source artifacts:
https://wandb.ai/yaak/rmind-palleter/runs/2dyo76vu.

All four runs finished 35,000 optimizer steps. The final comparison includes all
four held-out recordings (944 contexts), adding the previously omitted August 25
episode to the three research recordings. Predictions use streaming KV, four
Euler steps and eight paired Gaussian starts, scored individually over the first
six 30 Hz commands (0.2 seconds), not averaged into an ensemble action.

| Model                 | Near MAE | Traction MAE | Steering MAE | Fork MAE | Traction false-active on idle | Fork false-active on idle |
| --------------------- | -------: | -----------: | -----------: | -------: | ----------------------------: | ------------------------: |
| Forward 3cam          |   0.1362 |       0.1486 |       0.1701 |   0.0900 |                         65.2% |                     20.5% |
| Fork-only             |   0.1323 |       0.1431 |       0.1672 |   0.0865 |                         63.8% |                     19.0% |
| Backward 3cam         |   0.1366 |       0.1474 |       0.1709 |   0.0915 |                         66.1% |                     20.9% |
| Balanced forward 3cam |   0.1409 |       0.1503 |       0.1769 |   0.0956 |                         60.0% |                     28.7% |

The zero-command baseline is 0.2675. These are normalized command errors, not
task success. The 0.05 deadband is not a calibrated physical-motion threshold.
Fork-only leads the pooled comparison; backward cameras beat the original
forward triple on both October recordings but lose on both older recordings.
One seed and four correlated recordings cannot establish a universal winner.
Balancing helps traction holding but harms fork holding and overall error.
For strong steering targets (absolute command >0.4), mean predicted magnitude is
only 52–61% of mean recorded magnitude. None is ready for unsupervised robot use.

All models improve from 20k to 35k on the original three research recordings.
Eight/sixteen Euler steps worsen near error versus four; shifted noise gives a
small improvement, not guaranteed mode continuity. Keep four steps. The next
focused experiments should separate per-actuator neutral/active behavior and
test causal steering-state/command history, rather than increasing solver cost.

All four final local RTX4090/TRT11 artifacts pass native Coppernic parity.
The completion pipelines then fail at rsync with an ACL/GLIBC library mismatch;
the final engines have not been validated on `.203`. Local parity timing is not
a deployment benchmark, and the live Coppernic FM task is still unimplemented.

Reproduce the pooled comparison after generating the research and fourth-holdout
evaluations:

```sh
python -m rmind.scripts.compare_d12_fm --root /nasa/max/d12/flow_matching \
  --output /nasa/max/d12/flow_matching/comparison-final
```

### Action-separated goal sensitivity

`goal_sensitivity` now has traction, steering and fork columns, each showing
paired absolute change, mean absolute command versus the unperturbed baseline,
and probability of becoming near-zero conditional on a nonzero baseline. Near-zero
steering means centered, not stopped. The table retains individual offset
directions and train/validation/behavior slices, including image interventions.
All eight saved 20k/35k reports are refreshed without new model inference.

For the referenced forward-3cam 20k report, a 2 m drop-off perturbation changes
traction/steering/fork commands by 0.060/0.042/0.118 on average. Mean absolute
traction falls only 3.5%, but fork magnitude falls 39.4%. At 35k the corresponding
fork reduction is 48.5%. The previous combined drive/lift collapse statistic hid
this selective fork suppression: continued traction does not establish that the
fork action generalizes. These probes establish sensitivity, not whether a
changed action is correct; there are no counterfactual expert targets.
For comparison, the final fork-only/backward/balanced models reduce mean fork
magnitude by 19.6%/18.3%/42.7% under the same 2 m drop-off probes. The backward
model is less susceptible to this particular suppression than the forward model,
but unchanged activity alone does not demonstrate correct goal-conditioned control.

Refresh and optionally update an existing W&B run:

```sh
python -m rmind.scripts.probe_d12_fm --refresh-report \
  --output /nasa/max/d12/flow_matching/2026-10-14-h50/sensitivity-step20000 \
  --wandb-run yaak/rmind-palleter/dstv52z8
```

The W&B update replaces the latest `goal_sensitivity` panel, adds
`sensitivity_by_action`, and versions the report/plot artifact. Historical images
remain in run history.

## Fork-only verification on the actual truck (2026-10-15)

`root@172.30.0.169` is online as `delta-d12`, running Yaak Embedded OS 2.0-rc6,
Orin, CUDA 13.2 and TensorRT 10.16.2. The final fork-only checkpoint (`ptktynvw`,
35000 steps) was built there as FP32/no-TF32 and replayed through the SDK2-built
native Coppernic `d12-flow-parity` binary. All input/output names and shapes
match; maximum absolute errors are `5.66e-7` for actions and `2.20e-5` for KV.

After ten warmups, 100 repeated fixed-fixture calls give p50 **19.75 ms**, p95
**19.85 ms**, maximum **20.07 ms**. One call includes the current fork image,
cached trunk, all four Euler steps and the complete H50 output, plus output
download. It excludes camera capture/preprocessing, input upload and host-side
cache/control scheduling; it is not a live end-to-end benchmark.

Truck directory: `/data/d12-fm-validation/fork-only-step35000-20261015`.
NAS engine, binary, logs, parity report and runtime/checksums:
`/nasa/max/d12/flow_matching/2026-10-14-h50-1cam/export-final/validation-d12-169`.
The previous `.203` transfer failure does not affect this new actual-target
validation; historical completion status is retained rather than overwritten.

No services were restarted or actuation enabled. The existing `palleter` logger
remains running, and the actuation service is not installed. Native inference
works, but the live policy graph still requires the FM adapter: its current
contract consumes six packed images and emits three scalar heads, not one image
plus KV/noise/height and an H50 action chunk. Next is no-actuation shadow serving
with real camera/state inputs, cache resets and timestamp-aligned chunk handling.

The original target benchmark used 40W mode with dynamic clocks. On the user's
request, `.169` was subsequently switched to MAXN_SUPER and fixed CPU/GPU/EMC
clocks (1.984/1.173/3.199 GHz). The same 100-call benchmark passes parity and gives
p50 **19.147 ms**, p95 **19.180 ms**, maximum **19.567 ms**. The results are archived
as `coppernic-parity-maxn.json`/`.log`; the original benchmark is retained.
An explicit mode-0 systemd override and enabled `jetson-clocks.service` persist
the settings after reboot. No reboot or actuation was performed. Configuration
sources and rollback steps are in Coppernic's `scripts/deploy/d12-max-performance/`
and `crates/copper/cu-infer/D12_FLOW.md`. Automatic fan control is unchanged.

## Live FM adapter and sequence fixtures (2026-10-15)

Coppernic now has an opt-in fork-only live FM path selected by
`PALLETER_DRIVR_FLOW_MANIFEST`: one unit-RGB fork image at 5 Hz, masked height,
external KV/RoPE management, fresh Gaussian starts and timestamp-indexed 30 Hz
H50 playback. The no-control-seam shadow logger is deployed on `.169`; the
original logger binary is preserved. No actuation has been enabled. Live scoring
is blocked until mandatory vehicle/state inputs and an operator-set drop-off
are available; their gates are not bypassed.

Generate fresh streaming references without re-exporting the engine:

```sh
python -m rmind.scripts.fixture_d12_fm_sequence \
  --manifest RUN/export-final/manifest.json --config RUN/resolved.yaml \
  --output RUN/export-final/sequence-validation
```

The default 16-frame held-out sequence includes two eight-frame segments with
empty startup, cache eviction and a synthetic generation reset. Coppernic's
`d12-flow-parity --sequence .../sequence.json` uses its actual live host-state
implementation, deliberately zeroing uploaded reference cache/RoPE. All 16
steps pass on `.169`: maximum action error `4.86e-6`, K `7.03e-4`, V `6.76e-4`.
Median inference including host cache upload/advance is **34.62 ms**; the original
19.15 ms fixed-cache result does not include that work. Neither measures the
complete live-camera/control path. Reports are archived in `validation-d12-169`.

The deployed shadow app passes all 173 unit tests on the truck. Synthetic NVMM
black/white frames match direct unit-RGB inference exactly, including input
snapshot isolation. The actuation release builds but is not deployed or running.
Diagnostic recording is stopped; the shadow daemon remains ready. Real live
predictions still await truck speed/fork-switch, UWB poses and a selected drop-off.
