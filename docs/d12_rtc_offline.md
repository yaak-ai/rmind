# D12 native RTC offline comparison

This compares `nmsso8zf` step35000 with deployed baseline `ptktynvw` step35000.
It does not control the truck or estimate counterfactual closed-loop success.
The models differ in architecture and training; their difference is not a
controlled estimate of the effect of prefix conditioning alone.

Model and preprocessing configuration are loaded from the exact hash-verified
protocol records, not separately selected defaults. Newly extracted token caches
bind those records and their array checksum; older unbound caches must be
re-extracted before a new replay. Integrity checks are explicit exceptions, not
assertions that can disappear under optimized Python.

## Protocol

The frozen protocol and evidence live at
`/nasa/max/d12/flow_matching/d12-rtc-offline-20261018`.

- All four original held-out expert jobs: 944 windows, height-valid and missing
  strata, action-separated errors, command-defined hold/lift/lower diagnostics.
- Explicit paired Gaussian noise seeds131/137/139. These are sampling seeds,
  not independent training seeds or discovered mode labels.
- Unconditional expert predictions are primary. RTC prefixes3/6 taken from
  expert targets are separate oracle-conditioned suffix diagnostics; masked
  prefix tokens are excluded from errors and flow loss.
- Two previously documented stationary/no-acceleration intervals and one
  successful session. Failed recorded commands are never treated as targets.
  There is no grounded missed-pocket or semantic dropoff-event annotation in
  this cohort; command-defined lowering is not called a successful dropoff.
- Three scheduler arms: old unconditioned head, new unconditioned head, and new
  prefix-conditioned head. All use the same actual Coppernic `RtcController`.
  The old head in that executor is not an exact recreation of the legacy player.
- Zero latency, historical measured latency, and a fixed stress intervention:
  every50th observation250ms, every100th450ms. The measured trace is old-model
  input-to-first-output timing, not measured native-RTC Orin inference time.

The replay uses a 30Hz integer-nanosecond action grid and 5Hz observation
requests. At each request it chooses only the most recent already-recorded
input. The controller supplies the aligned, canonical old-plan prefix; the
model normalizes it once. Only accepted requests advance the KV cache.
Generation changes reset KV state and invalidate in-flight work. The previous
plan runs during inference; elapsed new actions are discarded at handover.
Freshness expiry is fail-closed and latched until a recorded generation reset,
not silently replaced by zero actions or automatic re-engagement.

Two request-clock variants are retained separately. The original fixed5Hz
grid samples the latest already-available recorded input, but these inputs
were themselves logged at approximately5Hz: phase mismatch adds artificial
image age. It causes a400ms freshness trip in the first stationary segment.
The corrected controller-paced variant polls available inputs at30Hz and lets
the unchanged controller enforce its six-tick minimum request interval. Busy
or not-yet-due observations are retried, not discarded. Its action clock is
rebased to the controller's generation-reset origin; an actual-Rust regression
test covers non-100ms-aligned resets and prevents one-nanosecond rational-grid
duplicate ticks. The intermediate paced-v1 harness bug and partial outputs are
preserved, excluded from the corrected cohort, and never pooled with it.

One second of recorded warmup precedes each scored interval. Recorded frames
are joined by PTS and decoded using the existing lossy-MP4 serving-resize
approximation. They are not exact historical input tensors. Histories do not
change in response to hypothetical commands: this remains open-loop replay.

## Reproduce

The branch includes the focused dependencies from PR #289 (native RTC) and
PR #290 (historical replay); no dirty checkout is required. Install the locked
rmind environment and select this checkout's `src` on PYTHONPATH. Coppernic's
`feat/d12-rtc-serving` branch provides the actual controller; initialize its
pinned iondrive submodule when building the full serving binaries. The small
evaluation wrapper only compiles that controller source and its locked Rust
dependencies, not a duplicate Python scheduler.

This is an exact, private-data reproduction, not a portable public dataset.
It requires NAS access to both model directories, their original 38 MCAP jobs,
the historical recording videos and extracted PTS-joined input JSONs under
`/nasa/max/d12/flow_matching/rollouts/2026-10-15`, and the measured latency JSON
under `rtc-robustness-20261016`. `freeze` hashes the selected model/config and
recording inputs; provenance additionally verifies all original dataset files.
Existing evidence directories are immutable; choose a new output directory.

```sh
ROOT=/nasa/max/d12/flow_matching/d12-rtc-offline-reproduction-NEW
CONTROLLER=/path/to/coppernic/crates/copper/cu-infer/src/d12_rtc.rs
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=GPU-d0a539a2-35e4-0445-be7e-969b1698d61c
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4

RTC_CONTROLLER_SOURCE="$CONTROLLER" CARGO_TARGET_DIR="$ROOT/rust-target" \
  cargo build --offline --locked --release --manifest-path tools/d12_rtc_stream/Cargo.toml
BINARY="$ROOT/rust-target/release/d12-rtc-stream"

# Use a NEW ROOT for a new run; exclusive report creation prevents replacing evidence.
.venv/bin/python -m rmind.scripts.evaluate_d12_rtc freeze --output "$ROOT"
for MODEL in baseline rtc; do
  .venv/bin/python -m rmind.scripts.evaluate_d12_rtc expert --model "$MODEL" --output "$ROOT"
  .venv/bin/python -m rmind.scripts.replay_d12_rtc extract --model "$MODEL" --output "$ROOT"
done
.venv/bin/python -m rmind.scripts.provenance_d12_rtc --output "$ROOT" \
  --controller-source "$CONTROLLER" --controller-binary "$BINARY"
export PYTHONPATH="$ROOT/source_snapshot/src"
for MODEL in baseline rtc; do
  .venv/bin/python -m rmind.scripts.replay_d12_rtc replay --model "$MODEL" \
    --output "$ROOT" --controller "$BINARY"
done
.venv/bin/python -m rmind.scripts.report_d12_rtc --output "$ROOT" --video
for MODEL in baseline rtc; do
  .venv/bin/python -m rmind.scripts.replay_d12_rtc replay --clock paced \
    --model "$MODEL" --output "$ROOT" --controller "$BINARY"
done
.venv/bin/python -m rmind.scripts.report_d12_rtc --output "$ROOT" --video \
  --async-folder async-paced --report-dir "$ROOT/paced-report"
.venv/bin/python -m rmind.scripts.audit_d12_rtc --output "$ROOT" \
  --folder async-paced --controller "$BINARY"
D12_RTC_STREAM_BINARY="$BINARY" .venv/bin/pytest -q tests/test_d12_rtc_evaluation.py
```

The measured profile remains the older deployed model's latency, not a new
Jetson measurement. Local TensorRT engine parity and target/shadow requirements
are documented separately in `docs/d12_native_rtc.md` and Coppernic's
`crates/copper/cu-infer/D12_RTC_VALIDATION.md`.

`report.html` links action plots and paired camera/action videos. Raw NPZs retain
explicit noise, commands, plan IDs, action indices and timestamps; JSON traces
retain every actual controller request/completion/action response. Command
coverage is reported separately from action magnitude. The .05/.10/.15
thresholds and15-tick sustained traction criterion are command-space proxies,
not calibrated actuator deadbands or proof of physical acceleration.

## Initial expert result

Unconditional first-six-action MAE, averaged over all944 windows and all seeds:

| Action   | Baseline | Native RTC |
| -------- | -------: | ---------: |
| Traction |   .14295 |     .13062 |
| Steering |   .16866 |     .16019 |
| Fork     |   .08620 |     .04273 |

The aggregate improvement is not uniform: October2 steering worsens from
.16747 to .17889. False activation during expert idle commands falls, but
missed active steering/fork commands increase; the full per-action report must
be considered rather than only average error. Neither this result nor smoother
replay commands authorizes deployment.

## Corrected paced replay result

All81 cells finish with three sampling seeds per arm/profile/segment. Under the
historical measured latency trace, command coverage is99.83%,99.83%,99.91% for
the two stationary intervals and successful session, respectively, equal across
model arms. Missing commands are not counted as zero-valued predictions.

In the successful session, baseline-to-RTC mean traction seam magnitude falls
.10893→.01692 and steering seam magnitude .07132→.01018. The fraction of
overlapping500ms windows with sustained same-sign traction above.15 rises
38.49%→74.93%; the new head without prefix conditioning achieves52.17%.
These are command-continuity proxies on recorded observations, not motion or
success-rate measurements.

The two stationary approaches are not rescued. Mean absolute traction falls
from6.15%→3.54% and6.03%→2.80%, with zero sustained-traction windows in every
arm. Prefix conditioning stabilizes those weak predictions rather than creating
a missing confident approach. Thus continuity looks substantially better, but
the OOD alignment/no-acceleration issue still needs targeted data and physical
actuator-response validation. Forced overruns deliberately exercise fail-closed
freshness behavior; their long gaps are not interpreted as small commands.

The preferred corrected visualization is `paced-report/report.html`. Original
fixed-grid reports remain at the root, and the rejected paced-v1 clock-bug
outputs remain under `async-paced-v1-grid-bug`. Exact Rust event re-execution,
paired explicit-noise checks, and causal observation/prefix checks are recorded
separately in the audit JSONs.

## Export check

The trained native checkpoint exports through the full D12 streaming graph,
including raw prefixes, normalization, explicit noise and KV outputs. On the
local4090, FP32/no-TF32 TensorRT11.1 with builder optimization level0 passes
the existing Coppernic numerical fixture gate: maximum action error2.68e-7,
new-K3.62e-5, new-V4.15e-5. ONNX Runtime passes the stricter1e-4 export gate.
This artifact is explicitly `test_only`; it is not deployed. A single fixed
fixture does not replace a trained-model stateful sequence test, Orin/TensorRT
version compatibility, real-kit latency measurements or shadow validation.

# Lint and integrity scope

The offline CLIs keep their reviewed sequential scheduler, paired-noise, and
cohort-audit loops through explicitly named per-file complexity exceptions.
Existing positional replay interfaces remain compatible with archived callers;
the audit predicate accepts a positional condition like the assertion it replaces.
Production audit predicates now raise `ValueError` explicitly, including under
`python -O`; no integrity check is disabled. CLI stdout is intentional. External
commands use resolved executable paths and fixed argument arrays without shells;
the test's Python subprocess proves optimized-mode checksum rejection. Private
frame-token access deliberately reuses production preprocessing rather than a
second implementation. These exceptions do not relax tests, evidence checks,
model numerical gates, or outcome criteria.
