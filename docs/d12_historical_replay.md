# D12 historical FM chunk replay

## Result: weak across the horizon, not a repeatedly discarded acceleration ramp

On 2026-10-16 we replayed the fork-only `ptktynvw` step-35000 model on recorded
2026-10-15 inputs. The two long stationary approaches predict weak traction across
the entire 50-action horizon. Sixteen independent noise samples per context show
the same pattern. Keeping a chunk longer would not generally expose a strong
acceleration suffix in these two examples.

| Interval (UTC)                        | Contexts | Mean absolute traction, indices 1–7 | Mean absolute traction, indices 8–49 | Chunks with a sustained stronger suffix | Emitted-action reconstruction MAE |
| ------------------------------------- | -------: | ----------------------------------: | -----------------------------------: | --------------------------------------: | --------------------------------: |
| 09:40:19.765–09:41:00.066, stationary |      202 |                               5.86% |                                6.47% |                                   2.48% |            0.51 percentage points |
| 09:42:27.869–09:43:06.670, stationary |      194 |                               5.91% |                                5.92% |                                   2.06% |            0.38 percentage points |
| 12:45:05.035–12:46:22.135, successful |      386 |                              26.26% |                               26.19% |                                  65.28% |            1.83 percentage points |

The illustrative stronger-suffix criterion is 15 consecutive actions (0.5 s)
with the same sign and absolute traction greater than 0.15. It is **not a
calibrated actuator deadband**. With 16 fresh-noise variants per context, those
fractions are respectively 2.75%, 1.13%, and 64.52%. Indices 1–7 approximate the
usual executed window; actual indices vary with latency and scheduling.

These are approximate input replays, **not exact historical ground truth or
physical counterfactual rollouts**. The result weakens the postponed-acceleration
hypothesis for these two stalls. It does not rule out benefits from RTC, which
can improve boundary continuity and condition on already committed actions.
Targeted alignment/recovery demonstrations and actuator-response measurements
remain important: smoothing alone cannot invent a confident missing approach.

## Evidence and caveats

- Both recordings identify the same archived Orin engine MD5,
  `d1173c15c4035e381b4b47274234dfd6`. Checkpoint SHA-256 is
  `3c867cc4d2364711a897d8ad6d725064c140ddf8c21aefc0846c98dba3a11dc0`.
- Model inputs include exact recorded auxiliary tokens and reset generations.
  Camera references are joined to recorded frames by **PTS**, not frame index.
  The policy and recorder counters differ by 36 frames in the morning recording
  and 40 frames in the successful session. Ignoring that would shift images by
  approximately 1.2–1.33 seconds.
- Images are decoded from lossy 1920×1080 MP4, then resized using the serving
  half-pixel bilinear/uint8 sequence, 576×324 followed by 256×144. The original
  UYVY samples and exact post-preprocessing tensors were not logged. This prevents
  exact numerical parity. No image reconstruction is described as an exact
  serving tensor.
- Streaming replay preserves the five-frame KV ring, generation resets and
  generation-local RoPE counter. Unknown dropped/rejected calls can affect cache
  reconstruction. The target stationary intervals follow explicit generation
  resets and exhibit substantially better emitted-action matches than the
  recording as a whole.
- `StdRng` seed 287 is reproduced using `rand 0.9.5`, `rand_distr 0.5.1`, and a
  pinned Rust lockfile matching the serving random-number dependencies. RNG state
  belongs to the `FlowInference` instance, not the KV generation. Rebuilding the
  recording graph creates a new instance and reseeds; ordinary cache resets do
  not reseed.
- The morning stream matches request-index noise until request 353. Requests
  354 and 355, separated by roughly 30 ms across a generation change, have no
  emitted predictions. From request 356 onward, **request index minus one** is
  consistently the best candidate noise alignment. At least one request did not
  consume the expected noise vector. Which of the two requests was skipped is
  not established. The CLI records the explicit piecewise hypothesis rather than
  silently optimizing a different seed per frame.
- The successful recording matches the unshifted stream. Per-chunk comparisons
  audit neighboring noise offsets and action-slot shifts. Candidate chunks are
  clamped to `[-1,1]` only for comparison with emitted actions, matching the player;
  raw chunks are retained in output arrays.
- Executed-slot estimates use prediction and camera wall timestamps. Exact
  control-clock action indices were not recorded, so ±1 slot comparisons are
  retained separately. Reported MAEs use nominal timestamp-derived slots, not
  best-per-frame slot selection.

## Artifacts

NAS root:
`/nasa/max/d12/flow_matching/rollouts/2026-10-15/replay/`

- `comparison/report.html`, `comparison/summary.json`, and
  `comparison/traction_horizons.png`: comparison and individual noise variants.
- `morning-aligned/`: 1,264 replayed inputs through the second stationary interval.
- `success/`: 652 replayed inputs through the successful interval.
- Each replay contains `chunks.npz` (candidate full chunks, 16 fresh-noise
  variants, wall/tov timestamps, generations, video frame indices and candidate
  noise indices), `features.npz`, `inputs.json`, and `report.json` with per-chunk
  validation/provenance. Feature reuse checks a checkpoint/input/preprocessing
  fingerprint. `morning/` additionally preserves the initial unshifted-noise audit.

## Reproduction

Run from the rmind checkout. Videos and complete MCAPs must be available locally.
The current script explicitly assumes 300 retained video frames per numbered
MP4 and validates the selected files; other chunking conventions need an explicit
video index instead of this assumption. GPU execution is offline; it does not
connect to or control the truck.

```sh
ROOT=/nasa/max/d12/flow_matching/rollouts/2026-10-15
CHECKPOINT=/nasa/max/d12/flow_matching/2026-10-14-h50-1cam/checkpoints/epoch=5-step=35000.ckpt

cargo run --release --locked --manifest-path tools/d12_noise_stream/Cargo.toml -- 10000 \
  > "$ROOT/rust-noise-seed287.f32"

.venv/bin/python -m rmind.scripts.replay_d12_rollouts \
  --job "$ROOT/raw/2026-10-15--09-33-42" --checkpoint "$CHECKPOINT" \
  --rust-noise "$ROOT/rust-noise-seed287.f32" --device cuda:1 \
  --max-inputs 1264 --noise-offset-change 356:-1 \
  --output "$ROOT/replay/morning-aligned"

.venv/bin/python -m rmind.scripts.replay_d12_rollouts \
  --job "$ROOT/raw/2026-10-15--12-43-19" --checkpoint "$CHECKPOINT" \
  --rust-noise "$ROOT/rust-noise-seed287.f32" --device cuda:1 \
  --max-inputs 652 --output "$ROOT/replay/success"

.venv/bin/python -m rmind.scripts.summarize_d12_replay \
  --segment stationary_40s "$ROOT/replay/morning-aligned" \
    2026-10-15T09:40:19.765+00:00 2026-10-15T09:41:00.066+00:00 \
  --segment stationary_39s "$ROOT/replay/morning-aligned" \
    2026-10-15T09:42:27.869+00:00 2026-10-15T09:43:06.670+00:00 \
  --segment successful_77s "$ROOT/replay/success" \
    2026-10-15T12:45:05.035+00:00 2026-10-15T12:46:22.135+00:00 \
  --output "$ROOT/replay/comparison"

.venv/bin/pytest -q tests/test_d12_replay.py tests/test_d12_rollouts.py
```

## Minimum next-rollout logging

1. An inference-instance UUID and actual inference-call counter, separate from
   submitted input count; record rejected/dropped inputs and reasons.
1. The complete `fm_x0` and full output action chunk, associated with frame PTS,
   generation, inference start/end and model/engine identity. H50 float32 noise
   plus actions is only about 6 kB/s at 5 Hz before framing overhead.
1. Plan adoption/replacement events and each executed `(plan_id, action_index)`
   with control-clock timestamp, published command and safety-gate outcome.
1. Cache reset reason and generation-local frame counter. Preserve engine MD5,
   checkpoint hash, manifest, RNG algorithm/version and preprocessing version.
1. For exact replay, lossless post-resize RGB inputs or exact input tensors, not
   only MP4 references. A 144×256 RGB uint8 stream at 5 Hz is approximately
   0.55 MB/s before compression. Include exact auxiliary input snapshots.

No serving code, truck services, actuation settings or deployment was changed.

Feature caches bind the selected MP4 content hashes and feature-array digest.
Older caches without these fields must be rebuilt. Supplying `--features-from`
requires its independently recorded `--features-sha256`; external caches cannot
be rebuilt in place. Noise-input path and SHA256 are recorded, but the generator
algorithm/seed are only declared unless independently verified from a receipt.

# Lint scope

The three historical replay CLIs retain their validated sequential parsing and
replay loops through narrowly named complexity exceptions in `pyproject.toml`.
Splitting those loops solely for lint would obscure timestamp, cache, and RNG
ordering. Their stdout progress/JSON output is intentional. Corrupt MCAP summary
exceptions are recorded in the explicit skipped-file evidence, never accepted as
successful inputs. Replay keeps Torch imports lazy so CPU-only provenance helpers
remain usable, and deliberately calls the policy's existing `_frame_tokens`
instead of duplicating model preprocessing. Video subprocesses use resolved
ffmpeg/ffprobe executables and fixed argument arrays without a shell. These
exceptions do not disable tests, checksum checks, numerical gates, or outcome
criteria; diagnostic thresholds are named without changing their values.
