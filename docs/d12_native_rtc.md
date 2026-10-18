# Native training-time RTC for D12

This opt-in head leaves the existing MLP flow head and legacy export contract
unchanged. It is not a converted MLP checkpoint: the token-time mixer must be
trained. New models use analytical native time encoding without JAX or an
externally supplied frequency vector. Explicit reference-compatibility and
legacy-dynamic modes exist only for previously converted research checkpoints.

## Contract

- `TokenTimeFlowHead` implements the existing `ActionHead` training interface.
- Prefix tokens have clean actions and time 1; suffix tokens follow Euler time.
  Sampling restores the prefix before every step and after the final step.
- Training samples prefix lengths proportional to `exp(-length)`. Loss excludes
  those tokens and normalizes by valid action elements, including action weights.
- Native frequencies are derived once from channel dimension and period bounds,
  serialized as a buffer, and checked on load. They do not depend on horizon.
- The mixer weights do depend on horizon: changing H requires a compatible model,
  not merely resizing its inputs.

The export adds `clean_prefix` `[1,H,3]` and `prefix_mask` `[1,H]`. Prefix actions
are **raw canonical commanded units**; the graph normalizes them once. Noise
`fm_x0` is already standardized. The host must supply a contiguous binary mask,
align the prefix to the request anchor, and never apply command gains twice.
Exports advertise `d12_flow_matching_rtc_v1`, per-token time, final prefix clamp,
and a last-completed causal latency forecast. Old exports retain their schema.

## Actual D12 experiment

`train_d12_rtc` consumes an audited fork-only baseline directory containing its
resolved configuration, audit, action statistics, and checkpoint. It preserves
the original 34 training / 4 validation jobs, normalization, fork-height missing
mask and `data_for_training.mcap` pipeline. It loads all compatible non-head
weights, rejects normalization or tensor mismatches, and initializes a new head.

The completed experimental run `nmsso8zf` uses cam_fork, 144x256 images, H50 at
30 Hz, six-frame history at 5 Hz, max prefix 6, five Euler steps, FP32/no-TF32,
frozen DINO, LR 3e-5, and 35,000 steps. Its baseline is `ptktynvw` step 35000.
This experiment was explicitly authorized despite unresolved native/reference
proxy efficacy and failed 5 Hz walker competence gates. It is not deployment
certification. The generic Hydra scaffold inherits its camera/data selection;
it is not the exact fork-only launcher recipe.

```sh
just generate-config
python -m rmind.scripts.train_d12_rtc --help
python -m rmind.scripts.export_d12_fm --checkpoint CHECKPOINT --config CONFIG \
  --output NEW_DIRECTORY --test-only
pytest -q tests/test_rtc_native_time_embedding.py tests/test_rtc_time_frequency.py \
  tests/test_rtc_token_time.py tests/test_d12_rtc_launch.py
```

Hydra configuration-composition tests require generated datasets from the
tracked ytt templates; `just generate-config` creates those ignored files.

Use only trusted Lightning checkpoints: loading them uses Python pickle.
Keep verification artifacts `test_only`; the Coppernic live app rejects them.
Smoke checkpoints carry durable quarantine metadata that the exporter preserves,
even without `--test-only` or after renaming the file. Eager sampling rejects
nonfinite prefixes and nonbinary, noncontiguous or overlong prefix masks. Exported
graphs retain the host-validation contract; the live controller constructs masks.
The launcher creates exclusive provenance records and does not resume runs.

Two narrowly scoped lint exceptions preserve the audited design: the head's
explicit-randomness loss API retains its six tensor arguments rather than hiding
them in a packed object; the provenance launcher imports `subprocess` for fixed
commands with resolved executables. These exceptions do not disable tests,
input-integrity checks, numerical gates or quality acceptance criteria.

## Evidence and limits

The local native D12 checkpoint passes eager/ONNX/ORT and FP32 TensorRT 11.1
parity on a 4090. That engine is not a Jetson artifact. Target engines must be
built and validated against the target TensorRT version, hardware and full
stateful KV/request-generation sequence before shadow testing.

The paired offline evaluation reports lower aggregate expert error and smoother
successful-session replanning, but stationary approaches remain weak. Recorded
images cannot establish counterfactual robot success. Failed commands are not
expert labels. Sampling seeds are not independent training seeds.

Clean-source verification evidence is kept under
`/nasa/max/d12/flow_matching/rtc-production-cleanup-20261018`.
Historical frozen research artifacts remain separate; this production change
does not silently reclassify failed proxy studies or promote a model.
