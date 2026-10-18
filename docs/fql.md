# Opt-in Flow Q-learning

This independent low-dimensional production model follows the FQL reference
architecture and loss at `https://github.com/seohongpark/fql` commit
`e8cd16eb490332924dfa2492097219f181765933`. The exact upstream repository URL and
reference runtime are bound by the benchmark protocol, not imported at runtime.
It is not the exploratory benchmark's two-layer ReLU rmind-prior adaptation.

`FQLPolicy` owns a four-layer GELU BC flow prior with raw scalar time, a separate
one-step distilled actor, two LayerNorm critics and two lagged target critics.
Actor Q gradients pass through actions but not critic parameters. BC and target
actions are detached. The target EMA uses **pre-optimizer** critic parameters.
Adam uses the reference beta/epsilon settings without weight decay or clipping.
Native Torch initialization has the same Xavier-uniform distribution, not the
same PRNG bits as JAX. Explicit `FQLNoise` supports matched numerical comparison.

## Interfaces and data contract

- The component consumes standardized vector observations and single canonical
  actions in [-1,1]. Rewards and binary bootstrap masks are explicit inputs.
- `FQLModel` is the Lightning integration: observations/actions at its public
  training/inference boundary use physical units. Observation mean/std and
  action low/high are checkpointed with the architecture and contract version.
- `model(observations, noise)` returns physical bounded actions from the **one-step
  actor**, not the BC prior. Noise is standard normal in canonical action units.
- `policy.compute_flow_actions` explicitly samples the BC prior with ten Euler
  steps. No RTC prefix, chunk execution, KV cache, image encoder or D12 serving
  compatibility is implied. Existing FM/VQ/RTC defaults are untouched.
- FQL currently requires FP32, no external gradient accumulation/clipping.
  The supplied config requests highest float32 matmul precision. No AMP or TRT
  compatibility is certified by these interfaces.

`FQLDataModule` consumes two NPZ archives, with `allow_pickle=False`, containing
`observations`, `next_observations`, `actions`, `rewards`, `masks`, `episode_ids`.
Features/actions are matrices, the other fields vectors. Episode IDs must be
globally unique recording identities; overlapping train/validation IDs fail.
Rewards/masks are never derived from a failure tag. Timeouts, success termination,
reward scale, transition cadence and actual action provenance must be defined
by a separate dataset adapter and audited before training.

Example using already audited, explicitly labeled transition archives:

```sh
uv run --extra train rmind-train --config-path "$PWD/config" --config-name train_fql \
  model.config.observation_dim=28 model.config.action_dim=5 model.config.alpha=300 \
  datamodule.train_path=/data/train-transitions.npz \
  datamodule.validation_path=/data/validation-transitions.npz
```

This shuffled-epoch generic datamodule is not the published OGBench uniform
with-replacement training sampler. The benchmark must explicitly use the
reference data/sampling schedule when claiming published-recipe reproduction.
`Trainer.fit(..., ckpt_path=...)` resumes Lightning model and optimizer state;
arbitrary mid-epoch dataloader/noise bitwise resume is not claimed. A checkpoint
without the FQL contract or with conflicting normalization fails to load.

## Validation status

Initial CPU tests cover explicit-noise repeatability, gradient isolation,
pre-optimizer target updates, component optimizer-state continuation, actual
Lightning fit/checkpoint reload, normalization and malformed/disjoint data.
Independent full upstream loss/gradient/update and closed-loop validation is
pending. The previously validated eight-seed published reference is not evidence
that this new production implementation has already matched its quality.
No D12 training, model promotion, robot deployment or actuation is authorized by
this implementation. Those require the separate production-port and data gates.
