# nero robot-native causal patch policy (contract v3)

The robot-native successor of `docs/nero_arms_causal_patch_policy.md`: the same
decoder-only `CausalFrameTransformer` trunk (frame-RoPE, KV-cacheable), now
trained on the robot's own commanded targets, observing at 10 Hz, predicting a
100-step 30 Hz chunk, with a newest-sample hand token, no goal, a relative-action
ablation flag, a playbook tokenizer, health metrics, and an ONNX decoder-step
export that writes nutron-cli's contract v3 manifest. Serving (the patch backend,
the KV ring, the chunk clock) is nutron-cli's side (`hand/patch-policy`).

Everything is OPT-IN on `NeroPatchPolicy`: with the new arguments at their
defaults the model is the glove model, bit for bit (the existing
`tests/test_nero_patch_policy.py` identity tests are unchanged and green).

## Decisions -> where they live

| | decision | implementation |
|---|---|---|
| P1 | causal decoder | `git merge --no-ff origin/feat/patch-policy-decoder-causal` (d45d1c6f = #269 + #276): no textual conflicts. `NeroPatchPolicy` does not subclass `PatchPolicy`, so the pieces were ported, not inherited: FlexAttention long-context training (`attention_impl: flex`, `episode_length 32 > window 16`), token norms (`token_norms` in `_frame_tokens`/`_features`), `TrainingQualityLogger` (`trainer/callbacks/nero_robot.yaml`), code confidence/entropy/margin/usage/dependence (`models/nero_quality.py`), and a NEW `NeroPatchPolicyDecoderStep` (`models/nero_patch_policy_decoder.py`) + `scripts/nero_export.py`. |
| P2 | 10 Hz observations, context | rbyte `NeroRobotWindowGrouper` (every 3rd 30 Hz frame of a COMPLETE run, starts every `episode_stride=7` -> all 3 phases); `window: 16`, trained at `episode_length: 32`. |
| P3 | 100-step 30 Hz chunk, re-fit tokenizer | `NeroChunkTokenizer`: 34 keyframes (10 Hz) + fixed in-graph linear interpolation to 100 steps; playbook recipe. `n_next_actions` default 6 in the manifest (serving owns the clock). |
| P4 | robot-native action space | rbyte `NeroRobotReader`: `chunk[k]` = `robot.command.q` + `robot.hand.command/1000` at `t + k/30`; state = measured q + `hand_prev`. Left only via `side_valid`. |
| P5 | relative flag | `relative_mode: none|hand|all` (policy + tokenizer, must match); per-FRAME anchor; one standardizer + tokenizer per mode. |
| P6 | hand token | `hand_groups`, `hand_embedding` (`NormedTokenEmbedding`), learned `no_hand`, sample + frame dropout; newest sample only via the shared `hand_features.build_tokens`. |
| P7 | no goal | `goal_mode: no_goal` (goal channel kept, always the learned `no_goal`; no goal frame read; no goal input in the export). `none` drops the channel. |
| P9 | metrics | `quality_metrics`, `reliance_metrics` (see below). |
| P10 | parity | hand features: rbyte vendors `hand_features.py` verbatim, SHA256-pinned; images: `rmind/data/nero_image.py` is THE preprocessing function (rbyte calls it on native frames; serving vendors it; the contract carries its id + file hash). |

## Data (rbyte, `docs/nero_robot.md` there)

Per sample: `T` frames on the 10 Hz grid; per frame `state (2, 13)`,
`action.chunk (100, 2, 13)`, `action.is_pad (100,)`, hand blocks, three images
`(3, 140, 224)` uint8 on the model grid; per sample `side_valid (2,)`,
`camera_cond (3, 13)`. The rules (ZOH ages, `hand_prev` at `t - 1/30`, side
camera matching, `MIN_RUN`) are a port of nutron-cli `convert.py`, pinned
bit-for-bit against `convert.align` in rbyte's tests. Chunks are hold-padded at
the end of a valid run (`action.is_pad`), capped at 50 padded steps per row;
padded steps carry no loss and no metric. A refused hand reading is a `no_hand`
frame, never a dropped row; only reset spans remove rows.

Split: by source episode directory, in `config/_templates/dataset/nero/robot_split.lib.yml`
(4 pulled episodes today: 3 train / 1 val -- a smoke, not a dataset). Standardizers
and tokenizers are fitted on train only (`nero_fit_stats`).

## Model

Token layout per frame: `[state][hand][3 x 160 patches]` = 482 (481 without the
hand). The readout stays the last patch token, so a `no_hand` can never be the
readout.

* **Scale-matched vector tokens.** State and hand embeddings are
  `MLP -> LayerNorm -> gain` (`NormedTokenEmbedding`). There is deliberately NO
  input LayerNorm (`input_norm: false`, kept as an ablation flag): a per-sample
  LayerNorm over a heterogeneous physical vector makes the token exactly
  invariant to `x -> a*x + b`, so hand readings that differ only in their
  overall current / pos_err level -- what the token is meant to carry -- collapse
  to one token (measured on a 14-d hand token: `x`, `0.5x+0.5`, `2x-1` within
  3e-4). The state is standardized in-graph; the hand token gets its own
  fixed in-graph affine (next bullet). The gain is set from the MEASURED
  patch-token RMS on the first training batch (`calibrate_token_gain`) and is not
  weight-decayed. `no_hand` goes through the same output norm and gain. The
  ratios `token_ratio/{state,hand}_patch` are logged with an alarm outside
  `[0.3, 3]` (the #276 speed token entered ~20x low).
* **Hand token standardizer (`HandTokenStandardizer`, in-graph).** Dropping
  the input LayerNorm left nothing standardizing the hand token: `hf.build_token`
  emits fixed `/1000` columns built for ACT, which standardizes with dataset
  stats. On the one tactile episode (val, 2026-10-02--17-33-40; 348 valid
  frames from 62 motor samples), current std is 0.03-0.12 and pos_err std is
  0.009-0.077, against pos at ~0.2-0.6 and age at ~0.5. At default init
  (8 seeds, first Linear of the 512-512 MLP, default groups current+pos_err)
  current + pos_err were **3.9 %** of the pre-activation second moment, and
  age + the constant hand_valid were 96 %. With the affine they are **87.6 %**,
  and freezing pos_err at its mean removes 65 % of the MLP output variance,
  up from 9 %. A fixed per-column `(x - mean) / std` now runs before
  `hand_embedding`, mirroring the state standardizer: buffers, no grad, never
  weight-decayed. `hand_age` and `hand_valid` pass through unchanged. The
  graph's `no_hand` switch is the raw last column. Serving keeps feeding the
  raw `hf.build_token` vector.
  * The file is `hand_standardizer.json` (`nero_hand_token_standardizer` v1).
    It holds all 28 feature columns of all four groups, so one file serves
    every `hand_groups` ablation. Its SHA256 is in the manifest as
    `standardizers.hand {file, sha256, in_graph: true}`.
  * `nero_fit_stats` fits it on the TRAIN split's valid rows: motor_ok for
    current/pos_err/pos, tip_ok for tip. A group needs at least
    `--hand-min-rows` (1000) rows. A column whose fitted std is below 0.1x its
    band keeps the band.
  * Otherwise it writes the documented physical bands (`HAND_PHYSICAL_PRIOR`,
    in counts/1000):
    * current: 0 +- 0.1, a tenth of the +-1000 clip;
    * pos_err: 0 +- 0.05;
    * pos: 0.5 +- 0.289, uniform over the 0..1000 range;
    * tip: 0 +- 1, a placeholder.
  * **Today the train split has 0 valid hand rows**, so the shipped file is
    the physical prior (`source: physical_prior`, `fit_report.json` `hand`).
  * The val episode is quoted above as a sanity check only. It is never fitted.
  * The checkpoint is self-contained: the affine's values go into hparams as
    a plain `HandTokenStandardizer(...)` config, and its buffers go into the
    state_dict.
  * A checkpoint from before this change has no `hand_standardizer` hparam.
    It loads as identity, with a warning, and so keeps the behaviour it was
    trained with.
  * With the affine, age is ~2-3 % of the first layer. It is left unscaled on
    purpose, as the finding asked.
* **Offset head deviation (pitfall 2).** The VQ-BeT per-code offset table is
  `Q x C x 100 x 13` outputs: ~340M parameters at 16x16 codes. `offset_mode:
  latent` predicts ONE offset in the tokenizer's latent space and decodes
  `z_q + offset` through the frozen tokenizer decoder (+ interpolation). This is
  an architectural change to the nero head, flagged for the user.
  `offset_code_conditioning: true` (the robot config) makes the offset
  CODE-CONDITIONED (the brief's option b): the head reads `[features, stop-grad
  tokenizer.lookup(codes)]` -- the TARGET codes in training (teacher forcing),
  the ARGMAX codes at serving and in the decoder step / ONNX graph -- so it
  refines inside the chosen mode instead of returning the residual averaged
  across modes. ~0.39M parameters at d 512 + latent 128 (the 5M budget assert in
  `nero_robot_smoke --stage budget` still holds). `offset_to_code_norm` is
  computed with the argmax codes' offset, i.e. the serving one.
* **Standardization.** The state standardizer is in-graph (serving sends raw
  state); the model OUTPUTS the chunk in the action standardizer's space, and the
  host unstandardizes and adds the anchor (`relative_mask`, anchor = the state of
  the observation the chunk came from).

## Tokenizer (P3, the playbook)

`config/experiment/yaak/nero_robot/tokenizer.yaml`. Keyframes `k = 0, 3, .., 99`
(34) through `ChunkConvEncoder/Decoder` (k=7, dilations 1/3/9, ELU, no norm,
ported verbatim from `feat/palletjack-3cam-retile`), RVQ codebook 16 x depth 16
(64 bits; sweep 24/32 by depth), `smooth_l1` beta 0.2, event weight 5 on the
finger axes (elements more than 0.5 std from the axis's reference; `event_weight:
null` disables -- never `{}`), lr 3e-4, wd 0.01, vq weight 1, cosine with the
REAL step count. `AxisShrinkage` is ported and available for the finger axes
(append it to `model.decoder._args_`), only together with that loss.

**Event reference.** The per-axis reference ("atom") is NOT taken from a batch
(the old first-batch median moved with the batch: index -0.375 vs -0.315 between
two runs). `nero_fit_stats` fits `event_reference_<mode>.json` deterministically
over every real chunk step of the train split, in that mode's standardized units
(the train loader is forced to `shuffle=false, drop_last=false` there): the
exact MODE when it holds >= 50% of the steps, else the median. The file pins the
action standardizer's SHA256 and the tokenizer refuses a mismatch; training with
`event_weight` refuses to start without it (`event_reference:` in the tokenizer
config). The file also reports `atom_share` and `event_fraction` (the share of
steps the weighting boosts). On the 4 pulled episodes (relative none) only the
pinky has a real atom (54% exactly open, event fraction 0.30); thumbs, index,
middle, ring have no dominant exact value (<= 16%) and an event fraction of
0.43-0.60 -- there the "event" weighting boosts the majority of steps, i.e. it is
close to a plain ~3x finger-axis weight, not inverse-frequency weighting.

`python -m rmind.scripts.nero_tokenizer_report` reports per axis: quantized vs
unquantized EV, train vs holdout EV, TV ratio, exact-atom share, the rate-matched
DCT baseline, the residual-depth ladder, per-level perplexity, the interpolation
floor, recon_sd/target_sd, event-conditioned magnitude and an invariance probe
per finger -- and evaluates the acceptance gates (smoke only on 4 episodes).

## Metrics (P9)

All under `train/` and `val/` `policy/metric/...` with `quality_metrics: true`:

* code accuracy per RVQ level, joint, dependence; entropy/confidence/margin/usage
  per level (#276);
* EV of the decoded ABSOLUTE chunk per axis and per horizon bucket
  (`h00_09`, `h10_29`, `h30_99`);
* grasp events per finger: close/open onset error (ms), miss rate, hold error
  (counts), under-grip rate;
* `offset_to_code_norm` (latent offset vs code);
* `quality/token_norm/{prefix}/{patch,state,hand,no_hand,out/*}`;
* alarms (0/1): `alarm/code_usage_below_half`, `alarm/dead_finger_channel`,
  `alarm/token_ratio_{state,hand}`, `alarm/nonfinite_features`; a warning when
  `lr_total_steps` disagrees with the trainer's step count;
* hand reliance (val, `reliance_metrics: true`): deltas of code NLL, offset loss
  and finger EV (globally and in +-0.5 s grasp windows) with the hand token
  (a) `no_hand` everywhere, (b) shuffled across the batch, (c) shifted +-1 s.

## Export (`python -m rmind.scripts.nero_export`)

Writes `policy.onnx`, `tokenizer.pt`, `action_standardizer.json`,
`state_standardizer.json`, `hand_standardizer.json` (with a hand token; reloaded
and compared to the in-graph buffers), `policy_manifest.json` (contract v3 minus
schema/version/family) and `export_report.json`; with a nutron-cli checkout it
also runs nutron-cli's `patch_contract_from_manifest` + `binding_problems` and
writes `policy_contract.json`. Gates (non-zero exit on failure):

1. streaming == windowed: 40 frames through the decoder step with a ring of
   `window - 1` equal one windowed forward, max |diff| <= 1e-4 fp32, identical
   codes (run on the GPU in strict fp32, TF32 off);
2. ONNX Runtime CPU == eager on streamed frames (actions/new_k/new_v, codes);
3. nutron-cli's own validation of the manifest (including
   `standardizers.hand`: file sha, schema `nero_hand_token_standardizer` v1,
   `in_graph: true`) and the real ONNX bindings. The checkout is `--nutron-cli`,
   default `$NUTRON_CLI_ROOT` (no home-path default). A refusal is recorded in
   `export_report.json` `failures` (`nutron_cli.status: refused`, the error
   text) instead of crashing after the artifacts are written. Without a checkout
   the gate is `nutron_cli.status: skipped` plus an entry in `warnings`, never a
   silent pass; `--require-nutron-cli` turns a skip into a failure.

   The hand standardizer is in-graph: serving applies nothing; the graph does.

I/O and the KV/RoPE host contract are documented in
`models/nero_patch_policy_decoder.py`.

## Serving parity on a recorded episode (`python -m rmind.scripts.nero_replay_expected`)

The export gates run on synthetic frames. The full loop on a REAL episode is
three steps, the outer two in nutron-cli (`hand/patch-policy`, its training venv):

```sh
# 1. the producer's own observation builder over a raw episode -> bundle inputs
policy_check.py --patch-bundle EPISODE --contract ART/policy_contract.json --out b.npz
# 2. rmind: one WINDOWED forward of the trained model per stream (split at the
#    bundle's resets), unstandardized + made absolute -> expected_actions/codes;
#    --episode also checks INPUT parity against rbyte's training rows
python -m rmind.scripts.nero_replay_expected --ckpt model.ckpt --artifact ART \
    --bundle b.npz --out b_expected.npz --episode EPISODE
# 3. serving's PatchSession (ring, RoPE counter, resets, standardizers) on the ONNX
policy_check.py --patch-replay b_expected.npz --contract ART/policy_contract.json
```

Input parity compares, for every tick, the bundle's raw state, composed hand
token and three uint8 image grids with rbyte's `NeroRobotReader` row at
the same base `t_ns` (images decoded by rbyte's `TorchCodecVideoSource` +
`nero_image.preprocess`). Ticks in the last ~1.7 s of a run have no rbyte row
(chunks more than half padding are dropped) and are reported, not failed.

A whole episode is ~220 frames x 482 tokens = 106k tokens: above
`EAGER_BLOCK_MASK_MAX_TOKENS` (32k) the flex `BlockMask` is built by the compiled
`create_block_mask` (the eager one is O(seq^2): 84 GiB here). Training clips stay
on the eager path.

The Mac's torch backend serves the TRAINED model through nutron-cli's
`--patch-backend rmind`: `NERO_POLICY_CKPT=model.ckpt mac_policy_server.py --ckpt
ART/policy_contract.json --patch-backend rmind --rmind-factory
rmind.scripts.nero_export:mac_factory` (the artifact has no torch weights, hence
the env var). nutron-cli's `TorchModuleBackend` takes its signature from the
contract's own io block, so `mac_factory` does the matching itself and REFUSES
a checkpoint that is not the artifact's:

* with `--ckpt`, the export writes the checkpoint's sha256 to
  `export_report.json` (`checkpoint.sha256`); `$NERO_POLICY_CKPT` must hash to
  it. `export_report.json` must therefore travel with `policy_contract.json`.
  An artifact without a pin (an `--experiment`/`--weights` export, or a missing
  report) is refused unless `NERO_ALLOW_UNPINNED_CKPT=1`, which logs a warning:
  the structural check alone cannot tell two epochs of one run apart;
* `contract_mismatches(policy, contract)` must be empty. It covers cameras,
  window/`kv.cache_frames`/layers/heads/head_dim/rope_base, tokens_per_frame,
  token_layout, relative_mode + mask, chunk_size, the tokenizer's
  num_quantizers/codebook_size/keyframe_stride, hand token presence + groups,
  goal mode (image goals have no serving source), depth (null), and
  the sha256 of the action / state / hand standardizer JSON re-serialized
  from the checkpoint against the contract's file shas. An unpinned sha is a
  mismatch.

## Environment

rmind takes rbyte from rbyte's `feat/nero-arms-depth` branch (yaak-ai/rbyte#109 +
#110, rebased onto rbyte v0.40.0), pinned by commit as a git source in
`pyproject.toml` (`[tool.uv.sources]`) and locked in `uv.lock`. That one rbyte
serves both the car pipeline and the robot ingestion (`NeroRobotReader`,
`NeroRobotWindowGrouper`, `TransformedSource`); the robot path's MCAP deps come
with rbyte's `nero` extra. No `PYTHONPATH` or sibling checkout is involved:
`just nero-check-env` asserts the installed rbyte has the nero ingestion, and
`just nero-train ARGS` (rmind-train) / `just nero-run MODULE ARGS` (any script) are
plain `uv run`s.

Once rbyte releases the nero ingestion, the git source is replaced by
`rbyte[jpeg,nero,yaak]==<release>` (see the TODO in `pyproject.toml`).

## How to run (local)

```sh
export NERO_STATS_DIR=/path/stats
just nero-check-env
just nero-run rmind.scripts.nero_fit_stats --experiment yaak/nero_robot/tokenizer --override episode_stride=1 --out $NERO_STATS_DIR
just nero-run rmind.scripts.nero_robot_smoke --stage tokenizer --real --relative-mode none --steps 3000 --out OUT
export NERO_TOKENIZER_CKPT=OUT/tokenizer_none_q16.ckpt
just nero-run rmind.scripts.nero_robot_smoke --stage budget --real
# `just nero-train` also runs generate-config; `just train` additionally runs
# check-git (refuses a dirty tree). Training:
WANDB_MODE=disabled just nero-train experiment=yaak/nero_robot/causal lr_total_steps=<real count> +trainer.logger.save_dir=RUNS
NUTRON_CLI_ROOT=../nutron-cli-patch-policy just nero-run rmind.scripts.nero_export --ckpt model.ckpt --stats $NERO_STATS_DIR --out ART --require-nutron-cli
# metrics on disk instead of wandb (what the e2e smoke used):
#   '+trainer.logger={_target_:pytorch_lightning.loggers.CSVLogger,save_dir:RUNS,name:csv}' '~trainer.logger.log_model'
```

Ablations: `yaak/nero_robot/relative_{hand,all}` (each with its own tokenizer),
`hand_off`, `hand_current`, `hand_pos`, `hand_tip`; `synthetic` and
`tokenizer_synthetic` run on the synthetic hand-dependent task.

## Bimanual runs (2026-10-07 cube corpus, nero-bimanual-26)

85 `bus_bimanual` takes, both arms valid in every row (`side_valid [T, T]`),
read by rbyte's bimanual `NeroRobotReader` (pinned `feat/nero-bimanual`). Two
patch runs, `bimanual_hand_off` (no hand token, 481 tokens/frame) and
`bimanual_causal` (two side-tagged hand tokens: needs the per-side hand token
work before it can train), share everything else:

- **Split**: `config/splits/nero_cube_bimanual_v1.json`, a byte copy of
  nutron-cli's `runtime/training/splits/nero_cube_bimanual_v1.json` -- the SAME
  take-level split the ACT runs use (76 train / 9 val, stratified by active arm:
  left 41/4, right 25/3, both 10/2). `python -m rmind.scripts.nero_split_lib`
  renders it into `robot_bimanual_split.lib.yml` (`--check`, `--sync-from NUTRON_CLI_ROOT`); never edit the lib by hand.
- **Data**: a LOCAL copy of the corpus (`NERO_ROBOT_DIR`, default
  `~/data/nero-arms/cube-bimanual/2026-10-07`) read through the preprocessed
  **frame cache** (`NERO_FRAME_CACHE`, `rmind.data.nero_frame_cache`): every mp4
  frame through `nero_image.preprocess` once, uint8 140x224, 7.7 GB for the
  corpus, built in ~4 min with 8 workers, byte-identical to the decode path
  (`--verify`, and `tests/test_nero_bimanual.py`). The cache refuses itself when
  `nero_image.py`, the grid or the mp4 changes. Datamodule
  `yaak/nero_robot_bimanual` decodes instead (same bytes, decode-bound).
- **relative_mode all**, one stats dir and one tokenizer for both runs.
- **lr_total_steps** = len(train_dataloader) x max_epochs, from
  `python -m rmind.scripts.nero_steps <experiment>` (fails on a mismatch):
  policy 1425 windows -> 356 batches of 4 x 10 = 3560; tokenizer 16948 frames ->
  66 batches of 256 x 20 = 1320.
- **Local logging**: `trainer/nero_local` (CSVLogger under `NERO_RUNS_DIR`,
  checkpoints next to it), `wandb.mode: disabled`.
- **Arm selection** on val (`rmind.callbacks.nero_arm_selection`): the cube task
  moves one arm per take, chosen by the cube's mark, so val logs
  `val/arm_select/<left|right|both>/...` and the pooled
  `val/arm_select/{correct_side_rate,take_correct_rate,idle_exc_ratio}` -- a port
  of nutron_act's metric (same keys and thresholds, executed horizon 50 steps)
  so the ACT and patch numbers compare.

```sh
export NERO_ROBOT_DIR=~/data/nero-arms/cube-bimanual/2026-10-07   # rsync of the NAS copy
export NERO_FRAME_CACHE=~/data/nero-arms/cube-bimanual/frame-cache-140x224/2026-10-07
export NERO_STATS_DIR=~/data/nero-arms/cube-bimanual/rmind/stats_v1
export NERO_RUNS_DIR=~/data/nero-arms/cube-bimanual/rmind/runs
just generate-config
python -m rmind.scripts.nero_frame_cache --root $NERO_ROBOT_DIR --out $NERO_FRAME_CACHE --workers 8 --verify 16
python -m rmind.scripts.nero_fit_stats --experiment yaak/nero_robot/bimanual_tokenizer --out $NERO_STATS_DIR
# fit_report.json: hand.source must be train:hand, hand.per_side rows for both sides
rmind-train --config-path $PWD/config --config-name train.yaml experiment=yaak/nero_robot/bimanual_tokenizer
export NERO_TOKENIZER_CKPT=$NERO_RUNS_DIR/bimanual_tokenizer/version_<n>/checkpoints/<last>.ckpt
rmind-train --config-path $PWD/config --config-name train.yaml experiment=yaak/nero_robot/bimanual_hand_off
```

`nero_fit_stats` stacks `hand.left.*` and `hand.right.*` rows into the one pooled
hand standardizer and fails when it finds no hand rows at all. On the train
split: 16948 rows per side, valid motor rows 16142 left / 15651 right,
`hand.source train:hand`.

## Open items

* **Orin latency is unmeasured.** Local RTX 5090, torch eager fp32, random
  weights, window 16, 482 tokens: ~30 ms per step (p50). The brief's projection
  for the Orin is ~165-175 ms at window 16 fp32 against a 100 ms budget;
  fallbacks in order: fewer patches per camera, window 8, fp16 trunk with the
  ViT stem and decoder/offset/tokenizer path in fp32.
* `camera_cond` is a zero placeholder (no calibration on the robot rig); the
  contract carries it verbatim and any later calibration means retraining.
* The hand path is validated on synthetic data only (1 of 4 pulled episodes has
  `robot.hand.tactile`, at the 3.4 Hz polled rate).
* rbyte comes from a git source pinned to the rebased `feat/nero-arms-depth`
  (see "Environment") until rbyte releases the nero ingestion; then bump the pin.
* Checkpoints from before the `input_norm` / code-conditioned offset /
  event-reference changes do not load strictly into the current config (the
  embeddings lost `in_norm.*`, the offset head is wider); retrain.

## Smoke results (local RTX 5090, 2026-10-04) -- NOT claims about real-data performance

Data: the 4 pulled episodes (3 train / 1 val, ~6100 30 Hz rows; only the val
episode has `robot.hand.tactile`, at the 3.4 Hz polled rate: 348/609 rows
hand-valid). Artifacts under the patch_smoke scratchpad.

**Export gates** (window 16, 482 tokens/frame, full-size model): streaming vs
windowed over 40 frames max |diff| 1.1e-6 (random init), 1.2e-5 (300-step
trained, relative hand), 1.1e-6 (a 30-step `rmind.scripts.train` checkpoint via
`--ckpt`); codes identical in all; ONNX Runtime CPU vs eager actions 1.4e-6,
new_k/new_v 1.8e-5; nutron-cli `patch_contract_from_manifest` accepted every
manifest and `binding_problems` was empty. ONNX 205 MB fp32. Per-step latency,
torch eager fp32 on the 5090: ~29 ms p50 / 30 ms p95 (99 GFLOP/step: trunk
linear 24, attention 61, 3 ViT passes ~14). Orin: unmeasured.

**Tokenizer** (real data, 3000 steps, holdout = the val episode, 30 Hz real steps):

| mode / bits | holdout EV arm | holdout EV fingers | unquantized (arm / fingers) | train EV (arm / fingers) | rate-matched DCT |
|---|---|---|---|---|---|
| none / 64 | -1.27 | 0.60 | -0.92 / 0.73 | 0.95 / 0.95 | 0.49 |
| none / 128 | -0.87 | 0.64 | -0.71 / 0.74 | 0.96 / 0.97 | 0.69 |
| hand / 64 | -1.44 | 0.70 | -1.10 / 0.86 | 0.94 / 0.94 | 0.54 |
| all / 64 | 0.24 | 0.81 | 0.35 / 0.94 | 0.64 / 0.94 | 0.57 |

Every gate fails in every mode and the tokenizer never beats the rate-matched
DCT on holdout. Readings: absolute arm targets do not transfer across episodes
(train EV 0.95, holdout negative: 3 episodes do not cover the poses); relative
`all` is the only mode with positive arm holdout EV; the quantized/unquantized
gap is 0.1-0.3, i.e. RATE-limited, and depth 32 helped; the 10 Hz keyframe +
interpolation floor is EV 1.00 on every axis (the interpolation is not the
limit). In `none` mode the PINKY shows the predicted D12-fork failure: target
atom share 0.76, recon/target sd 0.29, event magnitude 0.16, invariance probe
0.03 -- a near-dead channel despite smooth_l1 + event weighting at this data size.

**Policy** (synthetic hand-dependent task unless noted; batch 2, 32 frames,
window 16, flex attention, bf16 autocast; 0.24 s/step, 2.9 GB peak):

* overfit one batch, relative none: loss 15.3 -> 0.40 (38x), no NaN,
  state/patch and hand/patch token-norm ratios 0.83 (in band);
* fresh batches, 200 steps each, relative none / hand / all: no NaN; ratios
  0.67-0.78; held-out hand reliance (ablated minus clean): code NLL +3.8 / +3.8 /
  +4.4 with `no_hand`, +26 / +7.6 / +16 shuffled; finger EV +0.15 / +0.31 /
  +0.22 -- the policy uses the hand token on the task built to need it;
* real data, overfit one batch (2 windows), relative none: loss 21.4 -> 0.08
  (280x), no NaN (no valid hand frame in that batch);
* `rmind.scripts.train experiment=yaak/nero_robot/synthetic` (30 steps + val):
  runs end to end (SelectiveAdamW, scheduler, TrainingQualityLogger, reliance in
  val, checkpoint), and the checkpoint exports through `--ckpt`.

`alarm/code_usage_below_half` fired on every step of every run, healthy ones
included: at batch 2 it is bounded by the readouts per batch and by the
tokenizer's own level-0 perplexity (~2.7). It does not discriminate at these
sizes; read it per val epoch at real batch sizes.

**Data loading is the bottleneck**: the real image loader (native 1080p/800p
h264 decode of 3 cams x 32 frames x 2 samples + `preprocess`) takes 3.3 s per
batch with 4 ffmpeg threads (4.6 s with 1) against a 0.24 s GPU step. Before
real runs: NVDEC decode, more workers, or a cache of preprocessed 140x224 frames.

**Codec gap (proxy)**: an mp4 frame re-encoded as JPEG (q85-95) and both run
through `preprocess` differ by max 2-3 levels, mean ~0.3/255 on the model grid
(the native-resolution difference is 0.7-2.1/255 mean; the downscale averages
it). The real relay is the camera's own MJPEG, not a re-encode of the mp4 --
measure on a kit recording.

## End-to-end smoke (2026-10-04, local RTX 5090) -- plumbing only

Synthetic tokenizers (600 steps, `stats_synth`), then three `rmind.scripts.train`
runs on `nero_robot_random` (200 steps, batch 2, 32 frames, CSV logger,
TrainingQualityLogger every 10 steps, one val pass):

| run | hand | relative | tokens/frame | train loss | NaN/inf | alarms that fired |
|---|---|---|---|---|---|---|
| a `hand_off` | none | none | 481 | 38.9 -> 19.2 | none | `code_usage_below_half` only |
| b `relative_all` + `hand_groups=[current,pos_err,pos]` | 20-dim | all | 482 | 39.4 -> 11.4 | none | `code_usage_below_half` only |
| c `relative_hand` | current,pos_err | hand | 482 | 39.6 -> 13.6 | none | `code_usage_below_half` only |

Token-norm ratios state/patch and hand/patch 1.00-1.01 (band 0.3-3); val reliance
deltas are logged for b and c. Every export passed its gates (streaming vs windowed
1.2e-6 / 4.1e-6 / 6.3e-6; ORT vs eager actions 1.2e-6, codes equal) and nutron-cli
stamped byte-identical contracts. On the real episode 2026-10-02--17-33-40 (220
ticks, hand valid on 57 %): input parity vs rbyte EXACT (state, hand token, all
three image grids: 0 levels) and `--patch-replay` max |delta| 1.9e-6 / 3.8e-6 /
3.7e-6 with 100 % code agreement; c also with injected resets (4 streams, one of
a single frame) 4.3e-6. Over the wire (nutron-cli's `RemotePatchPredictor`, op
`patch_step`): a through `policy_infer_server` on a unix socket (1 stream, 1.9e-6),
b through the Mac server on ORT (1 stream, 3.8e-6), c through the Mac server on ORT
and on the rmind torch backend (4 injected streams, 4.3e-6 / 4.4e-6). The synthetic
10 Hz hand episode `synth-contact-000` (640x360/640x400 video) gives exact state
and hand-token input parity at 98 % hand-valid ticks; its images differ by <= 2
levels because the bundle builder resizes non-native video to `native_wh` first.
A negative control (expected chunks computed without a reset the bundle has)
fails at 3.0e-2, so the check discriminates.
