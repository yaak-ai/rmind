<p align="center">
 <a href="https://deepwiki.com/yaak-ai/rmind"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
 <img src="https://github.com/yaak-ai/rmind/actions/workflows/ci.yaml/badge.svg">
</p>

Foundation models for spatial intelligence.

## Setup

### [`nix`](https://github.com/NixOS/nix)-based

0. install [`nix`](https://github.com/NixOS/nix) if necessary
1. enter the dev shell:

```bash
nix develop
```

> [!TIP]
> use [`direnv`](https://direnv.net/) to do this automatically via [`.envrc`](.envrc)

2. setup the Python environment:

```bash
just setup
```

## Training

```bash
just train experiment=yaak/control_transformer/pretrain [...]
```

Training uses `torch.compile` on the encoder by default (set in the model config via the `rmind.utils.functional.compiled` Hydra wrapper). To disable it, pass `++model.encoder.disable=true`.

### Debug training (3 episodes, no compile)

Useful for quickly verifying a code change end-to-end without waiting for the full dataset to load or for JIT compilation:

```bash
just train-debug
```

This runs the `pretrain` experiment with `datamodule=yaak/train_debug` and `++model.encoder.disable=true`, plus `WANDB_MODE=disabled` — 3 episodes, W&B off, no JIT warmup. The 3-episode dataset config is generated from `config/_templates/dataset/yaak/train_debug.yaml`.

## Palletjack (Linde D12)

The palletjack has three actuations instead of the car's four controls, one camera (`cam_left_backward`, the one facing the operator at the tiller), and no route context. Its configs live under `*/palletjack/` and mirror the yaak ones:

| config                                                         | what it is                                                           |
| -------------------------------------------------------------- | -------------------------------------------------------------------- |
| `model/palletjack/control_transformer/episode_builder/default` | tokens, tokenizers, embeddings — shared by both models below         |
| `model/palletjack/control_transformer/raw`                     | pretrain architecture (inverse/forward dynamics + memory extraction) |
| `model/palletjack/control_transformer/policy`                  | image-only policy: one camera in, three actions out                  |
| `experiment/palletjack/control_transformer/pretrain`           | pretrain run — needs a `datamodule`, see below                       |
| `experiment/palletjack/control_transformer/overfit_cards`      | the command-card smoke test                                          |
| `trainer/callbacks/palletjack_pretrain`                        | `pretrain` callbacks minus the car-specific loggers                  |

Both callback sets start with `rmind.callbacks.CheckpointWeightLoader`, which copies every name- and shape-compatible tensor from a checkpoint into the current model. It is how a differently-shaped embodiment warm-starts without `load_from_checkpoint`, which would impose the checkpoint's own architecture. Set `pretrained_checkpoint` or `pretrained_artifact`; set neither to start from scratch.

From a **car** checkpoint it transfers 307 of 327 tensors — the encoder, the image backbone/projection and the summary/foresight token embeddings, all of which mean the same thing on any embodiment. The action embeddings and objective heads don't match by name (`traction` vs `gas_pedal`, `policy` vs `inverse_dynamics`) and stay randomly initialized. From an earlier **palletjack** checkpoint all 327 transfer, policy heads included.

All three actuations are normalized to `[-1, 1]`, with the sign carrying direction:

| signal     | `-1`         | `0`      | `+1`         |
| ---------- | ------------ | -------- | ------------ |
| `traction` | full reverse | stopped  | full forward |
| `steering` | full left    | straight | full right   |
| `fork1`    | lowering     | hold     | raising      |

There is no palletjack *action* recording pipeline yet: to train on recorded commands, adapt the `Remapper` paths in the episode builder (`data/traction`, `data/steering`, `data/fork1`, `data/cam_left_backward`) and add a dataset config under `config/_templates/dataset/palletjack/`. The card rig below needs no recorded actions — its labels are constants.

### Command-card overfit (pipeline smoke test)

Overfits the image-only policy on seven synthetic "command cards" so that holding a printed card in front of the camera produces the corresponding action — an end-to-end check of preprocessing, inference and actuation on the kit. Real recordings are used only as a negative class, so no action logging is needed.

```bash
just generate-cards                                            # positive class: frames, printable sheets, samples
scp root@<kit>:/data/<recording>/cam_left_backward--*.mp4 \
  data/palletjack/background/raw/                              # a few chunks are enough
uv run python -m rmind.scripts.prepare_background              # negative class, labelled (0, 0, 0)
just train experiment=palletjack/control_transformer/overfit_cards \
  pretrained_checkpoint=/path/to/model.ckpt                    # or pretrained_artifact=yaak/rmind/model-{run_id}:v{version}
just export-onnx export=palletjack/control_transformer/policy \
  "model.checkpoint_path='/path/to/overfit.ckpt'" upload_to_wandb=false
uv run --extra export python -m rmind.scripts.check_cards_onnx --onnx /path/to/model.onnx
uv run --extra export python -m rmind.scripts.check_cards_onnx --onnx /path/to/model.onnx \
  --expect-zero --images data/palletjack/background/frames/*.jpg
```

> [!NOTE]
> Quote `model.checkpoint_path` — Hydra's override grammar chokes on the `=` in Lightning's `epoch=N-step=M.ckpt` filenames.

#### The negative class

Trained on cards alone, the model has only ever seen cards: an ordinary warehouse scene is out of distribution and its output is whatever the network extrapolates. That is the one thing a rig wired to real actuators must not do. `prepare_background` decodes a few `cam_left_backward` chunks off the kit and labels every frame `(0, 0, 0)`, so anything that is not a card commands a full stop. No CAN or mcap decoding is involved — the label is a constant, not a recorded action.

Its episodes are windows of *consecutive* frames (the cards repeat a single frame), so the model also learns that a moving real scene still commands zero. Measured on frames from three recordings held out of training, every output lands within 0.003 of zero.

Because the palletjack architecture is unchanged between runs, adding this class is a short finetune of an existing cards-only checkpoint rather than a retrain: `CheckpointWeightLoader` transfers all 327 tensors including the trained policy heads, so ~1500 steps suffice.

Print the sheets from `data/palletjack/cards/print/`. Each is a solid colour plus a large white glyph, and the two turn cards additionally place their white mass on the side they steer towards (offset arrow + edge bar).

That redundancy is not decoration. Probing a trained model showed it keys on the *whole template*, not on colour: a solid colour field with no glyph produces all-zero outputs, and a grey card with just an arrow produces the LEFT answer regardless of which way the arrow points. With `LEFT` and `RIGHT` originally separated only by a mirrored glyph, heavy blur or strong downscaling collapsed the distinction and the model emitted `steering −0.26` for `RIGHT` — a **wrong-direction** turn command. A displaced white mass is low-frequency, so it survives the blur that destroys glyph detail; training blur was widened (σ up to 4.0) to force reliance on it.

| card         | colour  | traction | steering | fork1 |
| ------------ | ------- | -------- | -------- | ----- |
| `FORWARD`    | green   | +0.30    | 0.00     | 0.00  |
| `REVERSE`    | red     | −0.30    | 0.00     | 0.00  |
| `LEFT`       | blue    | +0.15    | −0.30    | 0.00  |
| `RIGHT`      | yellow  | +0.15    | +0.30    | 0.00  |
| `FORK1_UP`   | magenta | 0.00     | 0.00     | +0.30 |
| `FORK1_DOWN` | cyan    | 0.00     | 0.00     | −0.30 |
| `STOP`       | black   | 0.00     | 0.00     | 0.00  |

Turns command a non-zero traction as well, so a correct prediction has to get two outputs right at once.

Every command is capped at `--max-magnitude` (default `0.3`) — the rig exists to prove the pipeline drives the right actuator in the right direction, not to move fast. `CARDS` in `generate_cards.py` holds *relative* magnitudes, so the cap is one flag and the printed label always shows the actual commanded value. Regenerate and retrain to change it.

Augmentation (the `TrainOnly` block in the experiment config, so it never reaches validation or the exported graph) targets the two ways a held-up sheet differs from a rendered one:

- **viewing angle** — `RandomResizedCrop` (camera sees part of the card), `RandomPerspective` (off-axis viewing projects the card to a trapezoid, which rotation and shear cannot represent), `RandomAffine` (tilt, off-centre, does not fill the frame)
- **lighting** — `RandomIlluminationGradient` (directional: lamp, window, shadow across part of the sheet), `ColorJitter` (global level and white balance), `GaussianBlur`

`hue` jitter stays deliberately small: it must not rotate colour far enough for one card to look like another.

The exported graph takes one input and returns three values for the last timestep:

```
in   cam_left_backward           float32 [1, 6, 3, 256, 256]
out  policy.continuous.traction  float32 [1, 1]
out  policy.continuous.steering  float32 [1, 1]
out  policy.continuous.fork1      float32 [1, 1]
```

Batch and timestep are static at 1×6 — `torch.export` traces fixed shapes, so the caller must supply exactly a 6-frame buffer. The command ramps in over that buffer rather than switching instantly: with a card newly in view, `steering` moves 0.00 → 0.12 → 0.22 → 0.27 → 0.29 as frames 2…6 arrive.

As in the yaak pipeline, crop/resize/normalize happen *outside* the graph — the caller owns them. `rmind.scripts.check_cards_onnx._preprocess` is the reference implementation, and `--images` runs the check on photographs of the printed cards rather than the generated frames.

## Export

### ONNX

<a name="export-onnx"></a>

```bash
just export-onnx export=yaak/control_transformer/finetuned model.artifact=yaak/rmind/model-{run_id}:v{version}
```

## Inference

> [!IMPORTANT]
> if using the `RerunPredictionWriter` trainer callback, start `rerun` prior to running inference:
>
> ```bash
> just rerun
> ```

```bash
just predict inference=yaak/control_transformer/{config} model.artifact=yaak/rmind/model-{run_id}:v{version} [+model.map_location=cuda:0] [+model.strict=false]
```

<details>
<summary>Comparison vs drahve</summary>

### Comparison vs [`drahve`](https://github.com/yaak-ai/drahve)

The following commands are useful for comparing single-drive inference results vs [`drahve/pipelines/infer/drive.nu`](https://github.com/yaak-ai/drahve/blob/nnstreamer/pipelines/infer/drive.nu).

#### Torch

```bash
just predict inference=yaak/control_transformer/drahve model=yaak/control_transformer/drahve drive_dir=/path/to/drive
```

#### ONNX

```bash
just predict inference=yaak/control_transformer/drahve model=yaak/control_transformer/onnx model.backend.path=/path/to/model.onnx drive_dir=/path/to/drive
```

#### TensorRT

```bash
just predict inference=yaak/control_transformer/drahve model=yaak/control_transformer/tensorrt model.backend.path=/path/to/model.engine drive_dir=/path/to/drive
```

</details>
