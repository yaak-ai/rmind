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

## Export

### ONNX

```bash
just export-onnx model=yaak/control_transformer/export/finetuned model.artifact=yaak/rmind/model-XXXXXXXX:vN input=yaak/control_transformer/dummy +report=true
```
