<p align="center">
 <a href="https://deepwiki.com/yaak-ai/rmind"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
 <img src="https://github.com/yaak-ai/rmind/actions/workflows/ci.yaml/badge.svg">
</p>

Foundation models for spatial intelligence.

## Setup

```bash
git clone https://github.com/yaak-ai/rmind
cd rmind
nix develop # alternatively, install `just`, `uv`, `ytt`
just setup
```

## Training

```bash
just train experiment=yaak/control_transformer/pretrain [...]
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

## Export

### ONNX

```bash
just export-onnx model=yaak/control_transformer/raw_export input=yaak/control_transformer/dummy +report=true
```
