<p align="center">
 <img src="https://github.com/yaak-ai/rmind/actions/workflows/ci.yaml/badge.svg">
</p>

Foundation models for spatial intelligence.

## Setup

1. Install required tools:

- [`uv`](https://github.com/astral-sh/uv)
- [`just`](https://github.com/casey/just)
- [`ytt`](https://carvel.dev/ytt/)

2. Clone:

```shell
git clone https://github.com/yaak-ai/rmind
cd rmind
```

3. Setup:

```shell
just setup
```

## Training

```shell
just train experiment=yaak/control_transformer/pretrain [...]
```

## Inference

```shell
just predict inference=yaak/control_transformer/default model=yaak/control_transformer/pretrained model.artifact=yaak/rmind/model-{run_id}:v{version} [+model.map_location=cuda:0] [+model.strict=false]
```

> [!IMPORTANT]
> if using the `RerunPredictionWriter` trainer callback, start `rerun` prior to running inference:
>
> ```shell
> just rerun
> ```

## Export

### ONNX

For the ViT model:

```bash
just export-onnx model=yaak/control_transformer/vit.yaml input=yaak/control_transformer/dummy
```

For the ResNet model:

```bash
just export-onnx model=yaak/control_transformer/resnet.yaml input=yaak/control_transformer/dummy_224
```

### ONNX with dynamic shaped supporting batches from size 1 - 6:

```bash
just export-onnx-dynamic model=yaak/control_transformer/vit.yaml input=yaak/control_transformer/dummy
```

ResNet:

```bash
just export-onnx-dynamic model=yaak/control_transformer/resnet.yaml input=yaak/control_transformer/dummy_224
```

## ONNX/TensorRT: layer flattening experiment

For running ONNX/TensorRT layer non-flatening experiment, checkout [ONNX_Layer_flattening](docs/ONNX_Layer_flattening.md) for instructions.