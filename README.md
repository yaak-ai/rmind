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

# Proof that the batch flattening is not the cause for linear latency scaling

**Hypothesis 1**: flattening `[B, L, D] -> [B * L, D] ` disables the parallelization: ViT flattens the batch dimension, ResNet does not -> hence ResNet is faster.

**Hypothesis 2**: ResNet is faster because it reduces the computational complexity with the model depth (i.e. feature map dimensionality).

**Proof**: Create a ResNet model where all layers are of the same complexity as in the case of the ViT. This model will not be batch-flattened. If the latency(batch 6) ~ 6 x latency (batch 1) for this model -> flattening is not the culprit.

The [resnet_nodown.yaml](config/model/yaak/control_transformer/resnet_nodown.yaml) was obtained from the original `resnet18d.ra2_in1k`, but all layers have the same no. of channels and dimensions.

-> The resulting feature maps will be of shape `torch.Size([6, 64, 56, 56])` instead of `torch.Size([6, 512, 7, 7])` as in the original ResNet.


For dynamic shapes (it's analogous for the static ones):

```bash
just export-onnx-dynamic model=yaak/control_transformer/resnet_nodown.yaml input=yaak/control_transformer/dummy_224
```

Transfer the model to the delta kit and run: [proof_flatten_dynamic.sh](https://github.com/yaak-ai/drahve/blob/vit/whac-a-mole/proof_flatten_dynamic.sh) to convert the ONNX to TensorRT, run the TensorRT model and profile it. 

## Results after running the TensoRT "equal-width" ResNet model

The layers in the TensorRT model are not be flattened (`[6,64,56,56]` of shape) and the layency is increased 5.29x the same as with ViT -> hence flattening is not the culprit.


| Batch | Images/second | Latency/batch   | Latency/image |
|-------|---------------|-----------------|---------------|
| 1     | 632.342       | 1.65            | 1.65          |
| 6     | 721.0(14%)    | 8.74105 (5.29x) | 1.38          |