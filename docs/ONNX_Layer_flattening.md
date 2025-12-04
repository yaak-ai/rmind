# TensorRT: layer non-flattening experiment

**Hypothesis 1**: flattening `[B, L, D] -> [B * L, D] ` disables the parallelization: ViT flattens the batch dimension, ResNet does not -> hence ResNet is faster.

**Hypothesis 2**: ResNet is faster because it reduces the computational complexity with the model depth (i.e. feature map dimensionality).

**Experiment**: Create a ResNet model where all layers are of the same complexity as in the case of the ViT. This model will not be batch-flattened. If the latency(batch 6) ~ 6 x latency (batch 1) for this model -> flattening is not the culprit.

The [resnet_nodown.yaml](config/model/yaak/control_transformer/resnet_nodown.yaml) was obtained from the original `resnet18d.ra2_in1k`, but all layers have the same no. of channels and dimensions.

-> The resulting feature maps will be of shape `torch.Size([6, 64, 56, 56])` instead of `torch.Size([6, 512, 7, 7])` as in the original ResNet.


For dynamic shapes (it's analogous for the static ones):

```bash
just export-onnx-dynamic model=yaak/control_transformer/resnet_nodown.yaml input=yaak/control_transformer/dummy_224
```

Transfer the model to the delta kit and run: [proof_flatten_dynamic.sh](https://github.com/yaak-ai/drahve/blob/vit/whac-a-mole/proof_flatten_dynamic.sh) to convert the ONNX to TensorRT, run the TensorRT model and profile it. 

## Expected results
The layers in the TensorRT model are not be flattened (`[6,64,56,56]` of shape) and the layency is increased 5.29x the same as with ViT -> hence flattening is not the culprit.


| Batch | Images/second | Latency/batch   | Latency/image |
|-------|---------------|-----------------|---------------|
| 1     | 632.342       | 1.65            | 1.65          |
| 6     | 721.0(14%)    | 8.74105 (5.29x) | 1.38          |