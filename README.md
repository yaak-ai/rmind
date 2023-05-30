# carGPT
Self-supervised model trained on vehicle context and control signals from expert drives

## Data

### Sync

Managed with [dvc](https://dvc.org/), see [dvc.yaml](./dvc.yaml):

```bash
dvc repro
```

### Viz

Launch [FiftyOne](https://docs.voxel51.com/) remotely:

```bash
export MAPBOX_TOKEN=...

just dataviz
```

## Training

CIL++
```bash
just train experiment=cilpp [++trainer.fast_dev_run=1 ...]
```

Gato
```bash
just train experiment=gato [++trainer.fast_dev_run=1 ...]
```

## Visualize

### Attention

You can use `predict.py` to visualize attention maps as heatmaps with 
[Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).

```bash
just visualize output_file=test_vis.mp4 batch_size=4
```
