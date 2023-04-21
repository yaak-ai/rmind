# carGPT
Self-supervised model trained on vehicle context and control signals from expert drives

## Training

```bash
just train experiment=cilpp [++trainer.fast_dev_run=1 ...]
```


## Visualize

You can use `predict.py` to visualize attention maps as heatmaps with 
[Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).

```bash
just visualize output_file=test_vis.mp4 batch_size=4
```
