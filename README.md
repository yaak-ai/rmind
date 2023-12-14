# carGPT
Self-supervised model trained on vehicle context and control signals from expert drives

## Data

### Sync

Managed with [dvc](https://dvc.org/), see [dvc.yaml](./dvc.yaml):

```bash
dvc repro
```

### YAML templating

```
ytt --ignore-unknown-comments -f config/dataset/templates/ --output yaml --output-files config/dataset/
```

### Viz

Launch [FiftyOne](https://docs.voxel51.com/) remotely:

```bash
export MAPBOX_TOKEN=...

just dataviz
```

## Get Dalle models
TODO: Download once and cache
```
mkdir -p pretrained/dalle
wget https://cdn.openai.com/dall-e/encoder.pkl -P pretrained/dalle
wget https://cdn.openai.com/dall-e/decoder.pkl -P pretrained/dalle
```

## Installation
```
poetry install --with train,dev,lint
poetry shell
# https://github.com/facebookresearch/xformers/issues/705
pip install triton==2.0.0.dev20221105 --no-deps
```

## Training

### Gato
```bash
just train experiment=gato [++trainer.fast_dev_run=1 ...]
```

### SMART
```bash
just train experiment=smart [++trainer.fast_dev_run=1 ...]
```

## Inference

### Gato

1. start rerun:
```bash
just rerun
```

2. in another terminal:
```bash
just predict inference=gato model.name=<wandb_path>
```

3. open [http://localhost:9090/?url=ws://localhost:9877](http://localhost:9090/?url=ws://localhost:9877)
