# carGPT
Self-supervised model trained on vehicle context and control signals from expert drives

## Data

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


## Docker installation


1. Check the required Python version

```bash
cat pyproject.toml | grep python
```
According to the result, please set the `PYTHON_MAJOR` and `PYTHON_MINOR` arguments when building the image in step 3.

**Note** The `Dockerfile` uses `pyenv` to install the Python version `$PYTHON_MAJOR.$PYTHON_MINOR`, rather than `apt-get` from the distro repos. This is necesarry because we are currently using the Python version newer than those available in distro repos. Since the pyenv builds the python executable on the fly, building an image might take a little longer.

2. Optionally: check the Poetry version (although poetry version never seem to be an issue)

```bash
cat poetry.lock | grep Poetry
```
According to the result set the `POETRY_VER` argument for builing an image in step 3.

3. Building an image

```bash
docker compose -f docker-compose.yml build cargpt-gpu \
    --build-arg PYTHON_MAJOR=3.11 \
    --build-arg PYTHON_MINOR=8 \
    --build-arg POETRY_VER=1.8.2
```

4. Run the training

Firstly, create the output volumes so docker can store output artifacts in the same directories as we do when we run on bare metal (logs, checkpoints etc).
Without them, the non-root user inside the container will not be able to write resulting artifacts.

```bash
bash make_docker_volumes.sh
```
Then run  the training. Without logging to WANDB:

```bash
docker compose -f docker-compose.yml run cargpt-gpu \
    --train true  \
    --flags "experiment=smart paths.metadata_cache_dir=./yaak-datasets/metadata ~model.objectives.random_masked_hindsight_control ++datamodule.train.num_workers=2 ++trainer.devices=[1]"
```

With WANDB:

```bash 
docker compose -f docker-compose.yml run cargpt-gpu \
    --train true  \
    --flags "experiment=smart paths.metadata_cache_dir=./yaak-datasets/metadata ~model.objectives.random_masked_hindsight_control ++datamodule.train.num_workers=2 ++trainer.devices=[1]" \
    --WANDB_MODE online \
    --WANDB_API_KEY <api_key>
```

The output artifacts will have `uid=1000`, to change ownership to the current user, run:

```bash
bash chown_docker_volumes.sh
```

### Troubleshooting

1. Can't run the container with `nvidia` runtime.

Check if the `nvidia-container-toolkit` has been configured as suggested [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker), i.e. `/etc/docker/daemon.json` exists.

2. Can't run `docker` without `sudo`.

You need to be in the docker group (to check that run `cat /etc/group | grep docker`). If you are not, you can add $USER by running the below commands (notify Evgenii to add you as a USER in the`fax` repo).

```
sudo usermod -aG docker $USER
```

