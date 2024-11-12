[![CircleCI](https://dl.circleci.com/status-badge/img/gh/yaak-ai/carGPT/tree/main.svg?style=svg&circle-token=CCIPRJ_Bt9RtTi6AXM3i6UMagfC14_e353d6e7992027b2b724489ebbf258ee91a0532f)](https://dl.circleci.com/status-badge/redirect/gh/yaak-ai/carGPT/tree/main)

# carGPT

Self-supervised model trained on vehicle context and control signals from expert drives.

## Package

To run inference from another poetry venv:

```shell
uv add "git+ssh://git@github.com/yaak-ai/cargpt.git#main[predict]"

uv run cargpt-predict --config-path /path/to/config/dir --config-name config.yaml
```

## Setup
```shell
just setup
```

List all available recipes:
```shell
just
```

## Train

```shell
just train experiment=control_transformer/pretrain [...]
```

## Predict

1. start rerun and open [http://localhost:9090/?url=ws://localhost:9877](http://localhost:9090/?url=ws://localhost:9877)
```shell
just rerun
```

3. in another terminal:
```shell
just predict inference=control_transformer/default model=control_transformer/pretrained model.artifact=yaak/cargpt/model-{run_id}:v{version} [+model.map_location=cuda:0] [+model.strict=false]
```


To use another [sexy](https://github.com/yaak-ai/sexy) visualisation tool add `ScoresPredictionWriter`  [callback](src/cargpt/callbacks/scores_writer.py) to corresponding inference config.

