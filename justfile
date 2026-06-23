export PYTHONBREAKPOINT := "patdb.debug"
export PATDB_CODE_STYLE := "vim"
export BETTER_EXCEPTIONS := "1"
export LOVELY_TENSORS := "1"
export HYDRA_FULL_ERROR := "1"
export RERUN_STRICT := "1"
export WANDB_DIR := "wandb_logs"
export TORCHDYNAMO_VERBOSE := "1"
export PYTORCH_CUDA_ALLOC_CONF := "expandable_segments:True"

# export PYTHONOPTIMIZE := "1" # incompatible w/ torch.export in 2.10

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --all-groups --locked

setup: sync
    prek install --overwrite

format *ARGS:
    uv run ruff format {{ ARGS }}

lint *ARGS:
    uv run ruff check {{ ARGS }}

check-git:
    uv run rmind-check-git

typecheck *ARGS:
    uv run ty check {{ ARGS }}

prek *ARGS:
    prek --all-files {{ ARGS }}

# generate config files from templates with ytt
generate-config:
    ytt --file {{ justfile_directory() }}/config/_templates \
        --output-files {{ justfile_directory() }}/config/ \
        --output yaml \
        --ignore-unknown-comments \
        --strict

train *ARGS: generate-config
    uv run rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        {{ ARGS }}

train-debug *ARGS: generate-config
    WANDB_MODE=disabled uv run rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        experiment=yaak/control_transformer/pretrain \
        datamodule=yaak/train_debug \
        ++model.encoder.disable=true \
        {{ ARGS }}

train-action *ARGS: generate-config
    uv run rmind-train \
          --config-path {{ justfile_directory() }}/config \
          --config-name train.yaml \
          experiment=yaak/action_tokenizer/pretrain \
          datamodule=yaak/action_train \
          {{ ARGS }}

# train a residual-VQ tokenizer on waypoints and visualize the codebooks
tokenize-waypoints *ARGS: generate-config
    uv run python {{ justfile_directory() }}/scripts/waypoint_tokenizer.py {{ ARGS }}

train-waypoint *ARGS: generate-config
    uv run rmind-train \
          --config-path {{ justfile_directory() }}/config \
          --config-name train.yaml \
          experiment=yaak/waypoint_tokenizer/pretrain \
          {{ ARGS }}

predict +ARGS: generate-config
    uv run rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        {{ ARGS }}

predict-policy-with-permutations +ARGS: generate-config
    uv run rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        --multirun \
        inference=yaak/control_transformer/policy_with_features_permutation \
        permutation=baseline,speed,cam_front_left,waypoints,all_observations \
        {{ ARGS }}

test *ARGS: generate-config
    uv run pytest --capture=no -v {{ ARGS }}

# refresh recorded test snapshots (e.g. training_step_losses.json)
update-snapshots:
    uv run python -m tests.scripts.update_snapshots

export-onnx *ARGS: generate-config
    uv run --group export rmind-export-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name export/onnx.yaml \
        {{ ARGS }}

onnxvis *ARGS:
    uvx --python 3.12 --with=ai-edge-model-explorer --from=model-explorer-onnx onnxvis {{ ARGS }}

# start rerun server and viewer
rerun *ARGS:
    uv run rerun --serve-web {{ ARGS }}

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
