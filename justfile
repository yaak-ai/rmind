export HOSTNAME := `hostname`
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

check:
    uv format --check
    uv run ruff check
    uv check

check-git:
    uv run rmind-check-git

prek *ARGS:
    prek --all-files {{ ARGS }}

# generate config files from templates with ytt
generate-config:
    ytt --file {{ justfile_directory() }}/config/_templates \
        --output-files {{ justfile_directory() }}/config/ \
        --output yaml \
        --ignore-unknown-comments \
        --strict

train *ARGS: generate-config check-git
    uv run \
        --extra train \
        rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        {{ ARGS }}

train-debug *ARGS: generate-config
    WANDB_MODE=disabled \
    uv run \
        --extra train \
        rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        experiment=yaak/control_transformer/pretrain \
        datamodule=yaak/train_debug \
        ++model.encoder.disable=true \
        {{ ARGS }}

predict +ARGS: generate-config
    uv run \
        --extra train --extra predict \
        rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        {{ ARGS }}

predict-policy-with-permutations +ARGS: generate-config
    uv run \
        --extra train --extra predict \
        rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        --multirun \
        inference=yaak/control_transformer/policy_with_features_permutation \
        permutation=baseline,speed,cam_front_left,waypoints,all_observations \
        {{ ARGS }}

test *ARGS: generate-config
    uv run \
        --all-extras --group test \
        pytest --capture=no -v {{ ARGS }}

# refresh recorded test snapshots (e.g. training_step_losses.json)
update-snapshots:
    uv run python -m tests.scripts.update_snapshots

benchmark-onnx +ARGS:
    LD_LIBRARY_PATH="$(find /nix/store -maxdepth 1 -name '*gcc-15*-lib' -type d -print -quit 2>/dev/null)/lib:${LD_LIBRARY_PATH:-}" \
    uv run --group benchmark rmind-benchmark-onnx {{ ARGS }}

export-onnx *ARGS: generate-config
    uv run \
        --extra export \
        rmind-export-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name export_onnx.yaml \
        {{ ARGS }}

onnxvis *ARGS:
    uvx --python 3.12 --with=ai-edge-model-explorer --from=model-explorer-onnx onnxvis {{ ARGS }}

# start rerun server and viewer
rerun *ARGS:
    uv run rerun --serve-web {{ ARGS }}

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
