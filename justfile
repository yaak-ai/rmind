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

# fit the flow policy's action-normalization stats from data (Gaussianize knots
# + optional maneuver-LDS densities -> the action_norm json consumed at init)
action-norm +ARGS: generate-config
    uv run python -m rmind.scripts.flow_action_norm \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        {{ ARGS }}

# export the full policy pipeline (episode builder + encoder + flow decoder) to
# ONNX / pt2 — input is the raw sensor batch, for edge deployment
export-policy +ARGS: generate-config
    uv run \
        --extra export --extra train \
        python -m rmind.scripts.flow_export \
        --config-path {{ justfile_directory() }}/config \
        --config-name export/policy_onnx.yaml \
        {{ ARGS }}

# check a saved policy ONNX matches the eager PyTorch model, on CPU and GPU
# (per-action-channel diff table + per-pass latency); same args as export-policy
compare-policy-onnx +ARGS: generate-config
    uv run \
        --extra export --extra train \
        python -m rmind.scripts.flow_export_compare \
        --config-path {{ justfile_directory() }}/config \
        --config-name export/policy_onnx.yaml \
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
