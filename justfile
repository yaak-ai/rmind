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

format *ARGS:
    uv run ruff format {{ ARGS }}

lint *ARGS:
    uv run ruff check {{ ARGS }}

check-git:
    uv run rmind-check-git

# build image tagged with HEAD commit; requires a clean, pushed tree so the tag is truthful
docker-build *ARGS: check-git
    docker build \
        --build-arg COMMIT_SHA=$(git rev-parse HEAD) \
        -t rmind:$(git rev-parse HEAD) \
        {{ ARGS }} .

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

train *ARGS: generate-config check-git
    uv run rmind-train \
        --config-path {{ justfile_directory() }}/config \
        --config-name train.yaml \
        {{ ARGS }}

# train without the git cleanliness check — for docker, where the image is pinned to a commit
train-unsafe *ARGS: generate-config
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

benchmark-onnx *ARGS: generate-config
    LD_LIBRARY_PATH="$(find /nix/store -maxdepth 1 -name '*gcc-15*-lib' -type d -print -quit 2>/dev/null)/lib:${LD_LIBRARY_PATH:-}" \
    uv run --group benchmark rmind-benchmark-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name benchmark_onnx.yaml \
        {{ ARGS }}

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
