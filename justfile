export PYTHONOPTIMIZE := "1"
export PYTHONBREAKPOINT := "patdb.debug"
export PATDB_CODE_STYLE := "vim"
export BETTER_EXCEPTIONS := "1"
export LOVELY_TENSORS := "1"
export HYDRA_FULL_ERROR := "1"
export RERUN_STRICT := "1"

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --all-groups --locked

install-tools:
    uv tool install --force --upgrade basedpyright
    uv tool install --force --upgrade ruff
    uv tool install --force --upgrade pre-commit --with pre-commit-uv

setup: sync install-tools
    uvx pre-commit install --install-hooks

format *ARGS:
    uvx --from ruff@latest ruff format {{ ARGS }}

lint *ARGS:
    uvx --from ruff@latest ruff check {{ ARGS }}

typecheck *ARGS:
    uvx --from basedpyright@latest basedpyright {{ ARGS }}

# run pre-commit on all files
pre-commit *ARGS:
    uvx pre-commit run --all-files --color=always {{ ARGS }}

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
        {{ ARGS }}

predict +ARGS: generate-config
    uv run rmind-predict \
        --config-path {{ justfile_directory() }}/config \
        --config-name predict.yaml \
        {{ ARGS }}

test *ARGS: generate-config
    uv run pytest --capture=no -v {{ ARGS }}

export-onnx *ARGS: generate-config
    uv run --group export rmind-export-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name export/onnx.yaml \
        {{ ARGS }}

# start rerun server and viewer
rerun *ARGS:
    uvx --from rerun-sdk@latest rerun --serve-web {{ ARGS }}

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
