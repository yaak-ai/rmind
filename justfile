export HYDRA_FULL_ERROR := "1"
export PYTHONOPTIMIZE := "1"
export RERUN_STRICT := "1"

_default:
    @just --list --unsorted

sync:
    uv sync --all-extras --all-groups --locked

setup: sync install-duckdb-extensions
    uvx --with=pre-commit-uv pre-commit install --install-hooks

install-duckdb-extensions:
    uv run python -c "import duckdb; duckdb.connect().install_extension('spatial')"

format *ARGS:
    uvx ruff format {{ ARGS }}

lint *ARGS:
    uvx ruff check {{ ARGS }}

typecheck *ARGS:
    uvx ty@latest check {{ ARGS }}

# run pre-commit on all files
pre-commit:
    uvx --with=pre-commit-uv pre-commit run --all-files --color=always

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
    uv run pytest --capture=no {{ ARGS }}

export-onnx *ARGS: generate-config
    uv run --group export rmind-export-onnx \
        --config-path {{ justfile_directory() }}/config \
        --config-name export_onnx.yaml \
        {{ ARGS }}

# start rerun server and viewer
rerun bind="0.0.0.0" port="9877" web-viewer-port="9090":
    uvx --from rerun-sdk@latest rerun \
    	--bind {{ bind }} \
    	--port {{ port }} \
    	--web-viewer \
    	--web-viewer-port {{ web-viewer-port }} \
    	--memory-limit 95% \
    	--server-memory-limit 95% \
    	--expect-data-soon \

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
