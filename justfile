set shell := ["zsh", "-cu"]

export HYDRA_FULL_ERROR := "1"
export PYTHONOPTIMIZE := "1"

_default:
    @just --list --unsorted

setup:
    uv sync --all-extras
    for tool in basedpyright ruff pre-commit deptry; do uv tool install --force --upgrade $tool;  done
    uvx pre-commit install --install-hooks

update:
    uv sync --all-extras

# run pre-commit on all files
pre-commit:
    uvx pre-commit run --all-files --color=always

# generate config files from templates with ytt
template-config:
    ytt --file src/config/_templates \
        --output-files src/config/ \
        --output yaml \
        --ignore-unknown-comments \
        --strict

train *ARGS: template-config
    uv run python src/cargpt/scripts/train.py \
        --config-path {{ justfile_directory() }}/src/config \
        --config-name train.yaml {{ ARGS }}

# train with runtime type checking and no wandb
train-debug *ARGS: template-config
    WANDB_MODE=disabled uv run python src/cargpt/scripts/train.py \
        --config-path {{ justfile_directory() }}/src/config \
        --config-name train.yaml {{ ARGS }}

predict +ARGS:
    uv run python src/cargpt/scripts/predict.py \
        --config-path {{ justfile_directory() }}/src/config \
        --config-name predict.yaml {{ ARGS }}

# predict with runtime type checking
predict-debug +ARGS:
    uv run python src/cargpt/scripts/predict.py \
        --config-path {{ justfile_directory() }}/src/config \
        --config-name predict.yaml {{ ARGS }}

test *ARGS:
    uv run pytest --capture=no {{ ARGS }}

dataviz *ARGS: template-config
    uv run python src/cargpt/scripts/dataviz.py \
        --config-path {{ justfile_directory() }}/src/config \
        --config-name dataviz.yaml {{ ARGS }}

# start rerun server and viewer
rerun bind="0.0.0.0" port="9876" ws-server-port="9877" web-viewer-port="9090":
    RUST_LOG=debug uv run rerun \
    	--bind {{ bind }} \
    	--port {{ port }} \
    	--ws-server-port {{ ws-server-port }} \
    	--web-viewer \
    	--web-viewer-port {{ web-viewer-port }} \
    	--memory-limit 95% \
    	--server-memory-limit 95% \
    	--expect-data-soon \

clean:
    rm -rf dist outputs lightning_logs wandb artifacts
