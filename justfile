set shell := ["zsh", "-cu"]

export HYDRA_FULL_ERROR := "1"
export PYTHONOPTIMIZE := "1"

_default:
    @just --list --unsorted

# poetry venv setup
setup:
    poetry lock --no-update
    poetry install --sync --with=dev,test,lint,train,predict,notebook
    poetry run pre-commit install --install-hooks

update:
    poetry update --with=dev,test,lint,train,predict,notebook

# run pre-commit on all files
pre-commit:
    poetry run pre-commit validate-config
    poetry run pre-commit install --install-hooks
    poetry run pre-commit run --all-files

# generate config files from templates with ytt
template-config:
    ytt --file config/_templates --output-files config/ --output yaml --ignore-unknown-comments --strict

train *ARGS: template-config
    poetry run python cargpt/scripts/train.py --config-path ../../config --config-name train.yaml {{ ARGS }}

# train with runtime type checking and no wandb
train-debug *ARGS: template-config
    WANDB_MODE=disabled poetry run python cargpt/scripts/train.py --config-path ../../config --config-name train.yaml {{ ARGS }}

predict +ARGS:
    poetry run python cargpt/scripts/predict.py --config-path ../../config --config-name predict.yaml {{ ARGS }}

# predict with runtime type checking
predict-debug +ARGS:
    poetry run python cargpt/scripts/predict.py --config-path ../../config --config-name predict.yaml {{ ARGS }}

test *ARGS:
    poetry run pytest --capture=no {{ ARGS }}

dataviz *ARGS: template-config
    poetry run python cargpt/scripts/dataviz.py --config-path ../../config --config-name dataviz.yaml {{ ARGS }}

# start rerun server and viewer
rerun bind="0.0.0.0" port="9876" ws-server-port="9877" web-viewer-port="9090":
    RUST_LOG=debug poetry run rerun \
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
