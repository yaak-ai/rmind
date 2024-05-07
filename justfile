set shell := ["zsh", "-cu"]

export HYDRA_FULL_ERROR := "1"

_default:
	@just --list

install:
	poetry lock --no-update
	poetry install --no-root --sync --with=train,lint,dev,test,inference,notebook
	poetry run pre-commit install

pre-commit:
	poetry run pre-commit run --all-files

generate-dataset-config:
	ytt --ignore-unknown-comments -f config/dataset/templates --output yaml --output-files config/dataset

train *ARGS: generate-dataset-config
	PYTHONOPTIMIZE=1 poetry run python train.py {{ ARGS }}

train-debug *ARGS: generate-dataset-config
	PYTHONOPTIMIZE=0 WANDB_MODE=disabled poetry run python train.py {{ ARGS }}

predict +ARGS:
	PYTHONOPTIMIZE=1 poetry run python predict.py {{ ARGS }}

predict-debug +ARGS:
	PYTHONOPTIMIZE=0 poetry run python predict.py {{ ARGS }}

rerun bind="0.0.0.0" port="9876" ws-server-port="9877" web-viewer-port="9090":
	poetry run rerun --bind {{ bind }} --port {{ port }} --ws-server-port {{ ws-server-port }} --web-viewer --web-viewer-port {{ web-viewer-port }}

clean:
	rm -rf dist outputs lightning_logs wandb artifacts

test *ARGS:
	PYTHONOPTIMIZE=0 poetry run pytest --capture=no {{ ARGS }}
