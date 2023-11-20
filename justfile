set shell := ["zsh", "-cu"]

_default:
	@just --list

pre-commit:
	poetry run pre-commit run --all-files

generate-dataset-config:
	ytt --ignore-unknown-comments -f config/dataset/templates --output yaml --output-files config/dataset

train *ARGS: generate-dataset-config
	PYTHONOPTIMIZE=1 ./train.py {{ARGS}}

train-debug *ARGS: generate-dataset-config
	PYTHONOPTIMIZE=0 HYDRA_FULL_ERROR=1 WANDB_MODE=disabled ./train.py {{ARGS}}

visualize *ARGS:
	PYTHONOPTIMIZE=1 ./predict.py inference=visualize {{ARGS}}

trajectory *ARGS:
	PYTHONOPTIMIZE=1 ./predict.py inference=trajectory {{ARGS}}

predict *ARGS:
	PYTHONOPTIMIZE=1 ./predict.py {{ARGS}}

dataviz *ARGS: generate-dataset-config
	PYTHONOPTIMIZE=1 ./dataviz.py {{ARGS}}

dvc *ARGS:
	SHELL=$(which zsh) dvc {{ARGS}}

clean:
	rm -rf dist outputs lightning_logs wandb
