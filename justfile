set shell := ["zsh", "-cu"]

_default:
	@just --list

format:
	poetry run black .

fix:
	poetry run ruff check . --fix

lint:
	poetry run ruff check . --no-fix

check-format:
	poetry run black . --check

typecheck:
	poetry run mypy .

check-deps:
	poetry run deptry .

check: check-format lint typecheck check-deps

train *ARGS:
	PYTHONOPTIMIZE=1 ./train.py {{ARGS}}

train-debug *ARGS:
	PYTHONOPTIMIZE=0 HYDRA_FULL_ERROR=1 ./train.py {{ARGS}}

visualize *ARGS:
	PYTHONOPTIMIZE=1 ./predict.py inference=visualize {{ARGS}}

predict *ARGS:
	PYTHONOPTIMIZE=1 ./predict.py {{ARGS}}

dataviz *ARGS:
	PYTHONOPTIMIZE=1 ./dataviz.py {{ARGS}}

dvc *ARGS:
	SHELL=$(which zsh) dvc {{ARGS}}
