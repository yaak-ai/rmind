set shell := ["zsh", "-cu"]

_default:
	@just --list

format:
	poetry run black .
	poetry run isort .

check-format:
	poetry run black . --check
	poetry run isort . --check

typecheck:
	poetry run mypy .

lint:
	poetry run ruff .

check: check-format lint typecheck

train *ARGS:
	PYTHONOPTIMIZE=1 ./train.py {{ARGS}}

train-debug *ARGS:
	PYTHONOPTIMIZE=0 HYDRA_FULL_ERROR=1 ./train.py {{ARGS}}
