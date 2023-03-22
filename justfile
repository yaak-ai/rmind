set shell := ["nu", "-c"]

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

check-deps:
	poetry run deptry .

check: check-format lint typecheck check-deps

train *ARGS:
	PYTHONOPTIMIZE=1 ./train.py {{ARGS}}

train-debug *ARGS:
	PYTHONOPTIMIZE=0 HYDRA_FULL_ERROR=1 ./train.py {{ARGS}}
