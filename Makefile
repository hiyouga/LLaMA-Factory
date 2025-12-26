.PHONY: build commit license quality style test

check_dirs := scripts src tests tests_v1

build:
	uv build

commit:
	uv run pre-commit install
	uv run pre-commit run --all-files

license:
	uv run python tests/check_license.py $(check_dirs)

quality:
	uv run ruff check $(check_dirs)
	uv run ruff format --check $(check_dirs)

style:
	uv run ruff check $(check_dirs) --fix
	uv run ruff format $(check_dirs)

test:
	WANDB_DISABLED=true uv run pytest -vv --import-mode=importlib tests/ tests_v1/
