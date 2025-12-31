.PHONY: build commit license quality style test

check_dirs := scripts src tests tests_v1

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
BUILD := $(shell command -v uv >/dev/null 2>&1 && echo "uv build" || echo "python -m build")
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")

build:
	$(BUILD)

commit:
	$(TOOL) pre-commit install
	$(TOOL) pre-commit run --all-files

license:
	$(RUN) python3 tests/check_license.py $(check_dirs)

quality:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

test:
	WANDB_DISABLED=true $(RUN) pytest -vv --import-mode=importlib tests/ tests_v1/
