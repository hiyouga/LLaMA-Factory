.PHONY: build commit license quality style test

check_dirs := scripts src tests tests_v1 setup.py

build:
	pip3 install build && python3 -m build

commit:
	pre-commit install
	pre-commit run --all-files

license:
	python3 tests/check_license.py $(check_dirs)

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/
