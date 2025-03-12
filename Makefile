.PHONY: build commit license quality style test

check_dirs := scripts src tests setup.py

build:
	pip install build && python -m build

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
	CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest -vv tests/
