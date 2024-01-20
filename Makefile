.PHONY: quality style

check_dirs := src tests

quality:
	black --check $(check_dirs)
	ruff $(check_dirs)

style:
	black $(check_dirs)
	ruff $(check_dirs) --fix
