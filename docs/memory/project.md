# Project Preferences - LLaMA-Factory

## Dependency Management
- **Primary**: uv for Python package management
- **Configuration**: Uses both pyproject.toml and requirements.txt

## Code Quality
- **Formatter**: Ruff (replaces Black, isort, flake8)
- **Docstring Style**: Google-style
- **Type Annotations**: Comprehensive type hints required
- **Testing**: pytest framework

## Python Version
- **Minimum**: Python 3.9.0
- **Features**: Prioritize Python 3.10+ features when available

## Framework Preferences
- **Web Framework**: FastAPI for API development
- **ML Libraries**: transformers, datasets, accelerate, peft, trl
- **Configuration**: Hydra or YAML for experiments
- **Data Validation**: Pydantic models

## Execution
- **Primary**: `uv run` instead of `python` or `python3` (when working)
- **Fallback**: Activate virtual environment with `source .venv/bin/activate` then use `python`
- **Note**: Due to dependency conflicts in pyproject.toml, `uv run` may fail; use activated venv as workaround
