# Self Memory - LLaMA-Factory Project

## Environment Setup
- Project uses both pyproject.toml and requirements.txt for dependency management
- Uses uv for Python package management as per workspace rules
- Project requires Python >=3.9.0
- Successfully installed environment using `uv pip install -r requirements.txt` and `uv pip install -e .`
- Virtual environment located at `.venv/` (created by uv)
- To use: `source .venv/bin/activate` before running Python commands
- CLI available as `llamafactory-cli` with subcommands: api, chat, eval, export, train, webchat, webui, version

## Installation Issues Resolved
- `uv sync` failed due to dependency conflicts with swanlab extras and gradio versions
- Workaround: Used `uv pip install -r requirements.txt` instead, which resolved dependencies successfully
- Note: `uv run` may still have issues due to pyproject.toml dependency conflicts, use activated venv instead

## Automation Scripts
- Created `bash/install_env.sh` - Complete automated installation script
- Created `bash/activate_env.sh` - Environment activation helper
- Created `bash/README.md` - Documentation for the scripts
- All scripts are executable and tested working

## Code Style
- Uses Ruff for formatting and linting (configured in pyproject.toml)
- Google-style docstrings required
- Line length: 119 characters
- Target Python version: 3.9+

## Key Dependencies
- transformers, datasets, accelerate, peft, trl for ML/LLM functionality
- gradio for GUI
- fastapi for API development
- Various model and tokenizer libraries (sentencepiece, tiktoken, etc.)
