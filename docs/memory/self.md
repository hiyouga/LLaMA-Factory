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

## Git Repository Management
- Configured pre-commit hooks with 25MB file size limit
- Removed large files from git history using git-filter-repo
- Added data/squad_v2/train.json (158MB) and validation.json (15MB) to .gitignore
- Successfully pushed to GitHub repository without large file issues
- Pre-commit hooks now prevent large files from being committed

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
