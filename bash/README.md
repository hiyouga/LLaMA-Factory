# LLaMA-Factory Environment Scripts

This folder contains bash scripts for managing the LLaMA-Factory environment.

## Scripts

### `install_env.sh`
**Purpose**: Complete environment installation script using `uv`

**Usage**:
```bash
./bash/install_env.sh
```

**What it does**:
- Checks if `uv` is installed
- Creates necessary memory directories
- Installs dependencies from `requirements.txt`
- Installs LLaMA-Factory in editable mode
- Verifies the installation
- Provides usage instructions

**Requirements**:
- `uv` must be installed on the system
- Must be run from the project root directory

### `activate_env.sh`
**Purpose**: Helper script for activating the virtual environment

**Usage**:
```bash
# To see instructions:
./bash/activate_env.sh

# To activate environment directly:
source bash/activate_env.sh
```

**What it does**:
- Checks if virtual environment exists
- Provides activation instructions
- Optionally activates the environment if sourced

## Installation Summary

The successful installation process that these scripts automate:

1. **Install dependencies**: `uv pip install -r requirements.txt`
2. **Install project**: `uv pip install -e .`
3. **Activate environment**: `source .venv/bin/activate`

## Known Issues

- `uv sync` fails due to dependency conflicts between swanlab extras and gradio versions
- `uv run` may fail for the same reason
- **Workaround**: Use the activated virtual environment instead

## Environment Details

- **Virtual Environment**: `.venv/` (created by uv)
- **Python Packages**: 133+ packages including transformers, datasets, accelerate, peft, trl, gradio, fastapi
- **CLI Tool**: `llamafactory-cli` with subcommands: api, chat, eval, export, train, webchat, webui, version

## Verification

After installation, verify with:
```bash
source .venv/bin/activate
python -c "import llamafactory; print('Success!')"
llamafactory-cli version
```
