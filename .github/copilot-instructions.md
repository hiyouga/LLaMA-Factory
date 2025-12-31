# GitHub Copilot Instructions for LLaMA Factory

## Project Overview

LLaMA Factory is an efficient fine-tuning framework for 100+ large language models (LLMs). It provides:
- Support for various models: LLaMA, LLaVA, Mistral, Qwen, DeepSeek, Yi, Gemma, ChatGLM, Phi, etc.
- Multiple training methods: pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO
- Scalable resources: 16-bit full-tuning, freeze-tuning, LoRA and QLoRA variants
- Advanced algorithms: GaLore, BAdam, APOLLO, Adam-mini, Muon, OFT, DoRA, etc.
- Web UI (LLaMA Board) and CLI interfaces

### Architecture Versions

LLaMA Factory has two parallel architectures that can be switched via the `USE_V1` environment variable:

**v0 (default)** - File hierarchy:
- `api`, `webui` → `chat`, `eval`, `train` → `data`, `model` → `hparams` → `extras`

**v1** - File hierarchy:
- `trainers` → `core` → `accelerator`, `plugins`, `config` → `utils`

Set `USE_V1=1` to enable v1 architecture.

## Code Structure

### v0 Architecture (Default)

- `src/llamafactory/` - Main package directory
  - `api/` - OpenAI-style API implementation
  - `chat/` - Chat interface implementation
  - `cli.py` - Command-line interface
  - `data/` - Data processing and dataset handling
  - `eval/` - Model evaluation utilities
  - `extras/` - Additional utilities and helpers
  - `hparams/` - Hyperparameter definitions
  - `model/` - Model loading, patching, and utilities
  - `train/` - Training pipeline implementation
  - `webui/` - Gradio-based web interface
- `src/train.py` - Training entry script (delegates to `llamafactory.train.tuner`)
- `src/webui.py` - Web UI entry script (delegates to `llamafactory.webui.interface`)
- `src/api.py` - API server entry script (delegates to `llamafactory.api.app`)
- `tests/` - Test suite
- `examples/` - Example configurations for various training scenarios
- `data/` - Dataset definitions and examples

### v1 Architecture (USE_V1=1)

- `src/llamafactory/v1/` - Version 1 package directory
  - `trainers/` - Training implementations
  - `core/` - Core training utilities
  - `accelerator/` - Acceleration and distributed training
  - `plugins/` - Pluggable components (model, data, sampler, trainer)
  - `config/` - Configuration management
  - `utils/` - Utility functions

## Development Practices

### Code Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use ruff for linting and formatting
- Line length: 119 characters
- Indentation: 4 spaces
- Quote style: double quotes
- Use Google-style docstrings for documentation

### Import Organization

- Known first-party: `llamafactory`
- Known third-party: `accelerate`, `datasets`, `gradio`, `numpy`, `peft`, `torch`, `transformers`, `trl`
- Use 2 blank lines after imports

### Quality Checks

Before committing code, run:
```bash
make style      # Auto-fix style issues
make quality    # Check code quality
make test       # Run test suite
```

Or use the combined command:
```bash
make commit     # Run pre-commit hooks
```

### Testing

- Use pytest for testing
- Tests are located in `tests/` and `tests_v1/` directories
- Run tests with: `make test` (which runs `WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/`)
- Disable wandb during testing to avoid external dependencies
- **Note**: Training configurations require GPU machines, so training is typically not tested end-to-end. Use `make test` to validate file-level functionality.

### Building

Build the package with:
```bash
pip3 install build && python3 -m build
```

### License

- All source files must include the Apache 2.0 license header
- Check license headers with: `make license`

## Common Patterns

### Configuration Files

- Training configurations are typically YAML or JSON files in `examples/` directory
- Hyperparameters are defined using dataclasses in `src/llamafactory/hparams/`

### Model Support

- New model support is added through model patches in `src/llamafactory/model/`
- Visual models use the visual utilities in `src/llamafactory/model/model_utils/visual.py`
- Quantization support is in `src/llamafactory/model/model_utils/quantization.py`

### Data Processing

- Dataset definitions are in `data/dataset_info.json`
- Data templates and processors are in `src/llamafactory/data/`

### Training

- Training pipelines are in `src/llamafactory/train/`
- Support for different training methods: SFT, DPO, PPO, RM, PT, KTO, ORPO

## Key Dependencies

- Python >= 3.9.0
- PyTorch and transformers for model handling
- datasets for data processing
- peft for parameter-efficient fine-tuning
- accelerate for distributed training
- gradio for web UI
- trl for reinforcement learning
- Optional: vllm/sglang for inference, flash-attention-2, unsloth, liger-kernel

## Entry Points

- **CLI Training**: `llamafactory-cli train --config examples/train_lora/llama3_lora_sft.yaml`
- **Web UI**: `llamafactory-cli webui` or `python src/webui.py`
- **API Server**: `llamafactory-cli api` or `python src/api.py`
- **Chat Interface**: `llamafactory-cli chat --model_name_or_path MODEL_PATH`

## Environment Setup

For development:
```bash
pip install -e ".[dev]"
```

## Important Notes

- The project supports multiple backends: default PyTorch, vLLM, SGLang
- Megatron-core training is supported via mcore_adapter
- SwanLab and W&B are supported for experiment tracking
- Docker support is available with pre-built images
- Day-0/Day-1 support for latest cutting-edge models
- Multi-modal support for vision and audio understanding tasks

## Contribution Guidelines

1. Fork the repository
2. Create a development branch
3. Set up development environment with `pip install -e ".[dev]"`
4. Make changes following the style guide
5. Run quality checks: `make style && make quality`
6. Run tests: `make test`
7. Submit a pull request

## Common Commands

- `make style` - Format code
- `make quality` - Run linters
- `make test` - Run tests
- `make commit` - Install and run pre-commit hooks
- `make license` - Check license headers
