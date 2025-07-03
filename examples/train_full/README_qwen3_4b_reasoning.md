# Qwen3-4B Training Configurations for Big Reasoning Traces

This directory contains training configurations for fine-tuning Qwen3-4B on the Big Reasoning Traces dataset from AllenAI.

## Dataset

- **Source**: [allenai/big-reasoning-traces](https://huggingface.co/datasets/allenai/big-reasoning-traces)
- **Subset**: `deepseek_debug`
- **Format**: Question-Answer pairs for reasoning tasks
- **Dataset Config**: Added to `data/dataset_info.json`

## Configurations

### 1. Full Fine-tuning (`qwen3_4b_full_sft.yaml`)
- **Method**: Full parameter fine-tuning
- **Memory**: High (requires significant GPU memory)
- **Learning Rate**: 5.0e-6 (lower for full fine-tuning)
- **Batch Size**: 1 with gradient accumulation of 4
- **Context Length**: 4096 tokens

### 2. LoRA Fine-tuning (`qwen3_4b_lora_sft.yaml`)
- **Method**: Low-Rank Adaptation (LoRA)
- **Memory**: Moderate
- **Learning Rate**: 1.0e-4 (higher for LoRA)
- **LoRA Rank**: 16 (higher for reasoning tasks)
- **Batch Size**: 2 with gradient accumulation of 2

### 3. QLoRA Fine-tuning (`qwen3_4b_qlora_sft.yaml`)
- **Method**: Quantized LoRA (4-bit quantization)
- **Memory**: Low (most memory efficient)
- **Learning Rate**: 1.0e-4
- **LoRA Rank**: 16
- **Batch Size**: 2 with gradient accumulation of 4

## Key Features

- **Thinking Mode**: Enabled (`enable_thinking: true`) for enhanced reasoning capabilities
- **Template**: Uses `qwen3` template optimized for Qwen3 models
- **Context Length**: Increased to 4096 tokens for longer reasoning chains
- **DeepSpeed**: Uses ZeRO-3 for distributed training

## Usage

### Full Fine-tuning
```bash
llamafactory-cli train examples/train_full/qwen3_4b_full_sft.yaml
```

### LoRA Fine-tuning
```bash
llamafactory-cli train examples/train_lora/qwen3_4b_lora_sft.yaml
```

### QLoRA Fine-tuning
```bash
llamafactory-cli train examples/train_qlora/qwen3_4b_qlora_sft.yaml
```

## Hardware Requirements

- **Full Fine-tuning**: 4x A100 80GB or equivalent
- **LoRA**: 2x A100 40GB or equivalent
- **QLoRA**: 1x A100 24GB or equivalent

## Notes

- Adjust `max_samples` based on your dataset size and training time constraints
- The `enable_thinking` parameter is crucial for reasoning tasks with Qwen3
- Consider enabling evaluation by uncommenting the eval section for monitoring training progress
- Output models will be saved in `saves/qwen3-4b/` directory structure 