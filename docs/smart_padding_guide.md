# Smart Padding Configuration Guide

LLaMA-Factory now includes intelligent padding configuration that automatically optimizes sequence padding for your model architecture, training configuration, and hardware setup.

## Overview

The smart padding system automatically detects the optimal `pad_to_multiple_of` value based on:
- **Model architecture** - Number of attention heads for memory alignment
- **FP8 training** - Tensor dimension requirements (multiples of 16 for TorchAO)
- **Sequence parallelism** - Ulysses degree compatibility
- **Hardware optimization** - General memory alignment (minimum 8)

## Usage

### Auto Detection (Default)
```yaml
# Auto-detection is enabled by default - no configuration needed!
# The system will automatically determine the optimal padding

# Or explicitly enable auto-detection:
pad_to_multiple_of: auto
```

### Manual Override
```yaml
# Specify exact padding value
pad_to_multiple_of: 32

# Disable padding entirely
pad_to_multiple_of: 1

# Use None to disable (alternative syntax)
pad_to_multiple_of: null
```

## How Auto-Detection Works

### 1. Model Architecture Detection
- Reads `num_attention_heads` from model config
- Uses attention head count for optimal memory alignment
- Example: Qwen2.5-7B has 28 attention heads → suggests padding to multiple of 28

### 2. FP8 Optimization
- Detects FP8 training mode (`fp8: true`)
- For TorchAO backend: ensures padding is multiple of 16
- For Transformer Engine backend: more flexible dimension handling

### 3. Sequence Parallel Compatibility
- Detects sequence parallel training (`sequence_parallel_size > 1`)
- Ensures padding is compatible with `alst_ulysses_degree`
- Adjusts padding to satisfy parallelism requirements

### 4. Conservative Fallbacks
- Minimum padding of 8 for basic memory alignment
- Falls back gracefully if model config detection fails
- Provides clear logging of detection decisions

## Examples

### LLaMA-3-8B Training
```yaml
# Auto-detected: 32 (matches 32 attention heads)
model_name_or_path: meta-llama/Llama-3-8b
# pad_to_multiple_of: auto  # Default - no need to specify
```

### Qwen2.5-7B with FP8
```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
fp8: true
fp8_backend: torchao
# Auto-detected: 28 (attention heads) but adjusted to 32 (next multiple of 16 for FP8)
```

### Sequence Parallel Training
```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
sequence_parallel_size: 4
alst_ulysses_degree: 4
# Auto-detected: 32 (ensures compatibility with ulysses_degree=4)
```

### Manual Override for Special Cases
```yaml
model_name_or_path: custom/model
pad_to_multiple_of: 64  # Force specific value for custom optimization
```

## Configuration Validation

The system provides helpful warnings for suboptimal configurations:

### FP8 Warnings
```
WARNING: FP8 training with pad_to_multiple_of=20 may be suboptimal.
Consider using a multiple of 16 for TorchAO backend.
```

### Sequence Parallel Warnings
```
WARNING: Sequence parallel training may be suboptimal with pad_to_multiple_of=30.
Consider using a multiple of 4.
```

## Logging Output

During training initialization, you'll see logs like:
```
INFO: Detected 32 attention heads from model.config.num_attention_heads
INFO: Auto-detected optimal pad_to_multiple_of: 32
```

## Advanced Configuration

### Model-Specific Overrides
For models with unusual architectures, you can always override:
```yaml
model_name_or_path: unusual/model
pad_to_multiple_of: 16  # Manual override for special requirements
```

### Development and Testing
```yaml
# Disable padding for debugging
pad_to_multiple_of: 1

# Use small padding for development
pad_to_multiple_of: 8
```

## Performance Impact

Proper padding alignment can provide:
- **Memory efficiency**: Better GPU memory utilization
- **Computational performance**: Optimized tensor operations
- **FP8 acceleration**: Maximum performance from low-precision training
- **Parallel efficiency**: Better scaling in distributed training

## Backward Compatibility

This system is fully backward compatible:
- Existing configs without `pad_to_multiple_of` get auto-detection
- Manual values in configs are respected exactly
- No breaking changes to existing workflows
