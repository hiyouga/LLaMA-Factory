# Advanced Training Features

This directory contains examples for the newly integrated advanced training features in LLaMA-Factory, ported from Axolotl:

## Features

### 1. FP8 Native Training via TorchAO

FP8 (8-bit floating point) training provides significant speedup with minimal accuracy loss.

**Requirements:**
- PyTorch 2.7.0 or higher
- Hopper architecture GPUs (H100, H200) with compute capability 9.0+
- `torchao` package installed

**Configuration:**
```yaml
fp8: true                               # Enable FP8 training
fp8_enable_fsdp_float8_all_gather: true # Enable FP8 FSDP optimizations  
torch_compile: true                     # Recommended for best performance
```

**Example:** `llama3_fp8_sft.yaml`

### 2. Quantization Aware Training (QAT)

QAT trains models with fake quantization, enabling deployment of quantized models with minimal accuracy degradation.

**Requirements:**
- `torchao` package installed

**Configuration:**
```yaml
enable_qat: true                        # Enable QAT
qat_activation_dtype: int8              # Activation quantization (optional)
qat_weight_dtype: int8                  # Weight quantization  
qat_group_size: 32                      # Quantization group size
qat_quantize_embedding: true            # Quantize embeddings
fake_quant_after_n_steps: 100           # Delay fake quantization
```

**Example:** `llama3_qat_sft.yaml`

### 3. HuggingFace Kernels Support

Integration with the new HuggingFace kernels package for optimized operations.

**Requirements:**
- `kernels` package installed
- `cut-cross-entropy` package
- `triton` < 3.4.0

**Configuration:**
```yaml
use_kernels: true                       # Enable HF kernels
kernel_name: kernels-community/activation  # Specific kernel from HF Hub
```

**Example:** `llama3_kernels_sft.yaml`

## Installation

### Basic Installation
```bash
# Install core requirements
pip install -e .[advanced]
```

### Manual Installation
```bash
# For FP8 and QAT support
pip install "torchao>=0.8.0"

# For HuggingFace kernels support  
pip install "kernels>=0.9.0" cut-cross-entropy "triton<3.4.0"
```

## Usage

### Training with FP8
```bash
llamafactory-cli train examples/advanced_training/llama3_fp8_sft.yaml
```

### Training with QAT
```bash
llamafactory-cli train examples/advanced_training/llama3_qat_sft.yaml
```

### Training with HF Kernels
```bash
llamafactory-cli train examples/advanced_training/llama3_kernels_sft.yaml
```

## Performance Considerations

### FP8 Training
- **Speedup:** 20-30% faster training on Hopper GPUs
- **Memory:** Reduced memory usage
- **Accuracy:** Minimal impact on final model quality
- **Compatible with:** FSDP2, torch.compile, FlashAttention

### QAT Training
- **Purpose:** Prepare models for INT8 deployment
- **Accuracy:** <2% degradation compared to full precision
- **Deployment:** Models can be quantized post-training for inference
- **Memory:** Slight increase during training due to fake quantization

### HF Kernels
- **Purpose:** Optimized operations for specific model architectures
- **Flexibility:** Load different kernels from HuggingFace Hub
- **Performance:** Kernel-dependent optimizations

## Compatibility Matrix

| Feature | FP8 | QAT | HF Kernels | Standard Quantization |
|---------|-----|-----|------------|----------------------|
| FP8     | ✅  | ⚠️  | ✅         | ❌                   |
| QAT     | ⚠️  | ✅  | ✅         | ❌                   |
| HF Kernels | ✅ | ✅  | ✅         | ✅                   |
| LoRA    | ✅  | ✅  | ✅         | ✅                   |
| FSDP    | ✅  | ✅  | ✅         | ✅                   |

**Legend:**
- ✅ Compatible
- ⚠️ Compatible but may require careful tuning
- ❌ Not compatible

## Troubleshooting

### FP8 Issues
- **Error:** "FP8 requires PyTorch 2.7+"
  - **Solution:** Upgrade PyTorch to 2.7.0 or higher
- **Error:** "FP8 requires Hopper architecture"
  - **Solution:** Use H100/H200 GPUs or disable FP8

### QAT Issues
- **Error:** "torchao not available"
  - **Solution:** Install torchao: `pip install torchao>=0.8.0`
- **Convergence issues**
  - **Solution:** Try delaying fake quantization with `fake_quant_after_n_steps`

### HF Kernels Issues
- **Error:** "kernels package not available"
  - **Solution:** Install kernels: `pip install kernels>=0.9.0`
- **Kernel not found**
  - **Solution:** Verify kernel name exists on HuggingFace Hub

## Advanced Configuration

### Combining Features
You can combine compatible features for maximum optimization:

```yaml
# FP8 + HF Kernels (Recommended)
fp8: true
use_kernels: true
kernel_name: kernels-community/activation
torch_compile: true
use_liger_kernel: true

# QAT + HF Kernels
enable_qat: true
qat_weight_dtype: int8
use_kernels: true
kernel_name: kernels-community/activation
```

### Validation
The system includes automatic validation to prevent conflicting configurations:
- FP8 and standard quantization are mutually exclusive
- QAT and standard quantization are mutually exclusive
- Warnings for potentially conflicting combinations