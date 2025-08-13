# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import TYPE_CHECKING, Optional

from ...extras import logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def apply_fp8_optimization(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """Apply FP8 optimization using torchao if enabled."""
    if not model_args.fp8:
        return
    
    try:
        # Check PyTorch version requirement
        pytorch_version = torch.__version__.split('+')[0]  # Remove +cu121 etc.
        major, minor = map(int, pytorch_version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 7):
            logger.warning_rank0(
                f"FP8 training requires PyTorch 2.7+, but found {pytorch_version}. Skipping FP8 optimization."
            )
            return
        
        # Check GPU architecture
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability < (9, 0):  # Hopper architecture requirement
                logger.warning_rank0(
                    f"FP8 training requires Hopper architecture (compute capability 9.0+), "
                    f"but found {device_capability}. Skipping FP8 optimization."
                )
                return
        
        import torchao
        from torchao.float8 import convert_to_float8_training
        
        # Apply FP8 conversion
        logger.info_rank0("Applying FP8 optimization using torchao...")
        model = convert_to_float8_training(model)
        
        # Apply FSDP float8 all-gather if enabled
        if model_args.fp8_enable_fsdp_float8_all_gather:
            from torchao.float8.fsdp_utils import precompute_float8_dynamic_scale_for_fsdp
            precompute_float8_dynamic_scale_for_fsdp(model)
            logger.info_rank0("Enabled FP8 FSDP all-gather optimization.")
        
        logger.info_rank0("Successfully applied FP8 optimization.")
        
    except ImportError:
        logger.warning_rank0(
            "torchao not available. Please install with: pip install torchao>=0.8.0"
        )
    except Exception as e:
        logger.warning_rank0(f"Failed to apply FP8 optimization: {e}")


def prepare_model_for_qat(
    model: "PreTrainedModel", 
    model_args: "ModelArguments",
    activation_dtype: Optional[str] = None,
    weight_dtype: str = "int8",
    group_size: int = 32,
    quantize_embedding: bool = True,
) -> "PreTrainedModel":
    """Prepare model for quantization aware training using torchao."""
    if not model_args.enable_qat:
        return model
        
    try:
        import torchao
        from torchao.quantization.quant_api import IntXQuantizationAwareTrainingConfig, FakeQuantizeConfig
        
        logger.info_rank0("Preparing model for Quantization Aware Training (QAT)...")
        
        # Create QAT configuration
        activation_dtype = model_args.qat_activation_dtype or activation_dtype
        weight_dtype = model_args.qat_weight_dtype or weight_dtype
        group_size = model_args.qat_group_size or group_size
        quantize_embedding = model_args.qat_quantize_embedding if model_args.qat_quantize_embedding is not None else quantize_embedding
        
        # Configure weight quantization
        weight_fake_quant_config = FakeQuantizeConfig(
            dtype=getattr(torch, weight_dtype) if hasattr(torch, weight_dtype) else torch.int8,
            group_size=group_size,
        )
        
        # Configure activation quantization if specified
        activation_fake_quant_config = None
        if activation_dtype:
            activation_fake_quant_config = FakeQuantizeConfig(
                dtype=getattr(torch, activation_dtype) if hasattr(torch, activation_dtype) else torch.int8,
            )
        
        # Create QAT config
        qat_config = IntXQuantizationAwareTrainingConfig(
            weight_fake_quant_config=weight_fake_quant_config,
            activation_fake_quant_config=activation_fake_quant_config,
            quantize_embedding=quantize_embedding,
        )
        
        # Apply QAT to model
        from torchao.quantization.quant_api import prepare_model_for_quantization_aware_training
        model = prepare_model_for_quantization_aware_training(model, qat_config)
        
        logger.info_rank0(
            f"Successfully prepared model for QAT with weight_dtype={weight_dtype}, "
            f"activation_dtype={activation_dtype}, group_size={group_size}"
        )
        
        return model
        
    except ImportError:
        logger.warning_rank0(
            "torchao not available for QAT. Please install with: pip install torchao>=0.8.0"
        )
        return model
    except Exception as e:
        logger.warning_rank0(f"Failed to prepare model for QAT: {e}")
        return model


def load_hf_kernel(kernel_name: str):
    """Load a kernel from the Huggingface kernels package."""
    try:
        from kernels import get_kernel
        logger.info_rank0(f"Loading kernel: {kernel_name}")
        kernel = get_kernel(kernel_name)
        logger.info_rank0(f"Successfully loaded kernel: {kernel_name}")
        return kernel
    except ImportError:
        logger.warning_rank0(
            "kernels package not available. Please install with: pip install kernels>=0.9.0"
        )
        return None
    except Exception as e:
        logger.warning_rank0(f"Failed to load kernel {kernel_name}: {e}")
        return None


def apply_hf_kernels(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """Apply Huggingface kernels to model if enabled."""
    if not model_args.use_kernels or not model_args.kernel_name:
        return
    
    kernel = load_hf_kernel(model_args.kernel_name)
    if kernel is None:
        return
    
    # This is a placeholder for kernel application logic
    # The specific implementation depends on the kernel type and model architecture
    logger.info_rank0(f"Kernel {model_args.kernel_name} loaded and ready for use.")
    
    # Store kernel reference in model for use during training
    if not hasattr(model, '_hf_kernels'):
        model._hf_kernels = {}
    model._hf_kernels[model_args.kernel_name] = kernel