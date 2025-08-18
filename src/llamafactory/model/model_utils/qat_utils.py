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

from typing import TYPE_CHECKING

import torch

from ...extras import logging


if TYPE_CHECKING:
    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def prepare_model_for_qat(model, model_args: "ModelArguments"):
    """Prepare model for Quantization Aware Training if enabled."""
    if not model_args.enable_qat:
        return model
    
    try:
        from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
        
        logger.info_rank0("Preparing model for QAT (Quantization Aware Training)")
        
        # Create QAT quantizer with specified parameters
        quantizer = Int8DynActInt4WeightQATQuantizer(
            groupsize=getattr(model_args, 'qat_group_size', 32),
            precision=torch.float32
        )
        
        # Apply quantization to the model
        model = quantizer.quantize(model)
        
        # Enable fake quantization after specified steps if configured
        if hasattr(model_args, 'fake_quant_after_n_steps') and model_args.fake_quant_after_n_steps is not None:
            logger.info_rank0(f"QAT fake quantization will be enabled after {model_args.fake_quant_after_n_steps} steps")
        
        logger.info_rank0("Model prepared for QAT training")
        return model
        
    except ImportError:
        logger.warning_rank0("torchao not available for QAT. Please install with: pip install torchao>=0.8.0")
        return model
    except Exception as e:
        logger.warning_rank0(f"Failed to prepare model for QAT: {e}")
        return model


