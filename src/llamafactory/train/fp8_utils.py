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

from typing import TYPE_CHECKING, Any, Optional

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import ModelArguments

logger = logging.get_logger(__name__)


def create_fp8_kwargs(model_args: "ModelArguments") -> list[Any]:
    """Create AORecipeKwargs for FP8 training with HuggingFace Accelerate.
    
    Args:
        model_args: Model arguments containing FP8 configuration
        
    Returns:
        List containing AORecipeKwargs if FP8 is enabled, empty list otherwise
    """
    if not model_args.fp8:
        return []

    try:
        from accelerate.utils import AORecipeKwargs

        # Map LLaMA-Factory FP8 settings to AORecipeKwargs
        fp8_config = {}

        # Set backend if specified (default to auto selection)
        backend = getattr(model_args, 'fp8_backend', 'auto')
        if backend != 'auto':
            fp8_config['backend'] = backend

        # Map FSDP all-gather setting if available
        if hasattr(model_args, 'fp8_enable_fsdp_float8_all_gather') and model_args.fp8_enable_fsdp_float8_all_gather:
            # This setting may need to be handled differently depending on the backend
            fp8_config['enable_fsdp_float8_all_gather'] = True

        logger.info_rank0(f"Creating FP8 configuration with backend: {backend}")
        if fp8_config.get('enable_fsdp_float8_all_gather'):
            logger.info_rank0("FSDP float8 all-gather optimization enabled")

        return [AORecipeKwargs(config=fp8_config)]

    except ImportError as e:
        logger.error_rank0(f"Failed to import AORecipeKwargs: {e}")
        logger.error_rank0("Please ensure accelerate is installed and up to date")
        return []
    except Exception as e:
        logger.error_rank0(f"Failed to create FP8 configuration: {e}")
        return []


def get_fp8_mixed_precision(model_args: "ModelArguments") -> Optional[str]:
    """Get the mixed precision setting for Accelerate when using FP8.
    
    Args:
        model_args: Model arguments containing FP8 configuration
        
    Returns:
        "fp8" if FP8 is enabled, None otherwise
    """
    return "fp8" if model_args.fp8 else None


def validate_fp8_requirements() -> bool:
    """Validate that the system meets FP8 training requirements.
    
    Returns:
        True if FP8 requirements are met, False otherwise
    """
    try:
        import torch

        # Check PyTorch version requirement
        pytorch_version = torch.__version__.split("+")[0]  # Remove +cu121 etc.
        major, minor = map(int, pytorch_version.split(".")[:2])
        if major < 2 or (major == 2 and minor < 7):
            logger.warning_rank0(
                f"FP8 training requires PyTorch 2.7+, but found {pytorch_version}"
            )
            return False

        # Check GPU architecture
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability < (9, 0):  # Hopper architecture requirement
                logger.warning_rank0(
                    f"FP8 training requires Hopper architecture (compute capability 9.0+), "
                    f"but found {device_capability}"
                )
                return False
        else:
            logger.warning_rank0("CUDA not available - FP8 requires CUDA-enabled GPUs")
            return False

        return True

    except Exception as e:
        logger.warning_rank0(f"Failed to validate FP8 requirements: {e}")
        return False
