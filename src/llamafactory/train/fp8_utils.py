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
        List containing AORecipeKwargs if FP8 is enabled and supported, empty list otherwise
    """
    if not model_args.fp8:
        return []

    try:
        # Check if AORecipeKwargs is available (Accelerate 1.8.0+)
        from accelerate.utils import AORecipeKwargs

        backend = getattr(model_args, "fp8_backend", "auto")
        logger.info_rank0(f"Creating FP8 configuration with backend: {backend}")

        # Create Float8LinearConfig if torchao backend is used
        config = None
        if backend == "torchao" or backend == "auto":
            try:
                from torchao.float8 import Float8LinearConfig

                # Use rowwise scaling for better performance (as recommended by torchao)
                config = Float8LinearConfig.from_recipe_name("rowwise")
                logger.info_rank0("Using torchao Float8LinearConfig with rowwise scaling")
            except ImportError:
                logger.warning_rank0("torchao not available, using default FP8 configuration")

        # Create module filter function to optionally skip first/last layers
        # TorchAO recommends keeping first/last layers at full precision for stability
        def module_filter_func(module, layer_name):
            # Skip embedding and output layers for numerical stability
            skip_layers = ["embed", "lm_head", "output", "classifier"]
            if any(skip_name in layer_name.lower() for skip_name in skip_layers):
                return False
            # Only convert Linear layers
            return hasattr(module, 'weight') and len(module.weight.shape) == 2

        # Map FSDP all-gather setting if available (this affects the underlying implementation)
        if hasattr(model_args, "fp8_enable_fsdp_float8_all_gather") and model_args.fp8_enable_fsdp_float8_all_gather:
            logger.info_rank0("FSDP float8 all-gather optimization requested")

        return [AORecipeKwargs(config=config, module_filter_func=module_filter_func)]

    except ImportError as e:
        raise ImportError(
            "AORecipeKwargs not available. FP8 with Accelerate requires Accelerate >= 1.8.0. "
            "Please upgrade accelerate: pip install 'accelerate>=1.8.0'"
        ) from e
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
            logger.warning_rank0(f"FP8 training requires PyTorch 2.7+, but found {pytorch_version}")
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


def check_deepspeed_fp8_compatibility() -> bool:
    """Check if DeepSpeed supports FP8 training.

    Returns:
        True if FP8 can be used with or without DeepSpeed, False if incompatible version
    """
    try:
        import deepspeed
        deepspeed_version = deepspeed.__version__

        # Parse version to check if it's 0.17.3+
        version_parts = deepspeed_version.split(".")
        major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2].split("+")[0])

        if major == 0 and minor >= 17 and (minor > 17 or patch >= 3):
            logger.info_rank0(f"DeepSpeed {deepspeed_version} detected - FP8 support available")
            return True
        else:
            logger.warning_rank0(
                f"DeepSpeed {deepspeed_version} detected - FP8 requires DeepSpeed 0.17.3+. "
                f"Please upgrade: pip install 'deepspeed>=0.17.3'"
            )
            return False
    except ImportError:
        return True  # DeepSpeed not installed, FP8 can be used with native Accelerate


def create_deepspeed_fp8_kwargs(model_args: "ModelArguments") -> list[Any]:
    """Create FP8RecipeKwargs for DeepSpeed FP8 training.

    Args:
        model_args: Model arguments containing FP8 configuration

    Returns:
        List containing appropriate RecipeKwargs for DeepSpeed FP8
    """
    if not model_args.fp8:
        return []

    try:
        from accelerate.utils import FP8RecipeKwargs

        backend = getattr(model_args, "fp8_backend", "auto")

        # Use Transformer Engine for DeepSpeed FP8 (most stable)
        if backend == "auto" or backend == "te":
            # Create FP8RecipeKwargs for Transformer Engine
            fp8_kwargs = {"backend": "te", "use_during_eval": False}

            # Add FSDP float8 all-gather if requested
            if hasattr(model_args, "fp8_enable_fsdp_float8_all_gather") and model_args.fp8_enable_fsdp_float8_all_gather:
                logger.info_rank0("FSDP float8 all-gather optimization enabled for DeepSpeed")

            logger.info_rank0("Creating DeepSpeed FP8 configuration with Transformer Engine backend")
            return [FP8RecipeKwargs(**fp8_kwargs)]

        elif backend == "msamp":
            # MS-AMP backend for DeepSpeed
            fp8_kwargs = {"backend": "msamp", "optimization_level": "O2"}
            logger.info_rank0("Creating DeepSpeed FP8 configuration with MS-AMP backend")
            return [FP8RecipeKwargs(**fp8_kwargs)]

        else:
            logger.warning_rank0(f"Backend '{backend}' not supported with DeepSpeed, using Transformer Engine")
            return [FP8RecipeKwargs(backend="te", use_during_eval=False)]

    except ImportError:
        logger.warning_rank0("FP8RecipeKwargs not available - please upgrade accelerate")
        return []
    except Exception as e:
        logger.error_rank0(f"Failed to create DeepSpeed FP8 configuration: {e}")
        return []
