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
            from torchao.float8 import Float8LinearConfig

            # Use rowwise scaling for better performance (as recommended by torchao)
            # Configure alignment requirements for FP8 kernels
            config = Float8LinearConfig.from_recipe_name("rowwise")

            # Enable alignment for better kernel performance
            if hasattr(config, 'enable_amax_init'):
                config.enable_amax_init = True
            if hasattr(config, 'enable_pre_and_post_forward'):
                config.enable_pre_and_post_forward = True

        # Create module filter function to skip problematic layers
        # TorchAO FP8 requires dimensions divisible by 16 for optimal kernels
        def module_filter_func(module, layer_name):
            # Skip embedding and output layers for numerical stability
            skip_layers = ["embed", "lm_head", "output", "classifier"]
            if any(skip_name in layer_name.lower() for skip_name in skip_layers):
                return False

            # Only convert Linear layers
            if not (hasattr(module, 'weight') and len(module.weight.shape) == 2):
                return False

            # Check dimension alignment for FP8 kernels
            weight = module.weight
            in_features, out_features = weight.shape[1], weight.shape[0]

            # Skip layers with dimensions not divisible by 16 to avoid kernel errors
            if in_features % 16 != 0 or out_features % 16 != 0:
                logger.debug(f"Skipping layer {layer_name} with dimensions {out_features}x{in_features} (not divisible by 16)")
                return False

            return True

        # Map FSDP all-gather setting if available (this affects the underlying implementation)
        if hasattr(model_args, "fp8_enable_fsdp_float8_all_gather") and model_args.fp8_enable_fsdp_float8_all_gather:
            logger.info_rank0("FSDP float8 all-gather optimization requested")

        return [AORecipeKwargs(config=config, module_filter_func=module_filter_func)]
    except Exception as e:
        logger.info_rank0(f"Failed to create FP8 configuration: {e}")
        return []


def get_fp8_mixed_precision(model_args: "ModelArguments") -> Optional[str]:
    """Get the mixed precision setting for Accelerate when using FP8.

    Args:
        model_args: Model arguments containing FP8 configuration

    Returns:
        "fp8" if FP8 is enabled, None otherwise
    """
    return "fp8" if model_args.fp8 else None


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
        # Use the specific recipe kwargs classes instead of deprecated FP8RecipeKwargs
        from accelerate.utils import MSAMPRecipeKwargs, TERecipeKwargs

        backend = getattr(model_args, "fp8_backend", "auto")

        # Use Transformer Engine for DeepSpeed FP8 (most stable)
        if backend == "auto" or backend == "te":
            try:
                # Create TERecipeKwargs for Transformer Engine
                fp8_kwargs = {"use_autocast_during_eval": False}

                # Add FSDP float8 all-gather if requested
                if hasattr(model_args, "fp8_enable_fsdp_float8_all_gather") and model_args.fp8_enable_fsdp_float8_all_gather:
                    logger.info_rank0("FSDP float8 all-gather optimization enabled for DeepSpeed")

                logger.info_rank0("Creating DeepSpeed FP8 configuration with Transformer Engine backend")
                return [TERecipeKwargs(**fp8_kwargs)]
            except Exception as te_error:
                logger.warning_rank0(f"Transformer Engine not available: {te_error}. Falling back to MS-AMP.")
                # Fall back to MS-AMP
                fp8_kwargs = {"opt_level": "O2"}
                logger.info_rank0("Creating DeepSpeed FP8 configuration with MS-AMP backend (fallback)")
                return [MSAMPRecipeKwargs(**fp8_kwargs)]

        elif backend == "msamp":
            # MS-AMP backend for DeepSpeed
            fp8_kwargs = {"opt_level": "O2"}
            logger.info_rank0("Creating DeepSpeed FP8 configuration with MS-AMP backend")
            return [MSAMPRecipeKwargs(**fp8_kwargs)]

        else:
            logger.warning_rank0(f"Backend '{backend}' not supported with DeepSpeed, trying Transformer Engine, falling back to MS-AMP")
            try:
                return [TERecipeKwargs(use_autocast_during_eval=False)]
            except Exception:
                return [MSAMPRecipeKwargs(opt_level="O2")]

    except ImportError:
        logger.warning_rank0("TERecipeKwargs/MSAMPRecipeKwargs not available - please upgrade accelerate")
        return []
    except Exception as e:
        logger.info_rank0(f"Failed to create DeepSpeed FP8 configuration: {e}")
        return []


def configure_fp8_environment(model_args: "ModelArguments") -> None:
    """Centralized FP8 environment variable configuration for HuggingFace Accelerate.
    
    Args:
        model_args: Model arguments containing FP8 configuration
    """
    import importlib.util
    import os

    if not model_args.fp8:
        return

    # Always set mixed precision to fp8 first
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    logger.info_rank0("Set ACCELERATE_MIXED_PRECISION=fp8")

    # Configure FP8 backend and options
    backend = getattr(model_args, 'fp8_backend', 'auto')
    if backend != 'auto':
        os.environ["FP8_BACKEND"] = backend
        logger.info_rank0(f"Set FP8_BACKEND={backend}")

    # Create and validate recipe kwargs (for logging/debugging)
    if importlib.util.find_spec("deepspeed") is not None:
        deepspeed_fp8_kwargs = create_deepspeed_fp8_kwargs(model_args)
        logger.info_rank0(f"DeepSpeed FP8 kwargs created: {deepspeed_fp8_kwargs}")
    else:
        fp8_kwargs = create_fp8_kwargs(model_args)
        logger.info_rank0(f"Native FP8 kwargs created: {fp8_kwargs}")

        if hasattr(model_args, 'fp8_enable_fsdp_float8_all_gather') and model_args.fp8_enable_fsdp_float8_all_gather:
            os.environ["FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER"] = "true"
            logger.info_rank0("Set FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER=true")

    logger.info_rank0("FP8 environment variables configured for Accelerate")
