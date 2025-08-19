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
            if hasattr(config, "enable_amax_init"):
                config.enable_amax_init = True
            if hasattr(config, "enable_pre_and_post_forward"):
                config.enable_pre_and_post_forward = True

        # Create module filter function to skip problematic layers
        # TorchAO FP8 requires dimensions divisible by 16 for optimal kernels
        def module_filter_func(module, layer_name):
            # Skip embedding and output layers for numerical stability
            skip_layers = ["embed", "lm_head", "output", "classifier"]
            if any(skip_name in layer_name.lower() for skip_name in skip_layers):
                return False

            # Only convert Linear layers
            if not (hasattr(module, "weight") and len(module.weight.shape) == 2):
                return False

            # Check dimension alignment for FP8 kernels
            weight = module.weight
            in_features, out_features = weight.shape[1], weight.shape[0]

            # Skip layers with dimensions not divisible by 16 to avoid kernel errors
            if in_features % 16 != 0 or out_features % 16 != 0:
                logger.debug(
                    f"Skipping layer {layer_name} with dimensions {out_features}x{in_features} (not divisible by 16)"
                )
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


def configure_fp8_environment(model_args: "ModelArguments") -> None:
    """Configure FP8 environment for HuggingFace Accelerate.

    FP8 training is handled entirely through HuggingFace Accelerate, regardless of whether
    DeepSpeed or FSDP is used for distributed training. This function sets up the environment
    variables and validates the FP8 configuration.

    Args:
        model_args: Model arguments containing FP8 configuration
    """
    import os

    if not model_args.fp8:
        return

    # Set mixed precision to fp8 for HuggingFace Accelerate
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
    logger.info_rank0("Set ACCELERATE_MIXED_PRECISION=fp8")

    # Configure FP8 backend and options
    backend = getattr(model_args, "fp8_backend", "auto")
    if backend != "auto":
        os.environ["FP8_BACKEND"] = backend
        logger.info_rank0(f"Set FP8_BACKEND={backend}")

    # Create and validate FP8 recipe kwargs (for logging/debugging)
    fp8_kwargs = create_fp8_kwargs(model_args)
    logger.info_rank0(f"FP8 AORecipeKwargs created: {len(fp8_kwargs)} items")

    # Enable FSDP float8 all-gather optimization if requested
    if hasattr(model_args, "fp8_enable_fsdp_float8_all_gather") and model_args.fp8_enable_fsdp_float8_all_gather:
        os.environ["FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER"] = "true"
        logger.info_rank0("Set FP8_ENABLE_FSDP_FLOAT8_ALL_GATHER=true")

    logger.info_rank0("FP8 environment configured - all FP8 training handled by HuggingFace Accelerate")


def verify_fp8_status(accelerator, model_args: "ModelArguments") -> None:
    """Verify that FP8 training is actually working after model preparation.

    Args:
        accelerator: The HuggingFace Accelerator instance
        model_args: Model arguments containing FP8 configuration
    """
    if not model_args.fp8:
        return

    # Check Accelerate's FP8 status
    fp8_enabled = getattr(accelerator, "fp8_enabled", False)
    fp8_backend_type = getattr(accelerator, "fp8_backend", "UNKNOWN")

    backend = getattr(model_args, "fp8_backend", "auto")
    if backend == "torchao" or backend == "auto":
        logger.info_rank0(
            "FP8 training enabled with TorchAO backend. For optimal performance, "
            "ensure model layer dimensions are mostly divisible by 16. "
            "If you encounter issues, try fp8_backend='te' with Transformer Engine."
        )
    else:
        logger.info_rank0(f"FP8 training enabled with {backend} backend.")

    logger.info_rank0(f"Accelerate FP8 status - enabled: {fp8_enabled}, backend: {fp8_backend_type}")

    if not fp8_enabled:
        logger.info_rank0("WARNING: FP8 was requested but Accelerate shows fp8_enabled=False. FP8 may not be working.")
