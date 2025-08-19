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

from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available, is_torch_sdpa_available

from ...extras import logging
from ...extras.constants import AttentionImplementation


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    # If no attn specified, use default (eager)
    if model_args.attn is None:
        requested_attn_implementation = "eager"
        logger.info_rank0("No attention implementation specified, using default eager attention.")
    else:
        attn_value = model_args.attn.strip()

        # Check if it's a HuggingFace kernel (contains '/' or starts with 'hf:')
        if (
            "/" in attn_value
            or attn_value.startswith("hf:")
            or any(kernel_word in attn_value for kernel_word in ["kernel", "flash", "attn"])
        ):
            # Handle HuggingFace kernel
            kernel_name = attn_value[3:] if attn_value.startswith("hf:") else attn_value
            try:
                from kernels import get_kernel

                kernel_impl = get_kernel(kernel_name)
                if kernel_impl is not None:
                    logger.info_rank0(f"Using HuggingFace kernel: {kernel_name}")
                    setattr(config, "_hf_kernel", kernel_name)
                    return
                else:
                    logger.warning_rank0(
                        f"HuggingFace kernel '{kernel_name}' not found. Falling back to eager attention."
                    )
                    requested_attn_implementation = "eager"
            except ImportError:
                logger.warning_rank0(
                    "HuggingFace kernels not available. Install with: pip install -e .[hf-kernels]. Falling back to eager attention."
                )
                requested_attn_implementation = "eager"
            except Exception as e:
                logger.warning_rank0(
                    f"Failed to load HuggingFace kernel {kernel_name}: {e}. Falling back to eager attention."
                )
                requested_attn_implementation = "eager"
        else:
            # Handle standard attention implementations
            if attn_value == AttentionImplementation.EAGER:
                requested_attn_implementation = "eager"
            elif attn_value == AttentionImplementation.SDPA:
                if not is_torch_sdpa_available():
                    logger.warning_rank0("torch>=2.1.1 is required for SDPA attention. Falling back to eager.")
                    requested_attn_implementation = "eager"
                else:
                    requested_attn_implementation = "sdpa"
            elif attn_value == AttentionImplementation.FA2:
                if not is_flash_attn_2_available():
                    logger.warning_rank0("FlashAttention-2 is not installed. Falling back to eager.")
                    requested_attn_implementation = "eager"
                else:
                    requested_attn_implementation = "flash_attention_2"
            elif attn_value == AttentionImplementation.FA3:
                if not is_flash_attn_3_available():
                    logger.warning_rank0("FlashAttention-3 is not installed. Falling back to eager.")
                    requested_attn_implementation = "eager"
                else:
                    requested_attn_implementation = "flash_attention_3"
            else:
                logger.warning_rank0(f"Unknown attention implementation '{attn_value}'. Falling back to eager.")
                requested_attn_implementation = "eager"

    # Special handling for model-specific requirements
    if getattr(config, "model_type", None) == "gemma2":
        if requested_attn_implementation == "sdpa":
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while SDPA attention does not support it. "
                "Consider using 'fa2' instead."
            )
        elif requested_attn_implementation not in ["flash_attention_2", "eager"]:
            logger.warning_rank0("Gemma-2 works best with FlashAttention-2 or eager attention.")

    # Apply the attention implementation to config
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    # Check for HuggingFace kernel first
    hf_kernel = getattr(config, "_hf_kernel", None)
    if hf_kernel is not None:
        logger.info_rank0(f"Using HuggingFace kernel: {hf_kernel} for faster training and inference.")
        return

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "flash_attention_3":
        logger.info_rank0("Using FlashAttention-3 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    elif attn_implementation == "eager":
        logger.info_rank0("Using eager (vanilla) attention implementation.")
    else:
        logger.info_rank0(f"Using {attn_implementation or 'default'} attention implementation.")
