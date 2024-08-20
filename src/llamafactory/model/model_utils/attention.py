# Copyright 2024 the LlamaFactory team.
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

from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
from transformers.utils.versions import require_version

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = get_logger(__name__)


def configure_attn_implementation(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> None:
    if getattr(config, "model_type", None) == "gemma2" and is_trainable:
        if model_args.flash_attn == "auto" or model_args.flash_attn == "fa2":
            if is_flash_attn_2_available():
                require_version("transformers>=4.42.4", "To fix: pip install transformers>=4.42.4")
                require_version("flash_attn>=2.6.3", "To fix: pip install flash_attn>=2.6.3")
                logger.warning("Gemma-2 should use flash attention 2, change `flash_attn` to fa2.")
                model_args.flash_attn = "fa2"
            else:
                logger.warning("Gemma-2 should use eager attention, change `flash_attn` to disabled.")
                model_args.flash_attn = "disabled"
        elif model_args.flash_attn == "sdpa":
            logger.warning("Gemma-2 should use soft-capping attention, while the SDPA attention does not support it.")

    if model_args.flash_attn == "auto":
        return

    elif model_args.flash_attn == "disabled":
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == "sdpa":
        if not is_torch_sdpa_available():
            logger.warning("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == "fa2":
        if not is_flash_attn_2_available():
            logger.warning("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError("Unknown attention type: {}".format(model_args.flash_attn))

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info("Using torch SDPA for faster training and inference.")
    else:
        logger.info("Using vanilla attention implementation.")
