# Copyright 2025 LMSYS and the LlamaFactory team.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This code is inspired by the LMSYS's FastChat library.
# https://github.com/lm-sys/FastChat/blob/v0.2.30/fastchat/train/train.py
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

import math
from typing import TYPE_CHECKING

from ...extras import logging
from ...extras.constants import RopeScaling


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning_rank0("Current model does not support RoPE scaling.")
        return

    rope_kwargs = {"rope_type": getattr(model_args.rope_scaling, "value", model_args.rope_scaling)}  # handle enum
    if model_args.model_max_length is not None:
        if is_trainable and model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        current_max_length = getattr(config, "max_position_embeddings", None)
        if (not current_max_length) or model_args.model_max_length <= current_max_length:
            logger.warning_rank0("Input length is smaller than max length. Disabling rope scaling.")
            return

        logger.info_rank0(f"Enlarge max model length from {current_max_length} to {model_args.model_max_length}.")
        setattr(config, "max_position_embeddings", model_args.model_max_length)
        rope_kwargs["factor"] = float(math.ceil(model_args.model_max_length / current_max_length))
        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            rope_kwargs["original_max_position_embeddings"] = current_max_length
        elif model_args.rope_scaling == RopeScaling.LLAMA3:
            rope_kwargs["original_max_position_embeddings"] = current_max_length
            rope_kwargs["low_freq_factor"] = 1.0
            rope_kwargs["high_freq_factor"] = 4.0
    else:
        rope_kwargs["factor"] = 2.0

    setattr(config, "rope_scaling", rope_kwargs)
    logger.info_rank0(
        f"Using {rope_kwargs['rope_type']} scaling strategy and setting scaling factor to {rope_kwargs['factor']}."
    )
