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


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning_rank0("Current model does not support RoPE scaling.")
        return

    if hasattr(config, "max_position_embeddings"):
        old_max_length = getattr(config, "max_position_embeddings", None)
    else:
        logger.warning_rank0("Cannot find the max position embeddings in the config.")
        return

    if model_args.model_max_length is not None:  # training
        if model_args.model_max_length <= old_max_length:
            logger.warning_rank0("Input length is smaller than max length. Disabling rope scaling.")
            return

        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        rope_factor = float(math.ceil(model_args.model_max_length / old_max_length))
    else:  # inference
        rope_factor = 2.0

    rope_kwargs = {
        "rope_type": getattr(model_args.rope_scaling, "value", model_args.rope_scaling),  # handle enum
        "factor": rope_factor,
    }
    setattr(config, "max_position_embeddings", old_max_length * rope_factor)
    logger.info_rank0(f"Enlarge max model length from {old_max_length} to {old_max_length * rope_factor}.")

    if model_args.rope_scaling in [RopeScaling.DYNAMIC, RopeScaling.YARN]:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
    elif model_args.rope_scaling == RopeScaling.LLAMA3:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
        rope_kwargs["low_freq_factor"] = 1.0
        rope_kwargs["high_freq_factor"] = 4.0

    setattr(config, "rope_scaling", rope_kwargs)
    logger.info_rank0(
        f"Using {rope_kwargs['rope_type']} scaling strategy and setting scaling factor to {rope_kwargs['factor']}."
    )
