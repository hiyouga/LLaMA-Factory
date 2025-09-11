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

    # If user provided explicit dict, respect it and set lengths accordingly
    if isinstance(model_args.rope_scaling, dict):
        rope_kwargs = dict(model_args.rope_scaling)
        rope_type = rope_kwargs.get("rope_type")
        factor = float(rope_kwargs.get("factor"))
        orig = int(rope_kwargs.get("original_max_position_embeddings"))

        new_max_length = int(orig * factor)
        setattr(config, "max_position_embeddings", new_max_length)
        setattr(config, "rope_scaling", rope_kwargs)
        # Detailed diagnostics to aid debugging
        rope_theta = getattr(config, "rope_theta", None)
        model_type = getattr(config, "model_type", None)
        attn_impl = getattr(config, "_attn_implementation", None)
        logger.info_rank0(
            "RoPE config (explicit): "
            f"model_type={model_type}, rope_type={rope_type}, factor={factor}, "
            f"original_max_position_embeddings={orig}, new_max_position_embeddings={new_max_length}, "
            f"rope_theta={rope_theta}, attn_impl={attn_impl}"
        )
        # Heuristic warning for DeepSpeed + FlashAttention + YaRN
        try:
            from transformers.integrations import is_deepspeed_zero3_enabled

            if rope_type == "yarn" and is_deepspeed_zero3_enabled():
                logger.warning_rank0(
                    "DeepSpeed detected with YaRN scaling. If you see instability (NaNs), try one of: "
                    "(1) set model_args.attn='sdpa' to avoid FlashAttention kernels; "
                    "(2) reduce factor temporarily (e.g., 2.0→3.0→4.0); "
                    "(3) verify rope_theta and original_max_position_embeddings match the pretrained base."
                )
        except Exception:
            pass
        logger.info_rank0(f"Enlarge max model length from {old_max_length} to {new_max_length}.")
        return

    # Enum/string path: compute factor from target
    # Determine target length: model_capacity overrides model_max_length (cutoff_len)
    target_length = model_args.model_capacity or model_args.model_max_length

    if target_length is not None:  # training
        if target_length <= old_max_length:
            logger.warning_rank0("Target length is smaller than max length. Disabling rope scaling.")
            return

        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        rope_factor = float(math.ceil(target_length / old_max_length))

        if model_args.model_capacity is not None:
            logger.info_rank0(
                f"Using model_capacity={model_args.model_capacity} to determine RoPE scaling "
                f"(independent of cutoff_len={model_args.model_max_length})."
            )
    else:  # inference
        rope_factor = 2.0

    rope_kwargs = {
        "rope_type": getattr(model_args.rope_scaling, "value", model_args.rope_scaling),  # handle enum
        "factor": rope_factor,
    }
    new_max_length = old_max_length * rope_factor
    setattr(config, "max_position_embeddings", new_max_length)
    logger.info_rank0(f"Enlarge max model length from {old_max_length} to {new_max_length}.")

    rope_type_name = rope_kwargs["rope_type"]
    if rope_type_name in (
        getattr(RopeScaling.DYNAMIC, "value", "dynamic"),
        getattr(RopeScaling.YARN, "value", "yarn"),
    ):
        rope_kwargs["original_max_position_embeddings"] = old_max_length
    elif rope_type_name == getattr(RopeScaling.LLAMA3, "value", "llama3"):
        rope_kwargs["original_max_position_embeddings"] = old_max_length
        rope_kwargs["low_freq_factor"] = 1.0
        rope_kwargs["high_freq_factor"] = 4.0

    setattr(config, "rope_scaling", rope_kwargs)
    rope_theta = getattr(config, "rope_theta", None)
    model_type = getattr(config, "model_type", None)
    attn_impl = getattr(config, "_attn_implementation", None)
    logger.info_rank0(
        "RoPE config (auto): "
        f"model_type={model_type}, rope_type={rope_kwargs['rope_type']}, factor={rope_kwargs['factor']}, "
        f"original_max_position_embeddings={rope_kwargs.get('original_max_position_embeddings')}, "
        f"new_max_position_embeddings={new_max_length}, rope_theta={rope_theta}, attn_impl={attn_impl}"
    )
