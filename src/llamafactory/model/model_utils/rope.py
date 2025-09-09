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
from .dynamic_rope import DynamicYarnRotaryEmbedding


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
        logger.info_rank0(
            f"Set RoPE scaling from config dict: type={rope_type}, factor={factor}, original_max_position_embeddings={orig}."
        )
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


def apply_dynamic_yarn_rope(model) -> None:
    """Swap static YaRN rotary modules for a dynamic variant when requested.

    Activation conditions:
    - config.rope_scaling is a dict
    - rope_type == 'yarn'
    - dynamic == True
    Only touches submodules exposing attribute `rotary_emb`.
    """
    config = getattr(model, "config", None)
    rope_scaling = getattr(config, "rope_scaling", None)

    if not isinstance(rope_scaling, dict):
        return

    if rope_scaling.get("rope_type") != "yarn" or not rope_scaling.get("dynamic", False):
        return

    replaced = 0
    for name, module in model.named_modules():
        if not hasattr(module, "rotary_emb"):
            continue
        try:
            orig = getattr(module, "rotary_emb")
        except Exception:
            continue
        if not isinstance(orig, object):
            continue
        try:
            dyn = DynamicYarnRotaryEmbedding(orig, config)
        except Exception as e:
            # Skip on mismatch and warn once
            logger.debug_rank0(f"Skip dynamic YaRN for {name}: {e}")
            continue
        setattr(module, "rotary_emb", dyn)
        replaced += 1

    if replaced > 0:
        logger.info_rank0(f"Enabled dynamic YaRN rotary for {replaced} attention module(s).")
    else:
        logger.warning_rank0("Dynamic YaRN requested but no rotary_emb modules were replaced.")
