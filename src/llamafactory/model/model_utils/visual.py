# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava/modeling_llava.py
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

from typing import TYPE_CHECKING, List, Sequence, Set, Tuple, Union

import torch
import transformers.models
from transformers.activations import ACT2FN
from transformers.utils import logging

from ...extras.logging import get_logger


if TYPE_CHECKING:
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)
transformers_logger = logging.get_logger(__name__)


class LlavaMultiModalProjectorForYiVL(torch.nn.Module):
    def __init__(self, config: "LlavaConfig") -> None:
        super().__init__()

        self.config = config
        if config is None:
            return

        self.linear_1 = torch.nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, image_features: "torch.Tensor") -> "torch.Tensor":
        hidden_states = self.linear_1(image_features)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        hidden_states = self.linear_4(hidden_states)
        if hidden_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.linear_1.weight.dtype

            transformers_logger.warning_once("The hidden states seems to be silently casted in float32.")
            hidden_states = hidden_states.to(target_dtype)

        return hidden_states


class LlavaMultiModalProjectorForYiVLForVLLM(LlavaMultiModalProjectorForYiVL):
    def __init__(self, vision_hidden_size: int, text_hidden_size: int, projector_hidden_act: str) -> None:
        super().__init__(config=None)

        self.linear_1 = torch.nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.linear_2 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.linear_3 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        self.linear_4 = torch.nn.LayerNorm(text_hidden_size, bias=True)
        self.act = ACT2FN[projector_hidden_act]


def autocast_projector_dtype(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Casts projector output to half precision for fine-tuning quantized VLMs.
    """

    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)
        if model_type in ["llava", "paligemma"]:
            mm_projector: "torch.nn.Module" = getattr(model, "multi_modal_projector")
        elif model_type == "qwen2_vl":
            mm_projector: "torch.nn.Module" = getattr(getattr(model, "visual"), "merger")
        else:
            return

        logger.info("Casting multimodal projector outputs in {}.".format(model_args.compute_dtype))
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def configure_visual_model(config: "PretrainedConfig") -> None:
    r"""
    Patches VLMs before loading them.
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "llava":  # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))

    if getattr(config, "is_yi_vl_derived_model", None):
        logger.info("Detected Yi-VL model, applying projector patch.")
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> Set[str]:
    r"""
    Freezes vision tower and language model for VLM full/freeze tuning.
    """
    model_type = getattr(config, "model_type", None)
    forbidden_modules = set()
    if model_type in ["llava", "paligemma"]:
        if finetuning_args.freeze_vision_tower:
            forbidden_modules.add("vision_tower")

        if finetuning_args.train_mm_proj_only:
            forbidden_modules.add("language_model")

    elif model_type == "qwen2_vl":
        if finetuning_args.freeze_vision_tower:
            forbidden_modules.add("visual")

        if finetuning_args.train_mm_proj_only:
            raise ValueError("Qwen2-VL models do not support `train_mm_proj_only`.")

    return forbidden_modules


def get_image_seqlen(config: "PretrainedConfig") -> int:
    r"""
    Computes the number of special tokens per image.
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "llava":
        image_seqlen = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        if getattr(config, "vision_feature_select_strategy", "default") == "full":  # add [CLS] token
            image_seqlen += 1
    elif model_type == "paligemma":
        image_seqlen = config.vision_config.num_image_tokens
    elif model_type == "qwen2_vl":  # variable length
        image_seqlen = -1

    return image_seqlen


def patch_target_modules(
    config: "PretrainedConfig", finetuning_args: "FinetuningArguments", target_modules: Sequence[str]
) -> Union[str, List[str]]:
    r"""
    Freezes vision tower for VLM LoRA tuning.
    """
    model_type = getattr(config, "model_type", None)
    if finetuning_args.freeze_vision_tower:
        if model_type in ["llava", "paligemma"]:
            return "^(?!.*vision_tower).*(?:{}).*".format("|".join(target_modules))
        elif model_type == "qwen2_vl":
            return "^(?!.*visual).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules
    else:
        if model_type == "qwen2_vl":
            return "^(?!.*patch_embed).*(?:{}).*".format("|".join(target_modules))
        else:
            return target_modules
