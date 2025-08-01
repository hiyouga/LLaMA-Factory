# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import transformers
import transformers.models
from transformers.activations import ACT2FN

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than


if TYPE_CHECKING:
    from transformers import LlavaConfig, PretrainedConfig, PreTrainedModel

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)
transformers_logger = transformers.utils.logging.get_logger(__name__)


@dataclass
class CompositeModel:
    model_type: str
    projector_key: str
    vision_model_keys: list[str]
    language_model_keys: list[str]
    lora_conflict_keys: list[str]

    def get_projector(self, module: "torch.nn.Module") -> "torch.nn.Module":
        for key in self.projector_key.split("."):
            module = getattr(module, key)

        return module


COMPOSITE_MODELS: dict[str, "CompositeModel"] = {}


def _register_composite_model(
    model_type: str,
    projector_key: Optional[str] = None,
    vision_model_keys: Optional[list[str]] = None,
    language_model_keys: Optional[list[str]] = None,
    lora_conflict_keys: Optional[list[str]] = None,
):
    r"""Register a new composite model.

    Args:
        model_type: model type
        projector_key: multi_modal_projector
        vision_model_keys: vision_tower
        language_model_keys: language_model
        lora_conflict_keys: None

    """
    COMPOSITE_MODELS[model_type] = CompositeModel(
        model_type=model_type,
        projector_key=projector_key or "multi_modal_projector",
        vision_model_keys=vision_model_keys or ["vision_tower"],
        language_model_keys=language_model_keys or ["language_model", "lm_head"],
        lora_conflict_keys=lora_conflict_keys or [],
    )


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
    r"""Cast projector output to half precision for fine-tuning quantized VLMs."""

    def _mm_projector_forward_post_hook(
        module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
    ) -> "torch.Tensor":
        return output.to(model_args.compute_dtype)

    if getattr(model, "quantization_method", None):
        model_type = getattr(model.config, "model_type", None)
        if model_type in COMPOSITE_MODELS:
            mm_projector = COMPOSITE_MODELS[model_type].get_projector(model)
        else:
            return

        logger.info_rank0(f"Casting multimodal projector outputs in {model_args.compute_dtype}.")
        mm_projector.register_forward_hook(_mm_projector_forward_post_hook)


def configure_visual_model(config: "PretrainedConfig") -> None:
    r"""Patch VLMs before loading them."""
    if getattr(config, "text_config", None) and not getattr(config, "hidden_size", None):
        # required for ds zero3 and valuehead models
        setattr(config, "hidden_size", getattr(config.text_config, "hidden_size", None))

    if getattr(config, "is_yi_vl_derived_model", None):
        logger.info_rank0("Detected Yi-VL model, applying projector patch.")
        transformers.models.llava.modeling_llava.LlavaMultiModalProjector = LlavaMultiModalProjectorForYiVL


def get_forbidden_modules(config: "PretrainedConfig", finetuning_args: "FinetuningArguments") -> set[str]:
    r"""Freeze vision tower and language model for VLM full/freeze tuning."""
    model_type = getattr(config, "model_type", None)
    forbidden_modules = set()
    if model_type in COMPOSITE_MODELS:
        if finetuning_args.freeze_vision_tower:
            vision_model_keys = COMPOSITE_MODELS[model_type].vision_model_keys
            logger.info_rank0(f"Set vision model not trainable: {vision_model_keys}.")
            forbidden_modules.update(vision_model_keys)

        if finetuning_args.freeze_multi_modal_projector:
            projector_key = COMPOSITE_MODELS[model_type].projector_key
            logger.info_rank0(f"Set multi model projector not trainable: {projector_key}.")
            forbidden_modules.add(projector_key)

        if finetuning_args.freeze_language_model:
            language_model_keys = COMPOSITE_MODELS[model_type].language_model_keys
            logger.info_rank0(f"Set language model not trainable: {language_model_keys}.")
            forbidden_modules.update(language_model_keys)

    return forbidden_modules


def patch_target_modules(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", target_modules: list[str]
) -> list[str]:
    r"""Freeze vision tower for VLM LoRA tuning."""
    model_type = getattr(model.config, "model_type", None)
    if model_type in COMPOSITE_MODELS:
        forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
        forbidden_modules.update(COMPOSITE_MODELS[model_type].lora_conflict_keys)
        module_names = []
        for name, _ in model.named_modules():
            if any(target_module in name for target_module in target_modules) and not any(
                forbidden_module in name for forbidden_module in forbidden_modules
            ):
                module_names.append(name)

        return module_names
    else:
        return target_modules


_register_composite_model(
    model_type="gemma3",
)


_register_composite_model(
    model_type="gemma3n",
    vision_model_keys=["vision_tower", "audio_tower"],
    lora_conflict_keys=["timm_model", "subsample_conv_projection"],
)


# copied from qwen2vl
_register_composite_model(
    model_type="glm4v",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="internvl",
)


_register_composite_model(
    model_type="Keye",
    projector_key="mlp_AR",
    vision_model_keys=["visual.vision_model.patch_embedding", "visual.vision_model.encoder"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embedding"],
)


_register_composite_model(
    model_type="llama4",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="llava",
)


_register_composite_model(
    model_type="llava_next",
)


_register_composite_model(
    model_type="llava_next_video",
)


_register_composite_model(
    model_type="minicpmv",
    projector_key="resampler",
    vision_model_keys=["vpm"],
    language_model_keys=["llm"],
)


_register_composite_model(
    model_type="minicpmo",
    projector_key="resampler",
    vision_model_keys=["vpm", "apm", "audio_avg_pooler", "audio_projection_layer", "tts"],
    language_model_keys=["llm"],
    lora_conflict_keys=["audio_projection_layer"],
)

_register_composite_model(
    model_type="mistral3",
)


_register_composite_model(
    model_type="mllama",
    vision_model_keys=["vision_model"],
)


_register_composite_model(
    model_type="paligemma",
)


_register_composite_model(
    model_type="qwen2_audio",
    vision_model_keys=["audio_tower"],
)


_register_composite_model(
    model_type="qwen2_5_omni_thinker",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks", "audio_tower"],
    language_model_keys=["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="qwen2_5_vl",
    projector_key="visual.merger",
    vision_model_keys=["visual.patch_embed", "visual.blocks"],
    language_model_keys=["language_model", "lm_head"]
    if is_transformers_version_greater_than("4.52.0")
    else ["model", "lm_head"],
    lora_conflict_keys=["patch_embed"],
)


_register_composite_model(
    model_type="video_llava",
)
