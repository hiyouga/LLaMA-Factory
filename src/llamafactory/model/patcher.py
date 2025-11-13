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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextMLP(
                    config, intermediate_size=config.moe_intermediate_size
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # 计算所有专家的路由权重
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 保留top_k的权重，其余专家权重置为0（而非仅保留top_k专家）
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        # 初始化全零权重矩阵（形状与所有专家相同）
        full_routing_weights = torch.zeros_like(routing_weights)
        # 仅保留top_k专家的权重，其余专家权重保持为0
        full_routing_weights.scatter_(1, top_k_indices, top_k_weights)

        # 归一化top_k权重（保持与原逻辑一致）
        if self.norm_topk_prob:
            # 计算每行top_k权重的和（用于归一化）
            top_k_sum = full_routing_weights.sum(dim=-1, keepdim=True)
            # 避免除零（虽然softmax后和不为零，但归一化可能导致极端情况）
            top_k_sum = torch.clamp(top_k_sum, min=1e-9)
            full_routing_weights /= top_k_sum

        # 转换回输入数据类型
        full_routing_weights = full_routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 遍历所有专家（而非仅被选中的专家）
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # 获取当前专家的权重（未激活的专家此处权重为0）
            expert_weights = full_routing_weights[:, expert_idx, None]  # 形状: (batch*seq, 1)
            # 所有样本都参与当前专家的计算，但权重可能为0
            current_hidden_states = expert_layer(hidden_states) * expert_weights
            # 累加所有专家的输出（权重为0的专家不影响结果）
            final_hidden_states += current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    if "InternVLChatModel" in getattr(config, "architectures", []):
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in getattr(config, "architectures", []):
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "qwen3_omni_moe_thinker":
        # breakpoint()
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
