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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)

try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

        if "deepseek_v4" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("deepseek_v4", DeepseekV4Config)
    except Exception:
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        if "deepseek_v4" not in CONFIG_MAPPING:
            CONFIG_MAPPING.register("deepseek_v4", DeepseekV3Config)
except Exception:
    pass


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def patch_youtu_vl_model(model: "PreTrainedModel") -> None:
    original_forward = model.forward

    def forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        if "loss" not in outputs and "labels" in kwargs:
            logits = outputs.get("logits")
            labels = kwargs.get("labels")
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                outputs["loss"] = loss

        return outputs

    model.forward = MethodType(forward, model)


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
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
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

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    if isinstance(architectures, list) and "DeepseekV4ForCausalLM" in architectures:
        qk_rope = getattr(config, "qk_rope_head_dim", None)
        head_dim = getattr(config, "head_dim", None)
        qk_head = getattr(config, "qk_head_dim", None)
        qk_nope = getattr(config, "qk_nope_head_dim", None)
        target_nope = None

        if head_dim is not None and qk_rope is not None:
            target_nope = int(head_dim) - int(qk_rope)
        elif qk_head is not None and qk_rope is not None:
            target_nope = int(qk_head) - int(qk_rope)

        if target_nope is not None and qk_nope is not None and int(qk_nope) != target_nope:
            setattr(config, "qk_nope_head_dim", target_nope)
        elif target_nope is not None and qk_nope is None:
            setattr(config, "qk_nope_head_dim", target_nope)

        if getattr(config, "partial_rotary_factor", None) is None and head_dim and qk_rope:
            setattr(config, "partial_rotary_factor", float(qk_rope) / float(head_dim))

        has_deepseek_v4 = True
        try:
            import transformers.models.deepseek_v4  # noqa: F401
        except Exception:
            has_deepseek_v4 = False

        if not has_deepseek_v4:
            init_kwargs.setdefault("ignore_mismatched_sizes", True)

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
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

    architectures = getattr(model.config, "architectures", None)
    if isinstance(architectures, list) and "DeepseekV4ForCausalLM" in architectures:
        has_deepseek_v4 = True
        try:
            import transformers.models.deepseek_v4  # noqa: F401
        except Exception:
            has_deepseek_v4 = False

        if has_deepseek_v4:
            try:
                import torch.nn.functional as F
                from torch import nn
                from transformers.activations import ACT2FN

                if "sqrtsoftplus" not in ACT2FN:

                    class SqrtSoftplusActivation(nn.Module):
                        def forward(self, input):
                            return F.softplus(input).sqrt()

                    ACT2FN["sqrtsoftplus"] = SqrtSoftplusActivation
            except Exception:
                pass
        else:
            try:
                from transformers.models.deepseek_v3 import modeling_deepseek_v3 as dsv3

                if not hasattr(dsv3, "_llamafactory_dsv4_rope_patch"):
                    orig_interleave = dsv3.apply_rotary_pos_emb_interleave
                    orig_plain = dsv3.apply_rotary_pos_emb

                    def patched_interleave(q, k, cos, sin, position_ids=None):
                        if cos.size(-1) != q.size(-1):
                            cos = cos[..., : q.size(-1)]
                            sin = sin[..., : q.size(-1)]
                        return orig_interleave(q, k, cos, sin, position_ids)

                    def patched_plain(q, k, cos, sin, position_ids=None):
                        if cos.size(-1) != q.size(-1):
                            cos = cos[..., : q.size(-1)]
                            sin = sin[..., : q.size(-1)]
                        return orig_plain(q, k, cos, sin, position_ids)

                    dsv3.apply_rotary_pos_emb_interleave = patched_interleave
                    dsv3.apply_rotary_pos_emb = patched_plain
                    dsv3._llamafactory_dsv4_rope_patch = True
            except Exception:
                pass

            # Try pytorch index first, then safetensors
            index_path = os.path.join(model_args.model_name_or_path, "pytorch_model.bin.index.json")
            if not os.path.isfile(index_path):
                index_path = os.path.join(model_args.model_name_or_path, "model.safetensors.index.json")

            if os.path.isfile(index_path):
                with open(index_path) as f:
                    index = json.load(f)
                weight_map = index.get("weight_map", {})
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    layers = model.model.layers
                    # Group keys by shard to avoid repeated I/O
                    shard_to_keys = {}
                    for layer_id in range(len(layers)):
                        key = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
                        shard_name = weight_map.get(key)
                        if shard_name:
                            shard_to_keys.setdefault(shard_name, []).append((layer_id, key))

                    for shard_name, keys in shard_to_keys.items():
                        shard_path = os.path.join(model_args.model_name_or_path, shard_name)
                        if not os.path.isfile(shard_path):
                            continue

                        if shard_path.endswith(".safetensors"):
                            from safetensors.torch import load_file

                            shard = load_file(shard_path, device="cpu")
                        else:
                            shard = torch.load(shard_path, map_location="cpu", weights_only=True)

                        for layer_id, key in keys:
                            w = shard.get(key)
                            target = getattr(layers[layer_id].self_attn, "kv_b_proj", None)
                            if w is None or target is None or not hasattr(target, "weight"):
                                continue
                            tw = target.weight
                            if w.ndim != 2 or tw.ndim != 2:
                                continue
                            sw0, sw1 = min(w.size(0), tw.size(0)), min(w.size(1), tw.size(1))
                            with torch.no_grad():
                                tw[:sw0, :sw1].copy_(w[:sw0, :sw1].to(device=tw.device, dtype=tw.dtype))

                        del shard

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        if getattr(model.config, "model_type", None) == "youtu_vl":
            patch_youtu_vl_model(model)

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
