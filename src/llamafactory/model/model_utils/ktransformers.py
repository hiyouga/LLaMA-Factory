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

from typing import TYPE_CHECKING, Any, Optional
import importlib.util as _u

import torch

from ...extras import logging
from ...extras.misc import get_current_device


if TYPE_CHECKING:
    from ...hparams import FinetuningArguments, ModelArguments

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM

KT_AVAILABLE = _u.find_spec("ktransformers") is not None
if KT_AVAILABLE:
    from ktransformers.optimize.optimize import optimize_and_load_gguf
    from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
    from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
    from ktransformers.models.modeling_llama import LlamaForCausalLM
    from ktransformers.models.modeling_mixtral import MixtralForCausalLM
    from ktransformers.util.utils import load_weights, xpu_fp16_model
    from ktransformers.server.config.config import Config
    from ktransformers.sft.lora import inject_lora_layer
    from ktransformers.util.custom_loader import GGUFLoader, SafeTensorLoader
    from ktransformers.util.globals import GLOBAL_CONFIG

logger = logging.get_logger(__name__)

def _get_kt_kwargs(
    config: "PretrainedConfig",
    model_name_or_path: str,
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
) -> dict[str, Any]:
    return {
        "model_name": model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": finetuning_args.finetuning_type == "full",
        "device_map": {"": get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": "ktransformers",
    }


def load_kt_pretrained_model(
    config: "PretrainedConfig", model_args: "ModelArguments", finetuning_args: "FinetuningArguments"
) -> Optional["PreTrainedModel"]:
    r"""Optionally load pretrained model with KTransformers. Used in training."""
    custom_models = {
        "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
        "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
        "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
        "LlamaForCausalLM": LlamaForCausalLM,
        "MixtralForCausalLM": MixtralForCausalLM,
    }
    Config().cpu_infer = model_args.cpu_infer
    Config().chunk_size = model_args.chunk_size
    if torch.xpu.is_available():
        use_cuda_graph = False
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)

    if model_args.mode == 'long_context':
        assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
        torch.set_default_dtype(torch.float16)
    elif xpu_fp16_model(config):
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if (
                "Qwen2Moe" in config.architectures[0]
            ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            if "Llama" in config.architectures[0]:
                config._attn_implementation = "eager"
            if "Mixtral" in config.architectures[0]:
                config._attn_implementation = "flash_attention_2"
            if torch.xpu.is_available():
                config._attn_implementation = "eager"
            model = custom_models[config.architectures[0]](config)
        else:
            if torch.xpu.is_available():
                attn_implementation = "eager"
            else:
                attn_implementation = "flash_attention_2"
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation=attn_implementation
            )

    optimize_config_path = model_args.kt_optimize_rule
    gguf_path = model_args.model_name_or_path
    
    if optimize_config_path is None:
        optimize_config_path = input(
            "please input the path of your rule file(yaml file containing optimize rules):"
        )

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
        
    GLOBAL_CONFIG._config["mod"] = "infer"
    optimize_and_load_gguf(model, optimize_config_path, gguf_path, config)

    return model


def get_kt_peft_model(
    model: "PreTrainedModel", model_args: "ModelArguments", peft_kwargs: dict[str, Any]
) -> "PreTrainedModel":
    r"""Get the peft model for the pretrained model with KTransformers. Used in training."""
    from ktransformers.sft.peft_utils.mapping import get_peft_model

    return get_peft_model(model, peft_kwargs)


def load_kt_peft_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    model: "PreTrainedModel",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Load peft model with KTransformers. Used in both training and inference."""
    load_adapter_name_or_path = model_args.adapter_name_or_path[0]
    if load_adapter_name_or_path.endswith('.gguf'):
        inject_lora_layer(model, load_adapter_name_or_path)
        adapter_gguf_loader = GGUFLoader(load_adapter_name_or_path)
        load_weights(model, adapter_gguf_loader, adapter_gguf=True)
        model.train()
    else:
        inject_lora_layer(model, load_adapter_name_or_path)
        
        adapter_loader = SafeTensorLoader(load_adapter_name_or_path)
        device = next(model.parameters()).device
        for key in adapter_loader.tensor_file_map.keys():
            try:
                tensor = adapter_loader.load_tensor(key, device=device)
                
                model_key = key.replace("base_model.model.", "")
                model_key = model_key.replace(".weight", ".default.weight")
                model_key = model_key.replace(".default.default.weight", ".default.weight")
                
                param = model.get_parameter(model_key)
                param.data.copy_(tensor.data)
                
                print(f"Loaded adapter weight: {key} -> {model_key}")
            except AttributeError as e:
                print(f"Skipping {key}: not a model parameter")
            except KeyError as e:
                print(f"Key not found in model: {model_key} (original: {key})")

    return model
