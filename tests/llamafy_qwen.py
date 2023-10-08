# coding=utf-8
# Converts the Qwen models in the same format as LLaMA2.
# Usage: python llamafy_qwen.py --input_dir input --output_dir output --shard_size 10GB

import os
import fire
import json
import torch
from collections import OrderedDict
from safetensors import safe_open
from transformers.modeling_utils import shard_checkpoint, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from transformers.utils import check_min_version
from typing import Any, Dict

try:
    check_min_version("4.34.0")
except:
    raise ValueError("Please upgrade `transformers` to 4.34.0")


CONFIG_NAME = "config.json"


def save_weight(
    input_dir: str,
    output_dir: str,
    shard_size: str
) -> str:
    qwen_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for filepath in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".safetensors"):
            with safe_open(os.path.join(input_dir, filepath), framework="pt", device="cpu") as f:
                for key in f.keys():
                    qwen_state_dict[key] = f.get_tensor(key)

    llama2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    torch_dtype = None
    for key, value in qwen_state_dict.items():
        if torch_dtype is None:
            torch_dtype = value.dtype
        if "wte" in key:
            llama2_state_dict["model.embed_tokens.weight"] = value
        elif "ln_f" in key:
            llama2_state_dict["model.norm.weight"] = value
        else:
            key = key.replace("transformer.h", "model.layers")
            if "attn.c_attn" in key:
                proj_size = value.size(0) // 3
                llama2_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
                llama2_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value[proj_size:2*proj_size, ...]
                llama2_state_dict[key.replace("attn.c_attn", "self_attn.v_proj")] = value[2*proj_size:, ...]
            elif "attn.c_proj" in key:
                llama2_state_dict[key.replace("attn.c_proj", "self_attn.o_proj")] = value
                llama2_state_dict[key.replace("attn.c_proj.weight", "self_attn.o_proj.bias")] = (
                    torch.zeros_like(value[:, 0]).squeeze()
                )
            elif "ln_1" in key:
                llama2_state_dict[key.replace("ln_1", "input_layernorm")] = value
            elif "ln_2" in key:
                llama2_state_dict[key.replace("ln_2", "post_attention_layernorm")] = value
            elif "mlp.w1" in key:
                llama2_state_dict[key.replace("mlp.w1", "mlp.up_proj")] = value
            elif "mlp.w2" in key:
                llama2_state_dict[key.replace("mlp.w2", "mlp.gate_proj")] = value
            elif "mlp.c_proj" in key:
                llama2_state_dict[key.replace("mlp.c_proj", "mlp.down_proj")] = value
            elif "lm_head" in key:
                llama2_state_dict[key] = value
            else:
                raise KeyError("Unable to process key {}".format(key))

    shards, index = shard_checkpoint(llama2_state_dict, max_shard_size=shard_size, weights_name=WEIGHTS_NAME)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))

    if index is None:
        print("Model weights saved in {}".format(os.path.join(output_dir, WEIGHTS_NAME)))
    else:
        with open(os.path.join(output_dir, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(output_dir))

    return str(torch_dtype).replace("torch.", "")


def save_config(
    input_dir: str,
    output_dir: str,
    torch_dtype: str
):
    with open(os.path.join(input_dir, CONFIG_NAME), "r", encoding="utf-8") as f:
        qwen_config_dict: Dict[str, Any] = json.load(f)

    llama2_config_dict: Dict[str, Any] = OrderedDict()
    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict["hidden_act"] = "silu"
    llama2_config_dict["hidden_size"] = qwen_config_dict["hidden_size"]
    llama2_config_dict["initializer_range"] = qwen_config_dict["initializer_range"]
    llama2_config_dict["intermediate_size"] = qwen_config_dict["intermediate_size"] // 2
    llama2_config_dict["max_position_embeddings"] = qwen_config_dict["max_position_embeddings"]
    llama2_config_dict["model_type"] = "llama"
    llama2_config_dict["num_attention_heads"] = qwen_config_dict["num_attention_heads"]
    llama2_config_dict["num_hidden_layers"] = qwen_config_dict["num_hidden_layers"]
    llama2_config_dict["num_key_value_heads"] = qwen_config_dict["hidden_size"] // qwen_config_dict["kv_channels"]
    llama2_config_dict["pretraining_tp"] = 1
    llama2_config_dict["rms_norm_eps"] = qwen_config_dict["layer_norm_epsilon"]
    llama2_config_dict["rope_scaling"] = None
    llama2_config_dict["tie_word_embeddings"] = qwen_config_dict["tie_word_embeddings"]
    llama2_config_dict["torch_dtype"] = torch_dtype
    llama2_config_dict["transformers_version"] = "4.34.0"
    llama2_config_dict["use_cache"] = True
    llama2_config_dict["vocab_size"] = qwen_config_dict["vocab_size"]
    llama2_config_dict["attention_bias"] = True

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)
    print("Model config saved in {}".format(os.path.join(output_dir, CONFIG_NAME)))


def llamafy_qwen(
    input_dir: str,
    output_dir: str,
    shard_size: str
):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise print("Output dir already exists", e)

    torch_dtype = save_weight(input_dir, output_dir, shard_size)
    save_config(input_dir, output_dir, torch_dtype)    


if __name__ == "__main__":
    fire.Fire(llamafy_qwen)
