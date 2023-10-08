# coding=utf-8
# Converts the Baichuan2-7B model in the same format as LLaMA2-7B.
# Usage: python llamafy_baichuan2.py --input_dir input --output_dir output --shard_size 10GB
# Inspired by: https://huggingface.co/fireballoon/baichuan-llama-7b/blob/main/convert_baichuan_to_llama.py
# Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied

import os
import fire
import json
import torch
from collections import OrderedDict
from transformers.modeling_utils import shard_checkpoint, WEIGHTS_NAME, WEIGHTS_INDEX_NAME
from typing import Any, Dict


CONFIG_NAME = "config.json"


def save_weight(
    input_dir: str,
    output_dir: str,
    shard_size: str
):
    baichuan2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for filepath in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".bin"):
            shard_weight = torch.load(os.path.join(input_dir, filepath), map_location="cpu")
            baichuan2_state_dict.update(shard_weight)

    llama2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for key, value in baichuan2_state_dict.items():
        if "W_pack" in key:
            proj_size = value.size(0) // 3
            llama2_state_dict[key.replace("W_pack", "q_proj")] = value[:proj_size, :]
            llama2_state_dict[key.replace("W_pack", "k_proj")] = value[proj_size:2*proj_size, :]
            llama2_state_dict[key.replace("W_pack", "v_proj")] = value[2*proj_size:, :]
        elif "lm_head" in key:
            llama2_state_dict[key] = torch.nn.functional.normalize(value)
        else:
            llama2_state_dict[key] = value

    shards, index = shard_checkpoint(llama2_state_dict, max_shard_size=shard_size, weights_name=WEIGHTS_NAME)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))
    
    if index is None:
        print("Model weights saved in {}".format(os.path.join(output_dir, WEIGHTS_NAME)))
    else:
        with open(os.path.join(output_dir, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(output_dir))


def save_config(
    input_dir: str,
    output_dir: str
):
    with open(os.path.join(input_dir, CONFIG_NAME), "r", encoding="utf-8") as f:
        llama2_config_dict: Dict[str, Any] = json.load(f)

    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict.pop("auto_map", None)
    llama2_config_dict.pop("tokenizer_class", None)
    llama2_config_dict["model_type"] = "llama"

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)
    print("Model config saved in {}".format(os.path.join(output_dir, CONFIG_NAME)))


def llamafy_baichuan2(
    input_dir: str,
    output_dir: str,
    shard_size: str
):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise print("Output dir already exists", e)

    save_weight(input_dir, output_dir, shard_size)
    save_config(input_dir, output_dir)    


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)
