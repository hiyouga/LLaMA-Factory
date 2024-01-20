# coding=utf-8
# Converts the Baichuan2-7B model in the same format as LLaMA2-7B.
# Usage: python llamafy_baichuan2.py --input_dir input --output_dir output --shard_size 10GB
# Inspired by: https://huggingface.co/fireballoon/baichuan-llama-7b/blob/main/convert_baichuan_to_llama.py
# Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied

import json
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import fire
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)


CONFIG_NAME = "config.json"


def save_weight(input_dir: str, output_dir: str, shard_size: str, save_safetensors: bool):
    baichuan2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(input_dir), desc="Load weights"):
        if os.path.isfile(os.path.join(input_dir, filepath)) and filepath.endswith(".bin"):
            shard_weight = torch.load(os.path.join(input_dir, filepath), map_location="cpu")
            baichuan2_state_dict.update(shard_weight)

    llama2_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for key, value in tqdm(baichuan2_state_dict.items(), desc="Convert format"):
        if "W_pack" in key:
            proj_size = value.size(0) // 3
            llama2_state_dict[key.replace("W_pack", "q_proj")] = value[:proj_size, :]
            llama2_state_dict[key.replace("W_pack", "k_proj")] = value[proj_size : 2 * proj_size, :]
            llama2_state_dict[key.replace("W_pack", "v_proj")] = value[2 * proj_size :, :]
        elif "lm_head" in key:
            llama2_state_dict[key] = torch.nn.functional.normalize(value)
        else:
            llama2_state_dict[key] = value

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    shards, index = shard_checkpoint(llama2_state_dict, max_shard_size=shard_size, weights_name=weights_name)

    for shard_file, shard in tqdm(shards.items(), desc="Save weights"):
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))

    if index is None:
        print("Model weights saved in {}".format(os.path.join(output_dir, WEIGHTS_NAME)))
    else:
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(output_dir))


def save_config(input_dir: str, output_dir: str):
    with open(os.path.join(input_dir, CONFIG_NAME), "r", encoding="utf-8") as f:
        llama2_config_dict: Dict[str, Any] = json.load(f)

    llama2_config_dict["architectures"] = ["LlamaForCausalLM"]
    llama2_config_dict.pop("auto_map", None)
    llama2_config_dict.pop("tokenizer_class", None)
    llama2_config_dict["model_type"] = "llama"

    with open(os.path.join(output_dir, CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(llama2_config_dict, f, indent=2)
    print("Model config saved in {}".format(os.path.join(output_dir, CONFIG_NAME)))


def llamafy_baichuan2(input_dir: str, output_dir: str, shard_size: str, save_safetensors: Optional[bool] = False):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except Exception as e:
        raise print("Output dir already exists", e)

    save_weight(input_dir, output_dir, shard_size, save_safetensors)
    save_config(input_dir, output_dir)


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)
