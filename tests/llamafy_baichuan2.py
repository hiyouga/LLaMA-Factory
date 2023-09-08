# coding=utf-8
# Converts the Baichuan2-7B model in the same format as LLaMA2-7B.
# Usage: python llamafy_baichuan2.py --baichuan2_json baichuan2.index.json --llama2_json llama2.index.json
#                                    --input_dir baichuan2_original --output_dir baichuan2_llamafied
# Inspired by: https://huggingface.co/fireballoon/baichuan-llama-7b/blob/main/convert_baichuan_to_llama.py
# Converted model: https://huggingface.co/hiyouga/Baichuan2-7B-Base-LLaMAfied

import os
import fire
import json
import torch
from collections import OrderedDict


SHARD_A = "pytorch_model-00001-of-00002.bin"
SHARD_B = "pytorch_model-00002-of-00002.bin"


def llamafy_baichuan2(
    baichuan2_json: str,
    llama2_json: str,
    input_dir: str,
    output_dir: str
):
    weight_shard_a = torch.load(os.path.join(input_dir, SHARD_A), map_location="cpu")
    weight_shard_b = torch.load(os.path.join(input_dir, SHARD_B), map_location="cpu")

    baichuan2_state_dict = OrderedDict()
    baichuan2_state_dict.update(weight_shard_a)
    baichuan2_state_dict.update(weight_shard_b)

    llama2_state_dict = OrderedDict()
    for key, value in baichuan2_state_dict.items():
        if "W_pack" in key:
            llama2_state_dict[key.replace("W_pack", "q_proj")] = value[:4096, :]
            llama2_state_dict[key.replace("W_pack", "k_proj")] = value[4096:2*4096, :]
            llama2_state_dict[key.replace("W_pack", "v_proj")] = value[2*4096:, :]
        elif "lm_head" in key:
            llama2_state_dict[key] = torch.nn.functional.normalize(value)
        else:
            llama2_state_dict[key] = value

    with open(os.path.join(input_dir, baichuan2_json), "r", encoding="utf-8") as f:
        baichuan2_index = json.load(f)
    with open(os.path.join(input_dir, llama2_json), "r", encoding="utf-8") as f:
        llama2_index = json.load(f)

    merged_index = OrderedDict()
    merged_index["metadata"] = baichuan2_index["metadata"]
    merged_index["weight_map"] = llama2_index["weight_map"]

    state_dict_a, state_dict_b = OrderedDict(), OrderedDict()
    for key, value in llama2_state_dict.items():
        if merged_index["weight_map"][key] == SHARD_A:
            state_dict_a[key] = value
        else:
            state_dict_b[key] = value

    os.makedirs(output_dir, exist_ok=True)
    torch.save(state_dict_a, os.path.join(output_dir, SHARD_A))
    torch.save(state_dict_b, os.path.join(output_dir, SHARD_B))
    with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w", encoding="utf-8") as f:
        json.dump(merged_index, f)
    print("Completed!")


if __name__ == "__main__":
    fire.Fire(llamafy_baichuan2)
