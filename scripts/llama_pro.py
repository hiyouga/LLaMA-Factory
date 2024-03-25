# coding=utf-8
# Performs block expansion for LLaMA, Mistral or Qwen1.5 models.
# Usage: python llama_pro.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir llama2_pro --num_expand 8
# Inspired by: https://github.com/TencentARC/LLaMA-Pro/blob/main/scripts/block_expansion.py

import json
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

import fire
import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


def change_name(name: str, old_index: int, new_index: int) -> str:
    return name.replace(".{:d}.".format(old_index), ".{:d}.".format(new_index))


def block_expansion(
    model_name_or_path: str,
    output_dir: str,
    num_expand: int,
    shard_size: Optional[str] = "2GB",
    save_safetensors: Optional[bool] = False,
):
    config: "PretrainedConfig" = AutoConfig.from_pretrained(model_name_or_path)
    num_layers = getattr(config, "num_hidden_layers")
    setattr(config, "num_hidden_layers", num_layers + num_expand)
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.save_pretrained(output_dir)

    config: "PretrainedConfig" = AutoConfig.from_pretrained(model_name_or_path)  # load the original one
    if save_safetensors:
        setattr(config, "tie_word_embeddings", False)  # safetensors does not allow shared weights

    model: "PreTrainedModel" = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    state_dict = model.state_dict()

    if num_layers % num_expand != 0:
        raise ValueError("`num_layers` {} should be divisible by `num_expand` {}.".format(num_layers, num_expand))

    split = num_layers // num_expand
    layer_cnt = 0
    output_state_dict = OrderedDict()
    for i in range(num_layers):
        for key, value in state_dict.items():
            if ".{:d}.".format(i) in key:
                output_state_dict[change_name(key, i, layer_cnt)] = value

        print("Add layer {} copied from layer {}".format(layer_cnt, i))
        layer_cnt += 1
        if (i + 1) % split == 0:
            for key, value in state_dict.items():
                if ".{:d}.".format(i) in key:
                    if "down_proj" in key or "o_proj" in key:
                        output_state_dict[change_name(key, i, layer_cnt)] = torch.zeros_like(value)
                    else:
                        output_state_dict[change_name(key, i, layer_cnt)] = torch.clone(value)

            print("Add layer {} expanded from layer {}".format(layer_cnt, i))
            layer_cnt += 1

    for key, value in state_dict.items():
        if key not in output_state_dict:
            output_state_dict[key] = value

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=shard_size, weights_name=weights_name)

    for shard_file, shard in tqdm(shards.items(), desc="Save weights"):
        if save_safetensors:
            save_file(shard, os.path.join(output_dir, shard_file), metadata={"format": "pt"})
        else:
            torch.save(shard, os.path.join(output_dir, shard_file))

    if index is None:
        print("Model weights saved in {}".format(os.path.join(output_dir, weights_name)))
    else:
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_name), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        print("Model weights saved in {}".format(output_dir))

    print("Fine-tune this model with:")
    print("  --model_name_or_path {} \\".format(output_dir))
    print("  --finetuning_type freeze \\")
    print("  --name_module_trainable all \\")
    print("  --num_layer_trainable {} \\".format(num_expand))
    print("  --use_llama_pro")


if __name__ == "__main__":
    fire.Fire(block_expansion)
