# coding=utf-8
# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by HuggingFace's PEFT library.
# https://github.com/huggingface/peft/blob/v0.10.0/examples/loftq_finetuning/quantize_save_load.py
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

import os
from typing import TYPE_CHECKING, Optional

import fire
import torch
import torch.nn as nn
from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class Shell(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model: nn.Module, pattern=".base_layer") -> None:
    for name in {k.split(pattern)[0] for k, _ in model.named_modules() if pattern in k}:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name)
        child_module = getattr(parent_module, child_name)
        base_layer = getattr(child_module, "base_layer")
        weight = getattr(base_layer, "weight", None)
        bias = getattr(base_layer, "bias", None)
        setattr(parent_module, child_name, Shell(weight, bias))

    print("Model unwrapped.")


def quantize_loftq(
    model_name_or_path: str,
    save_dir: str,
    loftq_bits: Optional[int] = 4,
    loftq_iter: Optional[int] = 1,
    lora_alpha: Optional[int] = None,
    lora_rank: Optional[int] = 16,
    lora_target: Optional[str] = "q_proj,v_proj",
    save_safetensors: Optional[bool] = False,
):
    r"""
    Initializes LoRA weights with LoRA-fine-tuning-aware Quantization (LoftQ)
    Usage: python loftq_init.py --model_name_or_path path_to_model --save_dir output_dir
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="auto")
    loftq_config = LoftQConfig(loftq_bits=loftq_bits, loftq_iter=loftq_iter)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=lora_rank,
        lora_alpha=lora_alpha if lora_alpha is not None else lora_rank * 2,
        lora_dropout=0.1,
        target_modules=[name.strip() for name in lora_target.split(",")],
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )

    # Init LoftQ model
    lora_model = get_peft_model(model, lora_config)
    base_model: "PreTrainedModel" = lora_model.get_base_model()

    # Save LoftQ model
    setattr(lora_model.base_model.peft_config["default"], "base_model_name_or_path", save_dir)
    setattr(lora_model.base_model.peft_config["default"], "init_lora_weights", True)
    lora_model.save_pretrained(os.path.join(save_dir, "adapters"), safe_serialization=save_safetensors)

    # Save base model
    unwrap_model(base_model)
    base_model.save_pretrained(save_dir, safe_serialization=save_safetensors)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    fire.Fire(quantize_loftq)
