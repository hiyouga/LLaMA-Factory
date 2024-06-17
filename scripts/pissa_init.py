# coding=utf-8
# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is based on the HuggingFace's PEFT library.
# https://github.com/huggingface/peft/blob/v0.11.0/examples/pissa_finetuning/preprocess.py
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
from typing import TYPE_CHECKING

import fire
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedModel


def quantize_pissa(
    model_name_or_path: str,
    output_dir: str,
    pissa_iter: int = 4,
    lora_alpha: int = None,
    lora_rank: int = 16,
    lora_dropout: float = 0,
    lora_target: str = "q_proj,v_proj",
    save_safetensors: bool = True,
):
    r"""
    Initializes LoRA weights with Principal Singular values and Singular vectors Adaptation (PiSSA)
    Usage: python pissa_init.py --model_name_or_path path_to_model --output_dir output_dir
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="auto")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha if lora_alpha is not None else lora_rank * 2,
        lora_dropout=lora_dropout,
        target_modules=[name.strip() for name in lora_target.split(",")],
        init_lora_weights="pissa" if pissa_iter == -1 else "pissa_niter_{}".format(pissa_iter),
    )

    # Init PiSSA model
    peft_model = get_peft_model(model, lora_config)
    pissa_dir = os.path.join(output_dir, "pissa_init")

    # Save PiSSA model
    setattr(peft_model.peft_config["default"], "init_lora_weights", True)  # don't apply pissa again
    peft_model.save_pretrained(pissa_dir, safe_serialization=save_safetensors)
    print("Adapter weights saved in {}".format(pissa_dir))

    # Save base model
    base_model: "PreTrainedModel" = peft_model.unload()
    base_model.save_pretrained(output_dir, safe_serialization=save_safetensors)
    tokenizer.save_pretrained(output_dir)
    print("Model weights saved in {}".format(output_dir))

    print("- Fine-tune this model with:")
    print("model_name_or_path: {}".format(output_dir))
    print("adapter_name_or_path: {}".format(pissa_dir))
    print("finetuning_type: lora")
    print("pissa_init: false")
    print("pissa_convert: true")
    print("- and optionally with:")
    print("quantization_bit: 4")


if __name__ == "__main__":
    fire.Fire(quantize_pissa)
