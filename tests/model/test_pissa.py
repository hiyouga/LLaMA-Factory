# Copyright 2024 the LlamaFactory team.
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

import torch
from peft import LoraModel, PeftModel
from transformers import AutoModelForCausalLM

from llamafactory.extras.misc import get_current_device
from llamafactory.hparams import get_infer_args, get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_PISSA = os.environ.get("TINY_LLAMA_ADAPTER", "llamafactory/tiny-random-Llama-3-pissa")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "pissa_init": True,
    "pissa_iter": -1,
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA_PISSA,
    "adapter_name_or_path": TINY_LLAMA_PISSA,
    "adapter_folder": "pissa_init",
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
}


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module"):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5)


def test_pissa_init():
    model_args, _, _, finetuning_args, _ = get_train_args(TRAIN_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA_PISSA, torch_dtype=torch.float16, device_map=get_current_device()
    )
    ref_model = PeftModel.from_pretrained(base_model, TINY_LLAMA_PISSA, subfolder="pissa_init", is_trainable=True)
    for param in filter(lambda p: p.requires_grad, ref_model.parameters()):
        param.data = param.data.to(torch.float32)

    compare_model(model, ref_model)


def test_pissa_inference():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False)

    base_model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA_PISSA, torch_dtype=torch.float16, device_map=get_current_device()
    )
    ref_model: "LoraModel" = PeftModel.from_pretrained(base_model, TINY_LLAMA_PISSA, subfolder="pissa_init")
    ref_model = ref_model.merge_and_unload()
    compare_model(model, ref_model)
