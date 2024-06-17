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
from typing import Dict, Sequence

import pytest
import torch
from peft import LoraModel, PeftModel
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.extras.misc import get_current_device
from llamafactory.hparams import get_infer_args, get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_ADAPTER = os.environ.get("TINY_LLAMA_ADAPTER", "llamafactory/tiny-random-Llama-3-lora")

TINY_LLAMA_VALUEHEAD = os.environ.get("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
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
    "model_name_or_path": TINY_LLAMA,
    "adapter_name_or_path": TINY_LLAMA_ADAPTER,
    "finetuning_type": "lora",
    "template": "llama3",
    "infer_dtype": "float16",
}


def load_reference_model(is_trainable: bool = False) -> "LoraModel":
    model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, torch_dtype=torch.float16, device_map=get_current_device()
    )
    lora_model = PeftModel.from_pretrained(model, TINY_LLAMA_ADAPTER, is_trainable=is_trainable)
    for param in filter(lambda p: p.requires_grad, lora_model.parameters()):
        param.data = param.data.to(torch.float32)

    return lora_model


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module", diff_keys: Sequence[str] = []):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        if any(key in name for key in diff_keys):
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is False
        else:
            assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5) is True


@pytest.fixture
def fix_valuehead_cpu_loading():
    def post_init(self: "AutoModelForCausalLMWithValueHead", state_dict: Dict[str, "torch.Tensor"]):
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys() if k.startswith("v_head.")}
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

    AutoModelForCausalLMWithValueHead.post_init = post_init


def test_lora_train_qv_modules():
    model_args, _, _, finetuning_args, _ = get_train_args({"lora_target": "q_proj,v_proj", **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    linear_modules = set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    assert linear_modules == {"q_proj", "v_proj"}


def test_lora_train_all_modules():
    model_args, _, _, finetuning_args, _ = get_train_args({"lora_target": "all", **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    linear_modules = set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            linear_modules.add(name.split(".lora_", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    assert linear_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}


def test_lora_train_extra_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {"lora_target": "all", "additional_target": "embed_tokens,lm_head", **TRAIN_ARGS}
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    extra_modules = set()
    for name, param in model.named_parameters():
        if any(module in name for module in ["lora_A", "lora_B"]):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        elif "modules_to_save" in name:
            extra_modules.add(name.split(".modules_to_save", maxsplit=1)[0].split(".")[-1])
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16

    assert extra_modules == {"embed_tokens", "lm_head"}


def test_lora_train_old_adapters():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {"adapter_name_or_path": TINY_LLAMA_ADAPTER, "create_new_adapter": False, **TRAIN_ARGS}
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    ref_model = load_reference_model(is_trainable=True)
    compare_model(model, ref_model)


def test_lora_train_new_adapters():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {"adapter_name_or_path": TINY_LLAMA_ADAPTER, "create_new_adapter": True, **TRAIN_ARGS}
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    ref_model = load_reference_model(is_trainable=True)
    compare_model(
        model, ref_model, diff_keys=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    )


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_lora_train_valuehead():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(
        tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True, add_valuehead=True
    )

    ref_model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(
        TINY_LLAMA_VALUEHEAD, torch_dtype=torch.float16, device_map=get_current_device()
    )
    state_dict = model.state_dict()
    ref_state_dict = ref_model.state_dict()

    assert torch.allclose(state_dict["v_head.summary.weight"], ref_state_dict["v_head.summary.weight"])
    assert torch.allclose(state_dict["v_head.summary.bias"], ref_state_dict["v_head.summary.bias"])


def test_lora_inference():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False)

    ref_model = load_reference_model().merge_and_unload()
    compare_model(model, ref_model)
