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
from typing import Dict

import pytest
import torch
from transformers import AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

from llamafactory.extras.misc import get_current_device
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TINY_LLAMA_VALUEHEAD = os.environ.get("TINY_LLAMA_VALUEHEAD", "llamafactory/tiny-random-Llama-3-valuehead")

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "template": "llama3",
    "infer_dtype": "float16",
}


def compare_model(model_a: "torch.nn.Module", model_b: "torch.nn.Module"):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for name in state_dict_a.keys():
        assert torch.allclose(state_dict_a[name], state_dict_b[name], rtol=1e-4, atol=1e-5)


@pytest.fixture
def fix_valuehead_cpu_loading():
    def post_init(self: "AutoModelForCausalLMWithValueHead", state_dict: Dict[str, "torch.Tensor"]):
        state_dict = {k[7:]: state_dict[k] for k in state_dict.keys() if k.startswith("v_head.")}
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

    AutoModelForCausalLMWithValueHead.post_init = post_init


def test_base():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False)

    ref_model = AutoModelForCausalLM.from_pretrained(
        TINY_LLAMA, torch_dtype=torch.float16, device_map=get_current_device()
    )
    compare_model(model, ref_model)


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_valuehead():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(
        tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False, add_valuehead=True
    )

    ref_model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(
        TINY_LLAMA_VALUEHEAD, torch_dtype=torch.float16, device_map=get_current_device()
    )
    ref_model.v_head = ref_model.v_head.to(torch.float16)
    compare_model(model, ref_model)
