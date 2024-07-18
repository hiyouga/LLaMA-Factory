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

import pytest
import torch

from llamafactory.train.test_utils import (
    check_lora_model,
    compare_model,
    load_infer_model,
    load_reference_model,
    load_train_model,
    patch_valuehead_model,
)


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


@pytest.fixture
def fix_valuehead_cpu_loading():
    patch_valuehead_model()


def test_lora_train_qv_modules():
    model = load_train_model(lora_target="q_proj,v_proj", **TRAIN_ARGS)
    linear_modules, _ = check_lora_model(model)
    assert linear_modules == {"q_proj", "v_proj"}


def test_lora_train_all_modules():
    model = load_train_model(lora_target="all", **TRAIN_ARGS)
    linear_modules, _ = check_lora_model(model)
    assert linear_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}


def test_lora_train_extra_modules():
    model = load_train_model(additional_target="embed_tokens,lm_head", **TRAIN_ARGS)
    _, extra_modules = check_lora_model(model)
    assert extra_modules == {"embed_tokens", "lm_head"}


def test_lora_train_old_adapters():
    model = load_train_model(adapter_name_or_path=TINY_LLAMA_ADAPTER, create_new_adapter=False, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA, TINY_LLAMA_ADAPTER, use_lora=True, is_trainable=True)
    compare_model(model, ref_model)


def test_lora_train_new_adapters():
    model = load_train_model(adapter_name_or_path=TINY_LLAMA_ADAPTER, create_new_adapter=True, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA, TINY_LLAMA_ADAPTER, use_lora=True, is_trainable=True)
    compare_model(
        model, ref_model, diff_keys=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    )


@pytest.mark.usefixtures("fix_valuehead_cpu_loading")
def test_lora_train_valuehead():
    model = load_train_model(add_valuehead=True, **TRAIN_ARGS)
    ref_model = load_reference_model(TINY_LLAMA_VALUEHEAD, is_trainable=True, add_valuehead=True)
    state_dict = model.state_dict()
    ref_state_dict = ref_model.state_dict()
    assert torch.allclose(state_dict["v_head.summary.weight"], ref_state_dict["v_head.summary.weight"])
    assert torch.allclose(state_dict["v_head.summary.bias"], ref_state_dict["v_head.summary.bias"])


def test_lora_inference():
    model = load_infer_model(**INFER_ARGS)
    ref_model = load_reference_model(TINY_LLAMA, TINY_LLAMA_ADAPTER, use_lora=True).merge_and_unload()
    compare_model(model, ref_model)
