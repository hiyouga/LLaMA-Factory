# Copyright 2025 the LlamaFactory team.
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

from llamafactory.extras.misc import get_current_device
from llamafactory.train.test_utils import load_train_model


TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("disable_gradient_checkpointing", [False, True])
def test_vanilla_checkpointing(disable_gradient_checkpointing: bool):
    model = load_train_model(disable_gradient_checkpointing=disable_gradient_checkpointing, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") != disable_gradient_checkpointing


def test_unsloth_gradient_checkpointing():
    model = load_train_model(use_unsloth_gc=True, **TRAIN_ARGS)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert module._gradient_checkpointing_func.__self__.__name__ == "UnslothGradientCheckpointing"


def test_upcast_layernorm():
    model = load_train_model(upcast_layernorm=True, **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            assert param.dtype == torch.float32


def test_upcast_lmhead_output():
    model = load_train_model(upcast_lmhead_output=True, **TRAIN_ARGS)
    inputs = torch.randn((1, 16), dtype=torch.float16, device=get_current_device())
    outputs: "torch.Tensor" = model.get_output_embeddings()(inputs)
    assert outputs.dtype == torch.float32
