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

from llamafactory.extras.misc import get_current_device
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

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


def test_checkpointing_enable():
    model_args, _, _, finetuning_args, _ = get_train_args({"disable_gradient_checkpointing": False, **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") is True


def test_checkpointing_disable():
    model_args, _, _, finetuning_args, _ = get_train_args({"disable_gradient_checkpointing": True, **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    for module in filter(lambda m: hasattr(m, "gradient_checkpointing"), model.modules()):
        assert getattr(module, "gradient_checkpointing") is False


def test_upcast_layernorm():
    model_args, _, _, finetuning_args, _ = get_train_args({"upcast_layernorm": True, **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            assert param.dtype == torch.float32


def test_upcast_lmhead_output():
    model_args, _, _, finetuning_args, _ = get_train_args({"upcast_lmhead_output": True, **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
    inputs = torch.randn((1, 16), dtype=torch.float16, device=get_current_device())
    outputs: "torch.Tensor" = model.get_output_embeddings()(inputs)
    assert outputs.dtype == torch.float32
