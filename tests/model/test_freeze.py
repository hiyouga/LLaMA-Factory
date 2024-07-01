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

from llamafactory.hparams import get_infer_args, get_train_args
from llamafactory.model import load_model, load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "freeze",
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
    "finetuning_type": "freeze",
    "template": "llama3",
    "infer_dtype": "float16",
}


def test_freeze_train_all_modules():
    model_args, _, _, finetuning_args, _ = get_train_args({"freeze_trainable_layers": 1, **TRAIN_ARGS})
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    for name, param in model.named_parameters():
        if name.startswith("model.layers.1."):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16


def test_freeze_train_extra_modules():
    model_args, _, _, finetuning_args, _ = get_train_args(
        {"freeze_trainable_layers": 1, "freeze_extra_modules": "embed_tokens,lm_head", **TRAIN_ARGS}
    )
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)

    for name, param in model.named_parameters():
        if name.startswith("model.layers.1.") or any(module in name for module in ["embed_tokens", "lm_head"]):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16


def test_freeze_inference():
    model_args, _, finetuning_args, _ = get_infer_args(INFER_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=False)

    for param in model.parameters():
        assert param.requires_grad is False
        assert param.dtype == torch.float16
