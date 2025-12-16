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

from llamafactory.train.test_utils import load_infer_model, load_train_model


TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "freeze",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}

INFER_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "finetuning_type": "freeze",
    "template": "llama3",
    "infer_dtype": "float16",
}


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_freeze_train_all_modules():
    model = load_train_model(freeze_trainable_layers=1, **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if name.startswith("model.layers.1."):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_freeze_train_extra_modules():
    model = load_train_model(freeze_trainable_layers=1, freeze_extra_modules="embed_tokens,lm_head", **TRAIN_ARGS)
    for name, param in model.named_parameters():
        if name.startswith("model.layers.1.") or any(module in name for module in ["embed_tokens", "lm_head"]):
            assert param.requires_grad is True
            assert param.dtype == torch.float32
        else:
            assert param.requires_grad is False
            assert param.dtype == torch.float16


@pytest.mark.runs_on(["cpu", "npu", "cuda"])
def test_freeze_inference():
    model = load_infer_model(**INFER_ARGS)
    for param in model.parameters():
        assert param.requires_grad is False
        assert param.dtype == torch.float16
