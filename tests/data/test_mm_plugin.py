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
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

import pytest
import torch
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.hparams import ModelArguments
from llamafactory.model import load_tokenizer


if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


HF_TOKEN = os.environ.get("HF_TOKEN", None)

TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

MESSAGES = [
    {"role": "user", "content": "<image>What is in this image?"},
    {"role": "assistant", "content": "A cat."},
]

IMAGES = [Image.new("RGB", (32, 32), (255, 255, 255))]

INPUT_IDS = [0, 1, 2, 3, 4]

LABELS = [0, 1, 2, 3, 4]

FEATURE_SEQLENS = {"token_type_ids": 1024}


def _get_mm_inputs(processor: "ProcessorMixin") -> Dict[str, "torch.Tensor"]:
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    return image_processor(images=IMAGES, return_tensors="pt")


def _is_close(batch_a: Dict[str, Any], batch_b: Dict[str, Any]):
    assert batch_a.keys() == batch_b.keys()
    for key in batch_a.keys():
        if isinstance(batch_a[key], list):
            assert len(batch_a[key]) == len(batch_b[key])
            for i in range(len(batch_a[key])):
                if isinstance(batch_a[key][i], torch.Tensor):
                    assert torch.allclose(batch_a[key][i], batch_b[key][i], rtol=1e-4, atol=1e-5)
                else:
                    assert batch_a[key][i] == batch_b[key][i]
        elif isinstance(batch_a[key], torch.Tensor):
            assert torch.allclose(batch_a[key], batch_b[key], rtol=1e-4, atol=1e-5)
        else:
            raise NotImplementedError


def test_base_plugin():
    model_args = ModelArguments(model_name_or_path=TINY_LLAMA)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    base_plugin = get_mm_plugin(name="base", image_token="<image>")
    model_inputs = defaultdict(list)
    base_plugin.process_model_inputs(model_inputs, IMAGES, FEATURE_SEQLENS, processor)

    assert base_plugin.process_messages(MESSAGES, IMAGES, processor)
    assert base_plugin.process_token_ids(INPUT_IDS, LABELS, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(base_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), {})
    _is_close(model_inputs, {})


def test_llava_plugin():
    model_args = ModelArguments(model_name_or_path="llava-hf/llava-1.5-7b-hf")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    mm_inputs = _get_mm_inputs(processor)
    expected_model_inputs = {key: [value[0]] for key, value in mm_inputs.items()}

    llava_plugin = get_mm_plugin(name="llava", image_token="<image>")
    model_inputs = defaultdict(list)
    llava_plugin.process_model_inputs(model_inputs, IMAGES, FEATURE_SEQLENS, processor)

    assert llava_plugin.process_messages(MESSAGES, IMAGES, processor)
    assert llava_plugin.process_token_ids(INPUT_IDS, LABELS, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(llava_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    _is_close(model_inputs, expected_model_inputs)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_paligemma_plugin():
    model_args = ModelArguments(model_name_or_path="google/paligemma-3b-pt-224")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    image_seq_length: int = getattr(image_processor, "image_seq_length")

    mm_inputs = _get_mm_inputs(processor)
    mm_inputs["token_type_ids"] = [[0] * image_seq_length + [1] * (1024 - image_seq_length)]
    expected_model_inputs = {key: [value[0]] for key, value in mm_inputs.items()}
    expected_input_ids = [tokenizer.convert_tokens_to_ids("<image>")] * image_seq_length + INPUT_IDS
    expected_labels = [-100] * image_seq_length + LABELS

    paligemma_plugin = get_mm_plugin(name="paligemma", image_token="<image>")
    model_inputs = defaultdict(list)
    paligemma_plugin.process_model_inputs(model_inputs, IMAGES, FEATURE_SEQLENS, processor)

    assert paligemma_plugin.process_messages(MESSAGES, IMAGES, processor)
    assert paligemma_plugin.process_token_ids(INPUT_IDS, LABELS, tokenizer, processor) == (
        expected_input_ids,
        expected_labels,
    )
    _is_close(paligemma_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    _is_close(model_inputs, expected_model_inputs)


def test_qwen2_vl_plugin():
    model_args = ModelArguments(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]

    mm_inputs = _get_mm_inputs(processor)
    expected_model_inputs = {key: [value] for key, value in mm_inputs.items()}

    llava_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>")
    model_inputs = defaultdict(list)
    llava_plugin.process_model_inputs(model_inputs, IMAGES, FEATURE_SEQLENS, processor)

    assert llava_plugin.process_messages(MESSAGES, IMAGES, processor)
    assert llava_plugin.process_token_ids(INPUT_IDS, LABELS, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(llava_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    _is_close(model_inputs, expected_model_inputs)
