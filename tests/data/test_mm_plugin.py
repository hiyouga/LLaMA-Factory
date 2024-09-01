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
from typing import TYPE_CHECKING, Any, Dict, Tuple

import pytest
import torch
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.hparams import ModelArguments
from llamafactory.model import load_tokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor


HF_TOKEN = os.environ.get("HF_TOKEN", None)

TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

MM_MESSAGES = [
    {"role": "user", "content": "<image>What is in this image?"},
    {"role": "assistant", "content": "A cat."},
]

TEXT_MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
]

IMAGES = [Image.new("RGB", (32, 32), (255, 255, 255))]

NO_IMAGES = []

INPUT_IDS = [0, 1, 2, 3, 4]

LABELS = [0, 1, 2, 3, 4]

FEATURE_SEQLENS = {"token_type_ids": 1024}


def _get_mm_inputs(processor: "ProcessorMixin") -> Dict[str, "torch.Tensor"]:
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    return image_processor(images=IMAGES, return_tensors="pt")


def _is_close(batch_a: Dict[str, Any], batch_b: Dict[str, Any]) -> None:
    assert batch_a.keys() == batch_b.keys()
    for key in batch_a.keys():
        if isinstance(batch_a[key], torch.Tensor):
            assert torch.allclose(batch_a[key], batch_b[key], rtol=1e-4, atol=1e-5)
        else:
            assert batch_a[key] == batch_b[key]


def _load_tokenizer_module(model_name_or_path: str) -> Tuple["PreTrainedTokenizer", "ProcessorMixin"]:
    model_args = ModelArguments(model_name_or_path=model_name_or_path)
    tokenizer_module = load_tokenizer(model_args)
    return tokenizer_module["tokenizer"], tokenizer_module["processor"]


def test_base_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path=TINY_LLAMA)
    base_plugin = get_mm_plugin(name="base", image_token="<image>")
    # test mm_messages
    assert base_plugin.process_messages(MM_MESSAGES, IMAGES, processor) == MM_MESSAGES
    assert base_plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(base_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), {})
    # test text_messages
    assert base_plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, processor) == TEXT_MESSAGES
    assert base_plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(base_plugin.get_mm_inputs(NO_IMAGES, FEATURE_SEQLENS, processor), {})


def test_llava_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="llava-hf/llava-1.5-7b-hf")
    image_seqlen = 576

    mm_inputs = _get_mm_inputs(processor)
    expected_mm_messages = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]

    llava_plugin = get_mm_plugin(name="llava", image_token="<image>")
    # test mm_messages
    assert llava_plugin.process_messages(MM_MESSAGES, IMAGES, processor) == expected_mm_messages
    assert llava_plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(llava_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    # test text_messages
    assert llava_plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, processor) == TEXT_MESSAGES
    assert llava_plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(llava_plugin.get_mm_inputs(NO_IMAGES, FEATURE_SEQLENS, processor), {"pixel_values": None})


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_paligemma_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="google/paligemma-3b-pt-224")
    image_seqlen = 256

    mm_inputs = _get_mm_inputs(processor)
    mm_inputs["token_type_ids"] = [[0] * image_seqlen + [1] * (1024 - image_seqlen)]
    expected_mm_messages = [
        {key: value.replace("<image>", "") for key, value in message.items()} for message in MM_MESSAGES
    ]
    expected_input_ids = [tokenizer.convert_tokens_to_ids("<image>")] * image_seqlen + INPUT_IDS
    expected_labels = [-100] * image_seqlen + LABELS

    paligemma_plugin = get_mm_plugin(name="paligemma", image_token="<image>")
    # test mm_messages
    assert paligemma_plugin.process_messages(MM_MESSAGES, IMAGES, processor) == expected_mm_messages
    assert paligemma_plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, tokenizer, processor) == (
        expected_input_ids,
        expected_labels,
    )
    _is_close(paligemma_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    # test text_messages
    assert paligemma_plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, processor) == TEXT_MESSAGES
    assert paligemma_plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, tokenizer, processor) == (
        INPUT_IDS,
        LABELS,
    )
    _is_close(
        paligemma_plugin.get_mm_inputs(NO_IMAGES, FEATURE_SEQLENS, processor),
        {"pixel_values": None, "token_type_ids": [[1] * 1024]},
    )


def test_qwen2_vl_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    image_seqlen = 4

    mm_inputs = _get_mm_inputs(processor)
    expected_mm_messages = [
        {
            key: value.replace("<image>", "<|vision_start|>{}<|vision_end|>".format("<|image_pad|>" * image_seqlen))
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]

    qwen2_vl_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>")
    # test mm_messages
    assert qwen2_vl_plugin.process_messages(MM_MESSAGES, IMAGES, processor) == expected_mm_messages
    assert qwen2_vl_plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(qwen2_vl_plugin.get_mm_inputs(IMAGES, FEATURE_SEQLENS, processor), mm_inputs)
    # test text_messages
    assert qwen2_vl_plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, processor) == TEXT_MESSAGES
    assert qwen2_vl_plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, tokenizer, processor) == (INPUT_IDS, LABELS)
    _is_close(
        qwen2_vl_plugin.get_mm_inputs(NO_IMAGES, FEATURE_SEQLENS, processor),
        {"pixel_values": None, "image_grid_thw": None},
    )
