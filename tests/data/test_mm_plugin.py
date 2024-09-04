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
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import pytest
import torch
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.hparams import ModelArguments
from llamafactory.model import load_tokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    from llamafactory.data.mm_plugin import BasePlugin


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

NO_VIDEOS = []

IMGLENS = [1]

NO_IMGLENS = [0]

NO_VIDLENS = [0]

INPUT_IDS = [0, 1, 2, 3, 4]

LABELS = [0, 1, 2, 3, 4]

SEQLENS = [1024]


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


def _check_plugin(
    plugin: "BasePlugin",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    expected_mm_messages: Sequence[Dict[str, str]] = MM_MESSAGES,
    expected_input_ids: List[int] = INPUT_IDS,
    expected_labels: List[int] = LABELS,
    expected_mm_inputs: Dict[str, Any] = {},
    expected_no_mm_inputs: Dict[str, Any] = {},
) -> None:
    # test mm_messages
    assert plugin.process_messages(MM_MESSAGES, IMAGES, NO_VIDEOS, processor) == expected_mm_messages
    assert plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, NO_VIDEOS, tokenizer, processor) == (
        expected_input_ids,
        expected_labels,
    )
    _is_close(
        plugin.get_mm_inputs(IMAGES, NO_VIDEOS, IMGLENS, NO_VIDLENS, SEQLENS, processor),
        expected_mm_inputs,
    )
    # test text_messages
    assert plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, NO_VIDEOS, processor) == TEXT_MESSAGES
    assert plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, NO_VIDEOS, tokenizer, processor) == (
        INPUT_IDS,
        LABELS,
    )
    _is_close(
        plugin.get_mm_inputs(NO_IMAGES, NO_VIDEOS, NO_IMGLENS, NO_VIDLENS, SEQLENS, processor),
        expected_no_mm_inputs,
    )


def test_base_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path=TINY_LLAMA)
    base_plugin = get_mm_plugin(name="base", image_token="<image>")
    check_inputs = {"plugin": base_plugin, "tokenizer": tokenizer, "processor": processor}
    _check_plugin(**check_inputs)


def test_llava_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="llava-hf/llava-1.5-7b-hf")
    llava_plugin = get_mm_plugin(name="llava", image_token="<image>")
    image_seqlen = 576
    check_inputs = {"plugin": llava_plugin, "tokenizer": tokenizer, "processor": processor}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(processor)
    _check_plugin(**check_inputs)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_paligemma_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="google/paligemma-3b-pt-224")
    paligemma_plugin = get_mm_plugin(name="paligemma", image_token="<image>")
    image_seqlen = 256
    check_inputs = {"plugin": paligemma_plugin, "tokenizer": tokenizer, "processor": processor}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "") for key, value in message.items()} for message in MM_MESSAGES
    ]
    check_inputs["expected_input_ids"] = [tokenizer.convert_tokens_to_ids("<image>")] * image_seqlen + INPUT_IDS
    check_inputs["expected_labels"] = [-100] * image_seqlen + LABELS
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(processor)
    check_inputs["expected_mm_inputs"]["token_type_ids"] = [[0] * image_seqlen + [1] * (1024 - image_seqlen)]
    check_inputs["expected_no_mm_inputs"] = {"token_type_ids": [[1] * 1024]}
    _check_plugin(**check_inputs)


def test_qwen2_vl_plugin():
    tokenizer, processor = _load_tokenizer_module(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    qwen2_vl_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>")
    image_seqlen = 4
    check_inputs = {"plugin": qwen2_vl_plugin, "tokenizer": tokenizer, "processor": processor}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace("<image>", "<|vision_start|>{}<|vision_end|>".format("<|image_pad|>" * image_seqlen))
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(processor)
    _check_plugin(**check_inputs)
