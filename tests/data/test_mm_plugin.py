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
from typing import TYPE_CHECKING, Any, Dict, List, Sequence

import pytest
import torch
from PIL import Image

from llamafactory.data.mm_plugin import get_mm_plugin
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    from llamafactory.data.mm_plugin import BasePlugin
    from llamafactory.model.loader import TokenizerModule


HF_TOKEN = os.getenv("HF_TOKEN")

TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

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

NO_AUDIOS = []

IMGLENS = [1]

NO_IMGLENS = [0]

NO_VIDLENS = [0]

NO_AUDLENS = [0]

INPUT_IDS = [0, 1, 2, 3, 4]

LABELS = [0, 1, 2, 3, 4]

BATCH_IDS = [[1] * 1024]


def _get_mm_inputs(processor: "ProcessorMixin") -> Dict[str, "torch.Tensor"]:
    image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
    return image_processor(images=IMAGES, return_tensors="pt")


def _is_close(batch_a: Dict[str, Any], batch_b: Dict[str, Any]) -> None:
    assert batch_a.keys() == batch_b.keys()
    for key in batch_a.keys():
        if isinstance(batch_a[key], torch.Tensor):
            assert torch.allclose(batch_a[key], batch_b[key], rtol=1e-4, atol=1e-5)
        elif isinstance(batch_a[key], list) and all(isinstance(item, torch.Tensor) for item in batch_a[key]):
            assert len(batch_a[key]) == len(batch_b[key])
            for tensor_a, tensor_b in zip(batch_a[key], batch_b[key]):
                assert torch.allclose(tensor_a, tensor_b, rtol=1e-4, atol=1e-5)
        else:
            assert batch_a[key] == batch_b[key]


def _load_tokenizer_module(model_name_or_path: str) -> "TokenizerModule":
    model_args, *_ = get_infer_args({"model_name_or_path": model_name_or_path, "template": "default"})
    return load_tokenizer(model_args)


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
    if plugin.__class__.__name__ != "BasePlugin":
        assert plugin.process_messages(MM_MESSAGES, IMAGES, NO_VIDEOS, NO_AUDIOS, processor) == expected_mm_messages
        assert plugin.process_token_ids(INPUT_IDS, LABELS, IMAGES, NO_VIDEOS, NO_AUDIOS, tokenizer, processor) == (
            expected_input_ids,
            expected_labels,
        )
        _is_close(
            plugin.get_mm_inputs(IMAGES, NO_VIDEOS, NO_AUDIOS, IMGLENS, NO_VIDLENS, NO_AUDLENS, BATCH_IDS, processor),
            expected_mm_inputs,
        )

    # test text_messages
    assert plugin.process_messages(TEXT_MESSAGES, NO_IMAGES, NO_VIDEOS, NO_AUDIOS, processor) == TEXT_MESSAGES
    assert plugin.process_token_ids(INPUT_IDS, LABELS, NO_IMAGES, NO_VIDEOS, NO_AUDIOS, tokenizer, processor) == (
        INPUT_IDS,
        LABELS,
    )
    _is_close(
        plugin.get_mm_inputs(
            NO_IMAGES, NO_VIDEOS, NO_AUDIOS, NO_IMGLENS, NO_VIDLENS, NO_AUDLENS, BATCH_IDS, processor
        ),
        expected_no_mm_inputs,
    )


def test_base_plugin():
    tokenizer_module = _load_tokenizer_module(model_name_or_path=TINY_LLAMA)
    base_plugin = get_mm_plugin(name="base")
    check_inputs = {"plugin": base_plugin, **tokenizer_module}
    _check_plugin(**check_inputs)


def test_llava_plugin():
    image_seqlen = 576
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/llava-1.5-7b-hf")
    llava_plugin = get_mm_plugin(name="llava", image_token="<image>")
    check_inputs = {"plugin": llava_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


def test_llava_next_plugin():
    image_seqlen = 1176
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/llava-v1.6-vicuna-7b-hf")
    llava_next_plugin = get_mm_plugin(name="llava_next", image_token="<image>")
    check_inputs = {"plugin": llava_next_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


def test_llava_next_video_plugin():
    image_seqlen = 1176
    tokenizer_module = _load_tokenizer_module(model_name_or_path="llava-hf/LLaVA-NeXT-Video-7B-hf")
    llava_next_video_plugin = get_mm_plugin(name="llava_next_video", image_token="<image>", video_token="<video>")
    check_inputs = {"plugin": llava_next_video_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_paligemma_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="google/paligemma-3b-pt-224")
    paligemma_plugin = get_mm_plugin(name="paligemma", image_token="<image>")
    check_inputs = {"plugin": paligemma_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "") for key, value in message.items()} for message in MM_MESSAGES
    ]
    check_inputs["expected_input_ids"] = [
        tokenizer_module["tokenizer"].convert_tokens_to_ids(paligemma_plugin.image_token)
    ] * image_seqlen + INPUT_IDS
    check_inputs["expected_labels"] = [-100] * image_seqlen + LABELS
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"]["token_type_ids"] = [[0] * image_seqlen + [1] * (1024 - image_seqlen)]
    check_inputs["expected_no_mm_inputs"] = {"token_type_ids": [[1] * 1024]}
    _check_plugin(**check_inputs)


def test_pixtral_plugin():
    image_slice_height, image_slice_width = 2, 2
    tokenizer_module = _load_tokenizer_module(model_name_or_path="mistral-community/pixtral-12b")
    pixtral_plugin = get_mm_plugin(name="pixtral", image_token="[IMG]")
    check_inputs = {"plugin": pixtral_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace(
                "<image>",
                ("{}[IMG_BREAK]".format("[IMG]" * image_slice_width) * image_slice_height).rsplit("[IMG_BREAK]", 1)[0]
                + "[IMG_END]",
            )
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    check_inputs["expected_mm_inputs"].pop("image_sizes")
    check_inputs["expected_mm_inputs"]["pixel_values"] = check_inputs["expected_mm_inputs"]["pixel_values"][0]
    _check_plugin(**check_inputs)


def test_qwen2_vl_plugin():
    image_seqlen = 4
    tokenizer_module = _load_tokenizer_module(model_name_or_path="Qwen/Qwen2-VL-7B-Instruct")
    qwen2_vl_plugin = get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>")
    check_inputs = {"plugin": qwen2_vl_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {
            key: value.replace("<image>", "<|vision_start|>{}<|vision_end|>".format("<|image_pad|>" * image_seqlen))
            for key, value in message.items()
        }
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)


def test_video_llava_plugin():
    image_seqlen = 256
    tokenizer_module = _load_tokenizer_module(model_name_or_path="LanguageBind/Video-LLaVA-7B-hf")
    video_llava_plugin = get_mm_plugin(name="video_llava", image_token="<image>", video_token="<video>")
    check_inputs = {"plugin": video_llava_plugin, **tokenizer_module}
    check_inputs["expected_mm_messages"] = [
        {key: value.replace("<image>", "<image>" * image_seqlen) for key, value in message.items()}
        for message in MM_MESSAGES
    ]
    check_inputs["expected_mm_inputs"] = _get_mm_inputs(tokenizer_module["processor"])
    _check_plugin(**check_inputs)
