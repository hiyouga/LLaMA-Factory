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
from typing import TYPE_CHECKING, Sequence

import pytest
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.template import _get_jinja_template
from llamafactory.hparams import DataArguments


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


HF_TOKEN = os.getenv("HF_TOKEN")

TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "很高兴认识你！"},
]


def _check_tokenization(
    tokenizer: "PreTrainedTokenizer", batch_input_ids: Sequence[Sequence[int]], batch_text: Sequence[str]
) -> None:
    r"""
    Checks token ids and texts.

    encode(text) == token_ids
    decode(token_ids) == text
    """
    for input_ids, text in zip(batch_input_ids, batch_text):
        assert tokenizer.encode(text, add_special_tokens=False) == input_ids
        assert tokenizer.decode(input_ids) == text


def _check_template(model_id: str, template_name: str, prompt_str: str, answer_str: str, use_fast: bool) -> None:
    r"""
    Checks template.

    Args:
        model_id: the model id on hugging face hub.
        template_name: the template name.
        prompt_str: the string corresponding to the prompt part.
        answer_str: the string corresponding to the answer part.
        use_fast: whether to use fast tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, token=HF_TOKEN)
    content_str = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    content_ids = tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=template_name))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    assert content_str == prompt_str + answer_str
    assert content_ids == prompt_ids + answer_ids
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.parametrize("use_fast", [True, False])
def test_encode_oneturn(use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA, use_fast=use_fast)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    prompt_str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nI am fine!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = "很高兴认识你！<|eot_id|>"
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.parametrize("use_fast", [True, False])
def test_encode_multiturn(use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA, use_fast=use_fast)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    encoded_pairs = template.encode_multiturn(tokenizer, MESSAGES)
    prompt_str_1 = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str_1 = "I am fine!<|eot_id|>"
    prompt_str_2 = (
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str_2 = "很高兴认识你！<|eot_id|>"
    _check_tokenization(
        tokenizer,
        (encoded_pairs[0][0], encoded_pairs[0][1], encoded_pairs[1][0], encoded_pairs[1][1]),
        (prompt_str_1, answer_str_1, prompt_str_2, answer_str_2),
    )


@pytest.mark.parametrize("use_fast", [True, False])
def test_jinja_template(use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA, use_fast=use_fast)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA, use_fast=use_fast)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    tokenizer.chat_template = _get_jinja_template(template, tokenizer)  # llama3 template no replace
    assert tokenizer.chat_template != ref_tokenizer.chat_template
    assert tokenizer.apply_chat_template(MESSAGES) == ref_tokenizer.apply_chat_template(MESSAGES)


def test_get_stop_token_ids():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert set(template.get_stop_token_ids(tokenizer)) == {128008, 128009}


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_gemma_template(use_fast: bool):
    prompt_str = (
        "<bos><start_of_turn>user\nHow are you<end_of_turn>\n"
        "<start_of_turn>model\nI am fine!<end_of_turn>\n"
        "<start_of_turn>user\n你好<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = "很高兴认识你！<end_of_turn>\n"
    _check_template("google/gemma-2-9b-it", "gemma", prompt_str, answer_str, use_fast)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_llama3_template(use_fast: bool):
    prompt_str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nI am fine!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = "很高兴认识你！<|eot_id|>"
    _check_template("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", prompt_str, answer_str, use_fast)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize(
    "use_fast", [True, pytest.param(False, marks=pytest.mark.xfail(reason="Phi-4 slow tokenizer is broken."))]
)
def test_phi4_template(use_fast: bool):
    prompt_str = (
        "<|im_start|>user<|im_sep|>How are you<|im_end|>"
        "<|im_start|>assistant<|im_sep|>I am fine!<|im_end|>"
        "<|im_start|>user<|im_sep|>你好<|im_end|>"
        "<|im_start|>assistant<|im_sep|>"
    )
    answer_str = "很高兴认识你！<|im_end|>"
    _check_template("microsoft/phi-4", "phi4", prompt_str, answer_str, use_fast)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")  # TODO: why it is gated?
@pytest.mark.parametrize("use_fast", [True, False])
def test_qwen_template(use_fast: bool):
    prompt_str = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHow are you<|im_end|>\n"
        "<|im_start|>assistant\nI am fine!<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = "很高兴认识你！<|im_end|>\n"
    _check_template("Qwen/Qwen2-7B-Instruct", "qwen", prompt_str, answer_str, use_fast)


@pytest.mark.parametrize("use_fast", [True, False])
@pytest.mark.xfail(reason="Yi tokenizer is broken.")
def test_yi_template(use_fast: bool):
    prompt_str = (
        "<|im_start|>user\nHow are you<|im_end|>\n"
        "<|im_start|>assistant\nI am fine!<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = "很高兴认识你！<|im_end|>\n"
    _check_template("01-ai/Yi-1.5-6B-Chat", "yi", prompt_str, answer_str, use_fast)
