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

from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "很高兴认识你！"},
]


def test_encode_oneturn():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    template = get_template_and_fix_tokenizer(tokenizer, name="llama3")
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    assert tokenizer.decode(prompt_ids) == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nI am fine!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert tokenizer.decode(answer_ids) == "很高兴认识你！<|eot_id|>"


def test_encode_multiturn():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    template = get_template_and_fix_tokenizer(tokenizer, name="llama3")
    encoded_pairs = template.encode_multiturn(tokenizer, MESSAGES)
    assert tokenizer.decode(encoded_pairs[0][0]) == (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert tokenizer.decode(encoded_pairs[0][1]) == "I am fine!<|eot_id|>"
    assert tokenizer.decode(encoded_pairs[1][0]) == (
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert tokenizer.decode(encoded_pairs[1][1]) == "很高兴认识你！<|eot_id|>"


def test_jinja_template():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    get_template_and_fix_tokenizer(tokenizer, name="llama3")
    assert tokenizer.chat_template != ref_tokenizer.chat_template
    assert tokenizer.apply_chat_template(MESSAGES) == ref_tokenizer.apply_chat_template(MESSAGES)


def test_qwen_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    template = get_template_and_fix_tokenizer(tokenizer, name="qwen")
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    assert tokenizer.decode(prompt_ids) == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHow are you<|im_end|>\n"
        "<|im_start|>assistant\nI am fine!<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert tokenizer.decode(answer_ids) == "很高兴认识你！<|im_end|>"
