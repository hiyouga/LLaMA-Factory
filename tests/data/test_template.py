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
import json
import os
from typing import TYPE_CHECKING, List, Sequence

import pytest
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", None)

TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "很高兴认识你！"},
]

TOOL_MESSAGES = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "获取最新新闻文章",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "要检索的新闻文章类别"},
                        "country": {"type": "string", "description": "获取新闻文章的国家"}
                    },
                    "required": ["category"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_books",
                "description": "根据提供的标准搜索书籍",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "这本书的标题"},
                        "author": {"type": "string", "description": "这本书的作者"},
                        "genre": {"type": "string", "description": "这本书的类型"}
                    }
                }
            }
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": "你能帮我找到最新的美国体育新闻吗？"
        },
        {
            "role": "tool_calls",
            "content": [
                {
                    "type": "function",
                    "function": {"name": "get_news", "arguments": {"category": "运动", "country": "美国"}}
                }
            ]
        },
        {
            "role": "tool",
            "content": json.dumps(
                {"title": "NBA总决赛：湖人队对阵热火队", "link": "NBA官方网站"},
                ensure_ascii=False
            ),
        },
        {
            "role": "tool",
            "content": json.dumps(
                {"title": "NFL：爱国者队击败酋长队", "link": "https://www.nfl.com/新闻"},
                ensure_ascii=False
            ),
        },
        {
            "role": "tool",
            "content": json.dumps(
                {"title": "MLB：道奇队赢得世界系列赛", "link": "https://www.mlb.com/新闻"},
                ensure_ascii=False
            )
        },
        {
            "role": "assistant",
            "content": "1. NBA总决赛：湖人队对阵热火队\n2. NFL：爱国者队击败酋长队\n3. MLB：道奇队赢得世界系列赛"
        }
    ],
}


def _check_tokenization(
    tokenizer: "PreTrainedTokenizer", batch_input_ids: Sequence[Sequence[int]], batch_text: Sequence[str]
) -> None:
    for input_ids, text in zip(batch_input_ids, batch_text):
        assert input_ids == tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(input_ids) == text


def _check_single_template(
    model_id: str, template_name: str, prompt_str: str, answer_str: str, extra_str: str, use_fast: bool
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, token=HF_TOKEN)
    content_str = tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    content_ids = tokenizer.apply_chat_template(MESSAGES, tokenize=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=template_name))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES)
    assert content_str == prompt_str + answer_str + extra_str
    assert content_ids == prompt_ids + answer_ids + tokenizer.encode(extra_str, add_special_tokens=False)
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))
    return content_ids


def _check_template(model_id: str, template_name: str, prompt_str: str, answer_str: str, extra_str: str = "") -> None:
    """
    Checks template for both the slow tokenizer and the fast tokenizer.

    Args:
        model_id: the model id on hugging face hub.
        template_name: the template name.
        prompt_str: the string corresponding to the prompt part.
        answer_str: the string corresponding to the answer part.
        extra_str: the extra string in the jinja template of the original tokenizer.
    """
    slow_ids = _check_single_template(model_id, template_name, prompt_str, answer_str, extra_str, use_fast=False)
    fast_ids = _check_single_template(model_id, template_name, prompt_str, answer_str, extra_str, use_fast=True)
    assert slow_ids == fast_ids


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
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
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
    get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert tokenizer.chat_template != ref_tokenizer.chat_template
    assert tokenizer.apply_chat_template(MESSAGES) == ref_tokenizer.apply_chat_template(MESSAGES)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_gemma_template():
    prompt_str = (
        "<bos><start_of_turn>user\nHow are you<end_of_turn>\n"
        "<start_of_turn>model\nI am fine!<end_of_turn>\n"
        "<start_of_turn>user\n你好<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = "很高兴认识你！"
    _check_template("google/gemma-2-9b-it", "gemma", prompt_str, answer_str, extra_str="<end_of_turn>\n")


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
def test_llama3_template():
    prompt_str = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHow are you<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nI am fine!<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n你好<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = "很高兴认识你！<|eot_id|>"
    _check_template("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", prompt_str, answer_str)


def test_qwen_template():
    prompt_str = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHow are you<|im_end|>\n"
        "<|im_start|>assistant\nI am fine!<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = "很高兴认识你！<|im_end|>"
    _check_template("Qwen/Qwen2-7B-Instruct", "qwen", prompt_str, answer_str, extra_str="\n")


@pytest.mark.xfail(reason="The fast tokenizer of Yi model is corrupted.")
def test_yi_template():
    prompt_str = (
        "<|im_start|>user\nHow are you<|im_end|>\n"
        "<|im_start|>assistant\nI am fine!<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = "很高兴认识你！<|im_end|>"
    _check_template("01-ai/Yi-1.5-6B-Chat", "yi", prompt_str, answer_str)


@pytest.mark.xfail(reason="The fast tokenizer of mistral model is corrupted.")
def test_mistral_template():
    TEMPLATE = r"""
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}
{%- set user_messages = messages | selectattr("role", "equalto", "user") | list %}

{%- for message in lmessages | rejectattr("role", "equalto", "tool") | rejectattr("role", "equalto", "tool_results") | selectattr("tool_calls", "undefined") %}
    {%- if (message["role"] == "user") != (loop.index0 % 2 == 0) %}
        {{- raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif %}
{%- endfor %}

{{- bos_token }}
{%- for message in messages %}
    {%- if message["role"] == "user" %}
        {%- if tools is not none and (message == user_messages[-1]) %}
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() if key != "return" %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
        {%- endif %}
        {{- "[INST] " + message["content"] + "[/INST]" }}
    {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}
        {%- if message.tool_calls is defined %}
            {%- set tool_calls = message.tool_calls %}
        {%- else %}
            {%- set tool_calls = message.content %}
        {%- endif %}
        {{- "[TOOL_CALLS] [" }}
        {%- for tool_call in tool_calls %}
            {%- set out = tool_call.function|tojson %}
            {{- out }}
            {%- if not loop.last %}
                {{- ", " }}
            {%- else %}
                {{- "]" }}
            {%- endif %}
        {%- endfor %}
    {%- elif message["role"] == "assistant" %}
        {{- " " + message["content"] }}
    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}
        {%- if message.content is defined and message.content.content is defined %}
            {%- set content = message.content.content %}
        {%- else %}
            {%- set content = message.content %}
        {%- endif %}
        {{- '[TOOL_RESULTS] {"content": ' + content|string + "}[/TOOL_RESULTS]" }}
    {%- else %}
        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}
    {%- endif %}
{%- endfor %}
"""
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="mistral"))

    content_str = tokenizer.apply_chat_template(
        conversation=TOOL_MESSAGES['messages'],
        tools=TOOL_MESSAGES['tools'],
        chat_template=TEMPLATE,
        tokenize=False
    )
    content_ids = tokenizer.apply_chat_template(
        conversation=TOOL_MESSAGES['messages'],
        tools=TOOL_MESSAGES['tools'],
        chat_template=TEMPLATE,
        tokenize=True
    )
    encoded_pairs = template.encode_multiturn(
        tokenizer,
        [
            TOOL_MESSAGES['messages'][0],
            {
                "role": "function",
                "content": json.dumps([function['function'] for function in TOOL_MESSAGES['messages'][1]['content']])
            },
            {
                "role": "observation",
                "content": json.dumps([item['content'] for item in TOOL_MESSAGES['messages'][2:-1]])
            },
            TOOL_MESSAGES['messages'][-1],
        ],
        tools=json.dumps([tool['function'] for tool in TOOL_MESSAGES['tools']])
    )

    final_ids = []
    for prompt, response in encoded_pairs:
        final_ids.extend(prompt)
        final_ids.extend(response)

    final_str = tokenizer.decode(final_ids)

    assert content_str == final_str
    assert content_ids == final_ids
