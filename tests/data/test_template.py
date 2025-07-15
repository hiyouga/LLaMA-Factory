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
from typing import TYPE_CHECKING

import pytest
from transformers import AutoTokenizer

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.template import parse_template
from llamafactory.hparams import DataArguments


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


HF_TOKEN = os.getenv("HF_TOKEN")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")
TINY_LLAMA4 = os.getenv("TINY_LLAMA4", "llamafactory/tiny-random-Llama-4")

MESSAGES = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "很高兴认识你！"},
]

MESSAGES_WITH_THOUGHT = [
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "<think>\nModel thought here\n</think>\n\nI am fine!"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "<think>\n模型思考内容\n</think>\n\n很高兴认识你！"},
]


def _check_tokenization(
    tokenizer: "PreTrainedTokenizer", batch_input_ids: list[list[int]], batch_text: list[str]
) -> None:
    r"""Check token ids and texts.

    encode(text) == token_ids
    decode(token_ids) == text
    """
    for input_ids, text in zip(batch_input_ids, batch_text):
        assert tokenizer.encode(text, add_special_tokens=False) == input_ids
        assert tokenizer.decode(input_ids) == text


def _check_template(
    model_id: str,
    template_name: str,
    prompt_str: str,
    answer_str: str,
    use_fast: bool,
    messages: list[dict[str, str]] = MESSAGES,
) -> None:
    r"""Check template.

    Args:
        model_id: the model id on hugging face hub.
        template_name: the template name.
        prompt_str: the string corresponding to the prompt part.
        answer_str: the string corresponding to the answer part.
        use_fast: whether to use fast tokenizer.
        messages: the list of messages.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, token=HF_TOKEN)
    content_str = tokenizer.apply_chat_template(messages, tokenize=False)
    content_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=template_name))
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, messages)
    assert content_str == prompt_str + answer_str
    assert content_ids == prompt_ids + answer_ids
    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.parametrize("use_fast", [True, False])
def test_encode_oneturn(use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3, use_fast=use_fast)
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
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3, use_fast=use_fast)
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
@pytest.mark.parametrize("cot_messages", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_reasoning_encode_oneturn(use_fast: bool, cot_messages: bool, enable_thinking: bool):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=use_fast)
    data_args = DataArguments(template="qwen3", enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    prompt_ids, answer_ids = template.encode_oneturn(tokenizer, MESSAGES_WITH_THOUGHT if cot_messages else MESSAGES)

    prompt_str = (
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        f"{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )
    if not cot_messages or enable_thinking is False:
        answer_str = f"{MESSAGES[3]['content']}<|im_end|>\n"
        if enable_thinking:
            answer_str = "<think>\n\n</think>\n\n" + answer_str
        else:
            prompt_str = prompt_str + "<think>\n\n</think>\n\n"
    else:
        answer_str = f"{MESSAGES_WITH_THOUGHT[3]['content']}<|im_end|>\n"

    _check_tokenization(tokenizer, (prompt_ids, answer_ids), (prompt_str, answer_str))


@pytest.mark.parametrize("use_fast", [True, False])
@pytest.mark.parametrize("cot_messages", [True, False])
@pytest.mark.parametrize("enable_thinking", [True, False, None])
def test_reasoning_encode_multiturn(use_fast: bool, cot_messages: bool, enable_thinking: bool):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", use_fast=use_fast)
    data_args = DataArguments(template="qwen3", enable_thinking=enable_thinking)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    encoded_pairs = template.encode_multiturn(tokenizer, MESSAGES_WITH_THOUGHT if cot_messages else MESSAGES)

    messages = MESSAGES if not cot_messages or enable_thinking is False else MESSAGES_WITH_THOUGHT
    prompt_str_1 = f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    answer_str_1 = f"{messages[1]['content']}<|im_end|>\n"
    prompt_str_2 = f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n<|im_start|>assistant\n"
    answer_str_2 = f"{messages[3]['content']}<|im_end|>\n"
    if not cot_messages or enable_thinking is False:
        if enable_thinking:
            answer_str_1 = "<think>\n\n</think>\n\n" + answer_str_1
            answer_str_2 = "<think>\n\n</think>\n\n" + answer_str_2
        else:
            prompt_str_1 = prompt_str_1 + "<think>\n\n</think>\n\n"
            prompt_str_2 = prompt_str_2 + "<think>\n\n</think>\n\n"

    _check_tokenization(
        tokenizer,
        (encoded_pairs[0][0], encoded_pairs[0][1], encoded_pairs[1][0], encoded_pairs[1][1]),
        (prompt_str_1, answer_str_1, prompt_str_2, answer_str_2),
    )


@pytest.mark.parametrize("use_fast", [True, False])
def test_jinja_template(use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3, use_fast=use_fast)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3, use_fast=use_fast)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    tokenizer.chat_template = template._get_jinja_template(tokenizer)  # llama3 template no replace
    assert tokenizer.chat_template != ref_tokenizer.chat_template
    assert tokenizer.apply_chat_template(MESSAGES) == ref_tokenizer.apply_chat_template(MESSAGES)


def test_ollama_modelfile():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert template.get_ollama_modelfile(tokenizer) == (
        "# ollama modelfile auto-generated by llamafactory\n\n"
        "FROM .\n\n"
        'TEMPLATE """<|begin_of_text|>'
        "{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>{{ end }}"
        '{{ range .Messages }}{{ if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n{{ .Content }}'
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        '{{ else if eq .Role "assistant" }}{{ .Content }}<|eot_id|>{{ end }}{{ end }}"""\n\n'
        'PARAMETER stop "<|eom_id|>"\n'
        'PARAMETER stop "<|eot_id|>"\n'
        "PARAMETER num_ctx 4096\n"
    )


def test_get_stop_token_ids():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="llama3"))
    assert set(template.get_stop_token_ids(tokenizer)) == {128008, 128009}


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_gemma_template(use_fast: bool):
    prompt_str = (
        f"<bos><start_of_turn>user\n{MESSAGES[0]['content']}<end_of_turn>\n"
        f"<start_of_turn>model\n{MESSAGES[1]['content']}<end_of_turn>\n"
        f"<start_of_turn>user\n{MESSAGES[2]['content']}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<end_of_turn>\n"
    _check_template("google/gemma-3-4b-it", "gemma", prompt_str, answer_str, use_fast)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_gemma2_template(use_fast: bool):
    prompt_str = (
        f"<bos><start_of_turn>user\n{MESSAGES[0]['content']}<end_of_turn>\n"
        f"<start_of_turn>model\n{MESSAGES[1]['content']}<end_of_turn>\n"
        f"<start_of_turn>user\n{MESSAGES[2]['content']}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<end_of_turn>\n"
    _check_template("google/gemma-2-2b-it", "gemma2", prompt_str, answer_str, use_fast)


@pytest.mark.skipif(not HF_TOKEN, reason="Gated model.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_llama3_template(use_fast: bool):
    prompt_str = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{MESSAGES[0]['content']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{MESSAGES[1]['content']}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{MESSAGES[2]['content']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|eot_id|>"
    _check_template("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", prompt_str, answer_str, use_fast)


@pytest.mark.parametrize(
    "use_fast", [True, pytest.param(False, marks=pytest.mark.xfail(reason="Llama 4 has no slow tokenizer."))]
)
def test_llama4_template(use_fast: bool):
    prompt_str = (
        f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n{MESSAGES[0]['content']}<|eot|>"
        f"<|header_start|>assistant<|header_end|>\n\n{MESSAGES[1]['content']}<|eot|>"
        f"<|header_start|>user<|header_end|>\n\n{MESSAGES[2]['content']}<|eot|>"
        "<|header_start|>assistant<|header_end|>\n\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|eot|>"
    _check_template(TINY_LLAMA4, "llama4", prompt_str, answer_str, use_fast)


@pytest.mark.parametrize(
    "use_fast",
    [
        pytest.param(True, marks=pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")),
        pytest.param(False, marks=pytest.mark.xfail(reason="Phi-4 slow tokenizer is broken.")),
    ],
)
def test_phi4_template(use_fast: bool):
    prompt_str = (
        f"<|im_start|>user<|im_sep|>{MESSAGES[0]['content']}<|im_end|>"
        f"<|im_start|>assistant<|im_sep|>{MESSAGES[1]['content']}<|im_end|>"
        f"<|im_start|>user<|im_sep|>{MESSAGES[2]['content']}<|im_end|>"
        "<|im_start|>assistant<|im_sep|>"
    )
    answer_str = f"{MESSAGES[3]['content']}<|im_end|>"
    _check_template("microsoft/phi-4", "phi4", prompt_str, answer_str, use_fast)


@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
@pytest.mark.parametrize("use_fast", [True, False])
def test_qwen2_5_template(use_fast: bool):
    prompt_str = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer_str = f"{MESSAGES[3]['content']}<|im_end|>\n"
    _check_template("Qwen/Qwen2.5-7B-Instruct", "qwen", prompt_str, answer_str, use_fast)


@pytest.mark.parametrize("use_fast", [True, False])
@pytest.mark.parametrize("cot_messages", [True, False])
def test_qwen3_template(use_fast: bool, cot_messages: bool):
    prompt_str = (
        f"<|im_start|>user\n{MESSAGES[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{MESSAGES[1]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{MESSAGES[2]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if not cot_messages:
        answer_str = f"<think>\n\n</think>\n\n{MESSAGES[3]['content']}<|im_end|>\n"
        messages = MESSAGES
    else:
        answer_str = f"{MESSAGES_WITH_THOUGHT[3]['content']}<|im_end|>\n"
        messages = MESSAGES_WITH_THOUGHT

    _check_template("Qwen/Qwen3-8B", "qwen3", prompt_str, answer_str, use_fast, messages=messages)


def test_parse_llama3_template():
    tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA3, token=HF_TOKEN)
    template = parse_template(tokenizer)
    assert template.format_user.slots == [
        "<|start_header_id|>user<|end_header_id|>\n\n{{content}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
    assert template.format_assistant.slots == ["{{content}}<|eot_id|>"]
    assert template.format_system.slots == ["<|start_header_id|>system<|end_header_id|>\n\n{{content}}<|eot_id|>"]
    assert template.format_prefix.slots == ["<|begin_of_text|>"]
    assert template.default_system == ""


@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
def test_parse_qwen_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)
    template = parse_template(tokenizer)
    assert template.__class__.__name__ == "Template"
    assert template.format_user.slots == ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    assert template.format_assistant.slots == ["{{content}}<|im_end|>\n"]
    assert template.format_system.slots == ["<|im_start|>system\n{{content}}<|im_end|>\n"]
    assert template.format_prefix.slots == []
    assert template.default_system == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@pytest.mark.xfail(not HF_TOKEN, reason="Authorization.")
def test_parse_qwen3_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", token=HF_TOKEN)
    template = parse_template(tokenizer)
    assert template.__class__.__name__ == "ReasoningTemplate"
    assert template.format_user.slots == ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
    assert template.format_assistant.slots == ["{{content}}<|im_end|>\n"]
    assert template.format_system.slots == ["<|im_start|>system\n{{content}}<|im_end|>\n"]
    assert template.format_prefix.slots == []
    assert template.default_system == ""
