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

import json

import pytest
from transformers import AutoTokenizer

from llamafactory.v1.config import DataArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.utils.rendering import Renderer
from llamafactory.v1.utils.types import Processor


HF_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is LLM?"},
    {"role": "assistant", "content": "LLM stands for Large Language Model."},
]

V1_MESSAGES = [
    {"role": "system", "content": [{"type": "text", "value": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "value": "What is LLM?"}]},
    {"role": "assistant", "content": [{"type": "text", "value": "LLM stands for Large Language Model."}]},
]

HF_MESSAGES_WITH_TOOLS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 6*8?"},
    {
        "role": "assistant",
        "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 6, "b": 8}}}],
    },
    {"role": "tool", "content": "48."},
    {"role": "assistant", "content": "The result of 6*8 is 48."},
]

V1_MESSAGES_WITH_TOOLS = [
    {"role": "system", "content": [{"type": "text", "value": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "value": "What is 6*8?"}]},
    {
        "role": "assistant",
        "content": [{"type": "tool_call", "value": json.dumps({"name": "multiply", "arguments": {"a": 6, "b": 8}})}],
        "loss_weight": 0.0,
    },
    {"role": "tool", "content": [{"type": "text", "value": "48."}]},
    {"role": "assistant", "content": [{"type": "text", "value": "The result of 6*8 is 48."}]},
]

V1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "A function that multiplies two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number to multiply"},
                    "b": {"type": "number", "description": "The second number to multiply"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def test_chatml_rendering():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)

    hf_inputs = tokenizer.apply_chat_template(HF_MESSAGES[:-1], add_generation_prompt=True)
    v1_inputs = renderer.render_messages(V1_MESSAGES[:-1], is_generate=True)
    assert v1_inputs["input_ids"] == hf_inputs
    assert v1_inputs["attention_mask"] == [1] * len(hf_inputs)
    assert v1_inputs["labels"] == [-100] * len(hf_inputs)
    assert v1_inputs["loss_weights"] == [0.0] * len(hf_inputs)

    hf_inputs_part = tokenizer.apply_chat_template(HF_MESSAGES[:-1], add_generation_prompt=False)
    hf_inputs_full = tokenizer.apply_chat_template(HF_MESSAGES, add_generation_prompt=False)
    v1_inputs_full = renderer.render_messages(V1_MESSAGES, is_generate=False)
    assert v1_inputs_full["input_ids"] == hf_inputs_full
    assert v1_inputs_full["attention_mask"] == [1] * len(hf_inputs_full)
    assert v1_inputs_full["labels"] == [-100] * len(hf_inputs_part) + hf_inputs_full[len(hf_inputs_part) :]
    assert v1_inputs_full["loss_weights"] == [0.0] * len(hf_inputs_part) + [1.0] * (
        len(hf_inputs_full) - len(hf_inputs_part)
    )


def test_chatml_parse():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)
    generated_text = "LLM stands for Large Language Model."
    parsed_message = renderer.parse_message(generated_text)
    assert parsed_message == V1_MESSAGES[-1]


@pytest.mark.parametrize("num_samples", [16])
def test_chatml_rendering_remote(num_samples: int):
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)
    data_args = DataArguments(train_dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args.train_dataset)
    for index in range(num_samples):
        v1_inputs = renderer.render_messages(data_engine[index]["messages"], is_generate=True)
        prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
        print(tokenizer.decode(v1_inputs["input_ids"][: len(prefix)]))
        assert v1_inputs["input_ids"][: len(prefix)] == prefix


def test_qwen3_nothink_rendering():
    tokenizer: Processor = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    renderer = Renderer(template="qwen3_nothink", processor=tokenizer)

    hf_inputs = tokenizer.apply_chat_template(HF_MESSAGES_WITH_TOOLS[:-1], tools=V1_TOOLS, add_generation_prompt=True)
    v1_inputs = renderer.render_messages(V1_MESSAGES_WITH_TOOLS[:-1], tools=json.dumps(V1_TOOLS), is_generate=True)
    assert v1_inputs["input_ids"] == hf_inputs
    assert v1_inputs["attention_mask"] == [1] * len(hf_inputs)
    assert v1_inputs["labels"] == [-100] * len(hf_inputs)
    assert v1_inputs["loss_weights"] == [0.0] * len(hf_inputs)

    hf_inputs_part = tokenizer.apply_chat_template(
        HF_MESSAGES_WITH_TOOLS[:-1], tools=V1_TOOLS, add_generation_prompt=False
    )
    hf_inputs_full = tokenizer.apply_chat_template(HF_MESSAGES_WITH_TOOLS, tools=V1_TOOLS, add_generation_prompt=False)
    v1_inputs_full = renderer.render_messages(V1_MESSAGES_WITH_TOOLS, tools=json.dumps(V1_TOOLS), is_generate=False)
    assert v1_inputs_full["input_ids"] == hf_inputs_full
    assert v1_inputs_full["attention_mask"] == [1] * len(hf_inputs_full)
    assert v1_inputs_full["labels"] == [-100] * len(hf_inputs_part) + hf_inputs_full[len(hf_inputs_part) :]
    assert v1_inputs_full["loss_weights"] == [0.0] * len(hf_inputs_part) + [1.0] * (
        len(hf_inputs_full) - len(hf_inputs_part)
    )


def test_qwen3_nothink_parse():
    tokenizer: Processor = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    renderer = Renderer(template="qwen3_nothink", processor=tokenizer)
    generated_text = (
        "<thinking>I need to use the multiply function to calculate 6*8.</thinking>"
        "Let me call the multiply function."
        '<tool_call>{"name": "multiply", "arguments": {"a": 6, "b": 8}}</tool_call>'
    )
    parsed_message = renderer.parse_message(generated_text)
    assert parsed_message == {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "value": "I need to use the multiply function to calculate 6*8."},
            {"type": "text", "value": "Let me call the multiply function."},
            {"type": "tool_call", "value": json.dumps({"name": "multiply", "arguments": {"a": 6, "b": 8}})},
        ],
    }


@pytest.mark.parametrize("num_samples", [8])
def test_qwen3_nothink_rendering_remote(num_samples: int):
    tokenizer: Processor = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    renderer = Renderer(template="qwen3_nothink", processor=tokenizer)
    data_args = DataArguments(train_dataset="llamafactory/reason-tool-use-demo-1500")
    data_engine = DataEngine(data_args.train_dataset)
    for index in range(num_samples):
        v1_inputs = renderer.render_messages(data_engine[index]["messages"], tools=data_engine[index]["tools"])
        prefix_text = (
            "<|im_start|>system\nYou are a methodical and expert assistant. "
            "Your primary goal is to solve user requests by leveraging a set of available tools. "
            "You must reason for the best course of action in a structured manner before responding.\n\n"
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            '{"type": "function", "function": {"name":'
        )
        prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
        print(tokenizer.decode(v1_inputs["input_ids"][: len(prefix)]))
        assert v1_inputs["input_ids"][: len(prefix)] == prefix


def test_process_sft_samples():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)
    hf_inputs = tokenizer.apply_chat_template(HF_MESSAGES)

    samples = [{"messages": V1_MESSAGES, "extra_info": "test", "_dataset_name": "default"}]
    model_inputs = renderer.process_samples(samples)
    assert len(model_inputs) == 1
    assert model_inputs[0]["input_ids"] == hf_inputs
    assert model_inputs[0]["extra_info"] == "test"
    assert model_inputs[0]["_dataset_name"] == "default"


def test_process_dpo_samples():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)
    hf_inputs = tokenizer.apply_chat_template(HF_MESSAGES)

    samples = [
        {
            "chosen_messages": V1_MESSAGES,
            "rejected_messages": V1_MESSAGES,
            "extra_info": "test",
            "_dataset_name": "default",
        }
    ]
    model_inputs = renderer.process_samples(samples)
    assert len(model_inputs) == 1
    assert model_inputs[0]["input_ids"] == hf_inputs * 2
    assert model_inputs[0]["token_type_ids"] == [1] * len(hf_inputs) + [2] * len(hf_inputs)
    assert model_inputs[0]["extra_info"] == "test"
    assert model_inputs[0]["_dataset_name"] == "default"


if __name__ == "__main__":
    """
    python -m tests_v1.core.utils.test_rendering
    """
    test_chatml_rendering()
    test_chatml_parse()
    test_chatml_rendering_remote(16)
    test_qwen3_nothink_rendering()
    test_qwen3_nothink_parse()
    test_qwen3_nothink_rendering_remote(16)
    test_process_sft_samples()
    test_process_dpo_samples()
