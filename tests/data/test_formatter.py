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
from datetime import datetime

from llamafactory.data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter


FUNCTION = {"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}

TOOLS = [
    {
        "name": "test_tool",
        "description": "tool_desc",
        "parameters": {
            "type": "object",
            "properties": {
                "foo": {"type": "string", "description": "foo_desc"},
                "bar": {"type": "number", "description": "bar_desc"},
            },
            "required": ["foo"],
        },
    }
]


def test_empty_formatter():
    formatter = EmptyFormatter(slots=["\n"])
    assert formatter.apply() == ["\n"]


def test_string_formatter():
    formatter = StringFormatter(slots=["<s>", "Human: {{content}}\nAssistant:"])
    assert formatter.apply(content="Hi") == ["<s>", "Human: Hi\nAssistant:"]


def test_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}", "</s>"], tool_format="default")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        """Action: tool_name\nAction Input: {"foo": "bar", "size": 10}\n""",
        "</s>",
    ]


def test_multi_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}", "</s>"], tool_format="default")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        """Action: tool_name\nAction Input: {"foo": "bar", "size": 10}\n"""
        """Action: tool_name\nAction Input: {"foo": "bar", "size": 10}\n""",
        "</s>",
    ]


def test_default_tool_formatter():
    formatter = ToolFormatter(tool_format="default")
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "You have access to the following tools:\n"
        "> Tool Name: test_tool\n"
        "Tool Description: tool_desc\n"
        "Tool Args:\n"
        "  - foo (string, required): foo_desc\n"
        "  - bar (number): bar_desc\n\n"
        "Use the following format if using a tool:\n"
        "```\n"
        "Action: tool name (one of [test_tool])\n"
        "Action Input: the input to the tool, in a JSON format representing the kwargs "
        """(e.g. ```{"input": "hello world", "num_beams": 5}```)\n"""
        "```\n"
    ]


def test_default_tool_extractor():
    formatter = ToolFormatter(tool_format="default")
    result = """Action: test_tool\nAction Input: {"foo": "bar", "size": 10}\n"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]


def test_default_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="default")
    result = (
        """Action: test_tool\nAction Input: {"foo": "bar", "size": 10}\n"""
        """Action: another_tool\nAction Input: {"foo": "job", "size": 2}\n"""
    )
    assert formatter.extract(result) == [
        ("test_tool", """{"foo": "bar", "size": 10}"""),
        ("another_tool", """{"foo": "job", "size": 2}"""),
    ]


def test_glm4_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}"], tool_format="glm4")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == ["""tool_name\n{"foo": "bar", "size": 10}"""]


def test_glm4_tool_formatter():
    formatter = ToolFormatter(tool_format="glm4")
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
        "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具\n\n"
        f"## test_tool\n\n{json.dumps(TOOLS[0], indent=4, ensure_ascii=False)}\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
    ]


def test_glm4_tool_extractor():
    formatter = ToolFormatter(tool_format="glm4")
    result = """test_tool\n{"foo": "bar", "size": 10}\n"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]


def test_llama3_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|eot_id|>"], tool_format="llama3")
    tool_calls = json.dumps({"name": "tool_name", "arguments": {"foo": "bar", "size": 10}})
    assert formatter.apply(content=tool_calls) == [
        """{"name": "tool_name", "parameters": {"foo": "bar", "size": 10}}<|eot_id|>"""
    ]


def test_llama3_tool_formatter():
    formatter = ToolFormatter(tool_format="llama3")
    date = datetime.now().strftime("%d %b %Y")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        f"Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
        "You have access to the following functions. To call a function, please respond with JSON for a function call. "
        """Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. """
        f"Do not use variables.\n\n{json.dumps(wrapped_tool, indent=4, ensure_ascii=False)}\n\n"
    ]


def test_llama3_tool_extractor():
    formatter = ToolFormatter(tool_format="llama3")
    result = """{"name": "test_tool", "parameters": {"foo": "bar", "size": 10}}\n"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]


def test_mistral_function_formatter():
    formatter = FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", "</s>"], tool_format="mistral")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        "[TOOL_CALLS] " """[{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}]""",
        "</s>",
    ]


def test_mistral_multi_function_formatter():
    formatter = FunctionFormatter(slots=["[TOOL_CALLS] {{content}}", "</s>"], tool_format="mistral")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        "[TOOL_CALLS] "
        """[{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}, """
        """{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}]""",
        "</s>",
    ]


def test_mistral_tool_formatter():
    formatter = ToolFormatter(tool_format="mistral")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "[AVAILABLE_TOOLS] " + json.dumps([wrapped_tool], ensure_ascii=False) + "[/AVAILABLE_TOOLS]"
    ]


def test_mistral_tool_extractor():
    formatter = ToolFormatter(tool_format="mistral")
    result = """{"name": "test_tool", "arguments": {"foo": "bar", "size": 10}}"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]


def test_mistral_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="mistral")
    result = (
        """[{"name": "test_tool", "arguments": {"foo": "bar", "size": 10}}, """
        """{"name": "another_tool", "arguments": {"foo": "job", "size": 2}}]"""
    )
    assert formatter.extract(result) == [
        ("test_tool", """{"foo": "bar", "size": 10}"""),
        ("another_tool", """{"foo": "job", "size": 2}"""),
    ]


def test_qwen_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen")
    tool_calls = json.dumps(FUNCTION)
    assert formatter.apply(content=tool_calls) == [
        """<tool_call>\n{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}\n</tool_call><|im_end|>\n"""
    ]


def test_qwen_multi_function_formatter():
    formatter = FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen")
    tool_calls = json.dumps([FUNCTION] * 2)
    assert formatter.apply(content=tool_calls) == [
        """<tool_call>\n{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}\n</tool_call>\n"""
        """<tool_call>\n{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}\n</tool_call>"""
        "<|im_end|>\n"
    ]


def test_qwen_tool_formatter():
    formatter = ToolFormatter(tool_format="qwen")
    wrapped_tool = {"type": "function", "function": TOOLS[0]}
    assert formatter.apply(content=json.dumps(TOOLS)) == [
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        f"\n{json.dumps(wrapped_tool, ensure_ascii=False)}"
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
        """<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, """
        """"arguments": <args-json-object>}\n</tool_call>"""
    ]


def test_qwen_tool_extractor():
    formatter = ToolFormatter(tool_format="qwen")
    result = """<tool_call>\n{"name": "test_tool", "arguments": {"foo": "bar", "size": 10}}\n</tool_call>"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]


def test_qwen_multi_tool_extractor():
    formatter = ToolFormatter(tool_format="qwen")
    result = (
        """<tool_call>\n{"name": "test_tool", "arguments": {"foo": "bar", "size": 10}}\n</tool_call>\n"""
        """<tool_call>\n{"name": "another_tool", "arguments": {"foo": "job", "size": 2}}\n</tool_call>"""
    )
    assert formatter.extract(result) == [
        ("test_tool", """{"foo": "bar", "size": 10}"""),
        ("another_tool", """{"foo": "job", "size": 2}"""),
    ]
