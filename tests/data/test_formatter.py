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

from llamafactory.data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter


def test_empty_formatter():
    formatter = EmptyFormatter(slots=["\n"])
    assert formatter.apply() == ["\n"]


def test_string_formatter():
    formatter = StringFormatter(slots=["<s>", "Human: {{content}}\nAssistant:"])
    assert formatter.apply(content="Hi") == ["<s>", "Human: Hi\nAssistant:"]


def test_function_formatter():
    formatter = FunctionFormatter(slots=[], tool_format="default")
    tool_calls = json.dumps({"name": "tool_name", "arguments": {"foo": "bar", "size": 10}})
    assert formatter.apply(content=tool_calls) == [
        """Action: tool_name\nAction Input: {\"foo\": \"bar\", \"size\": 10}\n"""
    ]


def test_multi_function_formatter():
    formatter = FunctionFormatter(slots=[], tool_format="default")
    tool_calls = json.dumps([{"name": "tool_name", "arguments": {"foo": "bar", "size": 10}}] * 2)
    assert formatter.apply(content=tool_calls) == [
        """Action: tool_name\nAction Input: {\"foo\": \"bar\", \"size\": 10}\n""",
        """Action: tool_name\nAction Input: {\"foo\": \"bar\", \"size\": 10}\n""",
    ]


def test_default_tool_formatter():
    formatter = ToolFormatter(tool_format="default")
    tools = [
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
    assert formatter.apply(content=json.dumps(tools)) == [
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


def test_glm4_tool_formatter():
    formatter = ToolFormatter(tool_format="glm4")
    tools = [
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
    assert formatter.apply(content=json.dumps(tools)) == [
        "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
        "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具\n\n"
        "## test_tool\n\n{}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(json.dumps(tools[0], indent=4))
    ]


def test_glm4_tool_extractor():
    formatter = ToolFormatter(tool_format="glm4")
    result = """test_tool\n{"foo": "bar", "size": 10}\n"""
    assert formatter.extract(result) == [("test_tool", """{"foo": "bar", "size": 10}""")]
