import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


JSON_FORMAT_PROMPT = (
    """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""
)


TOOL_SYSTEM_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}]).\n"
    "Action Input: the input to the tool{format_prompt}.\n"
    "```\n"
)


GLM4_TOOL_SUFFIX_PROMPT = (
    "在调用上述函数时，请使用 Json 格式表示调用的参数。"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持，"
    "{tool_text}"
    
)


def default_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    tool_names = []
    for tool in tools:
        param_text = ""
        for name, param in tool["parameters"]["properties"].items():
            required = ", required" if name in tool["parameters"].get("required", []) else ""
            enum = ", should be one of [{}]".format(", ".join(param["enum"])) if param.get("enum", None) else ""
            items = (
                ", where each item should be {}".format(param["items"].get("type", "")) if param.get("items") else ""
            )
            param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                name=name,
                type=param.get("type", ""),
                required=required,
                desc=param.get("description", ""),
                enum=enum,
                items=items,
            )

        tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
            name=tool["name"], desc=tool.get("description", ""), args=param_text
        )
        tool_names.append(tool["name"])

    return TOOL_SYSTEM_PROMPT.format(
        tool_text=tool_text, tool_names=", ".join(tool_names), format_prompt=JSON_FORMAT_PROMPT
    )


def glm4_tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    for tool in tools:
        tool_name = tool["name"]
        tool_text += f"\n\n## {tool_name}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}\n{GLM4_TOOL_SUFFIX_PROMPT}"
    return GLM4_TOOL_PROMPT.format(tool_text=tool_text)
 

def default_tool_extractor(content: str) -> Union[str, Tuple[str, str]]:
    regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+).*?Action Input:\s*(.*)", re.DOTALL)
    action_match = re.search(regex, content)
    if not action_match:
        return content

    tool_name = action_match.group(1).strip()
    tool_input = action_match.group(2).strip().strip('"').strip("```")
    try:
        arguments = json.loads(tool_input)
    except json.JSONDecodeError:
        return content

    return tool_name, json.dumps(arguments, ensure_ascii=False)


def glm4_tool_extractor(content: str) -> Union[str, Tuple[str, str]]:
    lines = content.strip().split("\n")
    if len(lines) != 2:
        return content
    tool_name = lines[0].strip()
    tool_input = lines[1].strip()
    try:
        arguments = json.loads(tool_input)
    except json.JSONDecodeError:
        return content
    return tool_name, json.dumps(arguments, ensure_ascii=False)

    

@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[Literal["default", "glm4"]] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS: ...

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements


@dataclass
class FunctionFormatter(Formatter):
    def __post_init__(self):
        has_name, has_args = False, False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if "{{name}}" in slot:
                has_name = True
            if "{{arguments}}" in slot:
                has_args = True

        if not has_name or not has_args:
            raise ValueError("Name and arguments placeholders are required in the function formatter.")

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            function = json.loads(content)
            name = function["name"]
            arguments = json.dumps(function["arguments"], ensure_ascii=False)
        except Exception:
            name, arguments = "", ""

        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                slot = slot.replace("{{name}}", name).replace("{{arguments}}", arguments)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements


@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        if self.tool_format is None:
            raise ValueError("Tool format was not found.")

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            if not len(tools):
                return [""]

            if self.tool_format == "default":
                return [default_tool_formatter(tools)]
            elif self.tool_format == "glm4":
                """
                '[gMASK]<sop><|system|>\n你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n## get_current_weather\n\n{\n    "name": "get_current_weather",\n    "description": "Get the current weather",\n    "parameters": {\n        "type": "object",\n        "properties": {\n            "location": {\n                "type": "string",\n                "description": "The city and state, e.g. San Francisco, CA"\n            },\n            "format": {\n                "type": "string",\n                "enum": [\n                    "celsius",\n                    "fahrenheit"\n                ],\n                "description": "The temperature unit to use. Infer this from the users location."\n            }\n        },\n        "required": [\n            "location",\n            "format"\n        ]\n    }\n}\n在调用上述函数时，请使用 Json 格式表示调用的参数。<|user|>\nWhat\'s the weather like in San Francisco, Tokyo, and Paris? use celsius<|assistant|>'
                """
                return [glm4_tool_formatter(tools)]
            else:
                raise NotImplementedError
        except Exception:
            return [""]

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        if self.tool_format == "default":
            return default_tool_extractor(content)
        elif self.tool_format == "glm4":
            return glm4_tool_extractor(content)
        else:
            raise NotImplementedError
