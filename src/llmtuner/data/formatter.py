import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union


JSON_FORMAT_PROMPT = (
    """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""
)


TOOL_SYSTEM_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format to answer the question:\n"
    "```\n"
    "Action: the action to take, should be one of [{tool_names}] if using a tool.\n"
    "Action Input: the input to the action{format_prompt}.\n"
    "```"
)


@dataclass
class StringFormatter:
    container: List[Union[str, Dict[str, str]]]

    def __call__(self, **kwargs) -> List[Union[str, Dict[str, str]]]:
        elements = []
        for elem in self.container:
            if isinstance(elem, str):
                for name, value in kwargs.items():
                    elem = elem.replace("{{" + name + "}}", value)
                elements.append(elem)
            elif isinstance(elem, (dict, set)):
                elements.append(elem)
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return elements


@dataclass
class FunctionFormatter:
    container: List[Union[str, Dict[str, str]]]

    def __call__(self, content: str) -> List[Union[str, Dict[str, str]]]:
        try:
            function = json.loads(content)
            name = function["name"]
            arguments = json.dumps(function["arguments"], ensure_ascii=False)
        except Exception:
            name, arguments = "", ""

        elements = []
        for elem in self.container:
            if isinstance(elem, str):
                elem = elem.replace("{{name}}", name)
                elem = elem.replace("{{arguments}}", arguments)
                elements.append(elem)
            elif isinstance(elem, (dict, set)):
                elements.append(elem)
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return elements


@dataclass
class ToolFormatter:
    type: Literal["default"]

    def _default(self, tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required = ", required" if name in tool["parameters"].get("required", []) else ""
                enum = ", should be one of [{}]".format(", ".join(param["enum"])) if param.get("enum", None) else ""
                param_text += "  - {name} ({type}{required}): {desc}{enum}\n".format(
                    name=name, type=param.get("type", ""), required=required, desc=param.get("description", ""), enum=enum
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return TOOL_SYSTEM_PROMPT.format(
            tool_text=tool_text,
            tool_names=", ".join(tool_names),
            format_prompt=JSON_FORMAT_PROMPT
        )

    def __call__(self, content: str) -> List[Union[str, Dict[str, str]]]:
        try:
            tools = json.loads(content)
            if not len(tools):
                return [""]

            if self.type == "default":
                return [self._default(tools)]
        except Exception:
            return [""]
