import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Set, Tuple, Union


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

TOOL_SYSTEM_PROMPT_RUBRA = (
    "You have access to the following tools:\n{tool_text}",
    "Use the following format if using a tool:\n[toolname1(arg1=value1, arg2=value2, ...), toolname2(arg1=value1, arg2=value2, ...)]"
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


def rubra_fc_v1_tool_formatter(specs: List[Dict[str, Any]]) -> str:
    function_definitions = []
    for spec in specs:
        # Extracting basic information
        func_name = spec['name']
        description = spec.get('description', 'No description provided.')
        parameters = spec.get('parameters', {}).get('properties', {})
        required_params = spec.get('parameters', {}).get('required', [])
        
        if isinstance(required_params, bool):  # Incorrect structure handling
            required_params = []  # Resetting to empty if it's mistakenly a boolean
        
        # Creating the function signature
        func_args = []
        for param, details in parameters.items():
            type_annotation = details['type']
            # Handling enum types for parameters
            if 'enum' in details:
                type_annotation = 'str'  # Using str type for enums, might be enhanced to use Literal types in future versions
            arg_str = f"{param}: {type_annotation}"
            if param not in required_params:
                arg_str += " = None"
            func_args.append(arg_str)
        func_args_str = ", ".join(func_args) if func_args else ""
        
        # Generating the docstring with dynamic parameter documentation, including required status
        docstring_lines = ['"""', description, '']
        for param, details in parameters.items():
            required_text = "Required" if param in required_params else "Optional"
            docstring_lines.append(f":param {param}: {details.get('description', 'No description provided.')} ({required_text})")
            docstring_lines.append(f":type {param}: {details['type']}")
        docstring_lines.append('"""')
        docstring = "\n    ".join(docstring_lines)
        
        # Generating the full function definition
        function_definition = f"""def {func_name}({func_args_str}) -> None:
    {docstring}
"""
        function_definitions.append(function_definition)
    to_return = TOOL_SYSTEM_PROMPT_RUBRA.format(
        tool_text="\n".join(function_definitions)
    )
    print(to_return)
    return to_return


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


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Literal["default"] = "default"

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        ...

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
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
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            if not len(tools):
                return [""]

            if self.tool_format == "default":
                return [default_tool_formatter(tools)]
            elif self.tool_format == "rubra-fc-v1":
                return [rubra_fc_v1_tool_formatter(tools)]
            else:
                raise NotImplementedError
        except Exception:
            return [""]

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        if self.tool_format == "default":
            return default_tool_extractor(content)
        elif self.tool_format == "rubra-fc-v1":
            return default_tool_extractor(content)
        else:
            raise NotImplementedError
