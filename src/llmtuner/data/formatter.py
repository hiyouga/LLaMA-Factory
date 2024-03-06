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
    "You have access to the following tools:\n{tool_text}"
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
    
    type_mapping = {
        "string": "str",
        "number": "float",  # Default to float; consider context for int usage
        "object": "Dict[str, Any]",  # Placeholder, detailed handling will vary
        "array": "List",
        "boolean": "bool",
        "null": "None",
    }
    
    for spec in specs:
        func_name = spec['name']
        description = spec.get('description', 'No description provided.')
        parameters = spec.get('parameters', {}).get('properties', {})
        required_params = spec.get('parameters', {}).get('required', [])
        
        func_args = []
        for param, details in parameters.items():
            json_type = details['type']
            default_value = details.get('default')
            python_type = type_mapping.get(json_type, "Any")  # Use Any as a fallback
            
            if json_type == 'array':
                items_type = type_mapping.get(details.get('items', {}).get('type', 'Any'), "Any")
                python_type = f"List[{items_type}]"
            elif json_type == 'object':
                # Simple representation; consider enhancing for nested objects
                python_type = "Dict[str, Any]"
            
            if 'enum' in details:
                python_type = 'str'  # Consider using Enum class
                
            arg_str = f"{param}: {python_type}"
            
            if required_params:
                if param not in required_params:
                    arg_str += " = None"  # Indicate optional by setting default to None
                
            func_args.append(arg_str)
                
        func_args_str = ", ".join(func_args) if func_args else ""
        
        docstring_lines = ['"""', description, '']
        for param, details in parameters.items():
            if required_params:
                required_text = "(Optional)" if param not in required_params else ""
            else:
                required_text = ""
            json_type = details['type']
            python_type = type_mapping.get(json_type, "Any")
            
            param_description = details.get('description', 'No description provided.')
            docstring_lines.append(f":param {param}: {param_description} {required_text}")
            docstring_lines.append(f":type {param}: {python_type}")
            
        docstring_lines.append('"""')
        docstring = "\n    ".join(docstring_lines)
        
        function_definition = f"def {func_name}({func_args_str}):\n    {docstring}\n"
        function_definitions.append(function_definition)
    
    return "\n".join(function_definitions)


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
    def apply(self, **kwargs) -> SLOTS: ...

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
        try:
            content = kwargs.pop("content")
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
        except Exception as e:
            print(e)
            return [""]

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        print("tool_format", self.tool_format)
        if self.tool_format == "default":
            return default_tool_extractor(content)
        elif self.tool_format == "rubra-fc-v1":
            return rubra_fc_v1_tool_formatter(content)
        else:
            raise NotImplementedError
