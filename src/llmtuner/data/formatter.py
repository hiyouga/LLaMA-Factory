import json
import re
import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


JSON_FORMAT_PROMPT = """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""


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
            required = (
                ", required" if name in tool["parameters"].get("required", []) else ""
            )
            enum = (
                ", should be one of [{}]".format(", ".join(param["enum"]))
                if param.get("enum", None)
                else ""
            )
            items = (
                ", where each item should be {}".format(param["items"].get("type", ""))
                if param.get("items")
                else ""
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
        tool_text=tool_text,
        tool_names=", ".join(tool_names),
        format_prompt=JSON_FORMAT_PROMPT,
    )


# def rubra_fc_v1_tool_formatter(specs: List[Dict[str, Any]]) -> str:
#     function_definitions = []

#     type_mapping = {
#         "string": "str",
#         "number": "float",  # Default to float; consider context for int usage
#         "object": "Dict[str, Any]",  # Placeholder, detailed handling will vary
#         "array": "List",
#         "boolean": "bool",
#         "null": "None",
#     }

#     for spec in specs:
#         spec = spec['function']
#         func_name = spec['name']
#         try:
#             description = spec.get('description', 'No description provided.')
#             parameters = spec.get('parameters', {}).get('properties', {})
#             required_params = spec.get('parameters', {}).get('required', [])
#         except Exception as e:
#             print(f"An error occurred: {e}")
#             print("spec:", spec)

#         func_args = []
#         for param, details in parameters.items():
#             json_type = details['type']
#             default_value = details.get('default')
#             python_type = type_mapping.get(json_type, "Any")  # Use Any as a fallback

#             if json_type == 'array':
#                 items_type = type_mapping.get(details.get('items', {}).get('type', 'Any'), "Any")
#                 python_type = f"List[{items_type}]"
#             elif json_type == 'object':
#                 # Simple representation; consider enhancing for nested objects
#                 python_type = "Dict[str, Any]"

#             if 'enum' in details:
#                 python_type = 'str'  # Consider using Enum class

#             arg_str = f"{param}: {python_type}"

#             if required_params:
#                 if param not in required_params:
#                     arg_str += " = None"  # Indicate optional by setting default to None

#             func_args.append(arg_str)

#         func_args_str = ", ".join(func_args) if func_args else ""

#         docstring_lines = ['"""', description, '']
#         for param, details in parameters.items():
#             if required_params:
#                 required_text = "(Optional)" if param not in required_params else ""
#             else:
#                 required_text = ""
#             json_type = details['type']
#             python_type = type_mapping.get(json_type, "Any")

#             param_description = ""
#             if "description" in details:
#                 param_description = details["description"]
#             if "enum" in details:
#                 quoted_enum = [f'"{val}"' for val in details["enum"]]
#                 values_text = "Only acceptable value is" if len(quoted_enum) == 1 else "Only acceptable values are"
#                 param_description += f" {values_text}: {' or '.join(quoted_enum)}"
#             if not param_description:
#                 param_description = "No description provided."
#             docstring_lines.append(f":param {param}: {param_description} {required_text}")
#             docstring_lines.append(f":type {param}: {python_type}")

#         docstring_lines.append('"""')
#         docstring = "\n    ".join(docstring_lines)

#         function_definition = f"def {func_name}({func_args_str}):\n    {docstring}\n"
#         function_definitions.append(function_definition)

#     res = TOOL_SYSTEM_PROMPT_RUBRA.format( tool_text="\n".join(function_definitions))
#     # print("res:")
#     # print(res)
#     return res

def rubra_fc_v1_tool_formatter(specs: List[Dict[str, Any]]) -> str:
    function_definitions = []

    type_mapping = {
        "string": "str",
        "number": "float",
        "object": "Dict[str, Any]",
        "array": "List",
        "boolean": "bool",
        "null": "None",
    }

    for spec in specs:
        try:
            if "function" not in spec:
                continue
            func_spec = spec["function"]

            func_name = func_spec.get("name", "unnamed_function")
            description = func_spec.get("description", "No description provided.") or "No description provided."

            prop_dict = {}
            required_params = []

            parameters = func_spec.get('parameters')
            if parameters and isinstance(parameters, dict) and 'properties' in parameters:
                temp_properties = parameters['properties']
                if isinstance(temp_properties, dict) and ('required' in temp_properties or 'optional' in temp_properties):
                    for status in ['required', 'optional']:
                        for item in temp_properties.get(status, []):
                            item['required'] = (status == 'required')
                            prop_dict[item['name']] = item
                elif isinstance(temp_properties, list):
                    for item in temp_properties:
                        if isinstance(item, dict) and 'name' in item:
                            prop_dict[item['name']] = item
                elif isinstance(temp_properties, dict):
                    prop_dict = temp_properties
                else:
                    print(f"Unexpected properties format in function {func_name}.")
            elif parameters is None:
                print(f"No parameters defined for function {func_name}.")
                continue  # Skip further processing for this function

            if 'required' in parameters and isinstance(parameters['required'], list):
                required_params = parameters['required']

            func_args = []
            for param, details in prop_dict.items():
                param_type = details.get("type", "Any")
                python_type = type_mapping.get(param_type.lower(), "Any") if isinstance(param_type, str) else "Any"
                is_required = details.get('required', False)

                arg_str = f"{param}: {python_type}"
                if not is_required:
                    arg_str += " = None"
                func_args.append(arg_str)

            func_args_str = ", ".join(func_args) if func_args else ""
            docstring_lines = ['"""', description]
            for param, details in prop_dict.items():
                required_text = "" if details.get('required', False) else "(Optional)"
                param_description = details.get("description", "No description provided.")
                docstring_lines.append(f":param {param}: {param_description} {required_text}")

            docstring_lines.append('"""')
            docstring = "\n    ".join(docstring_lines)
            function_definition = f"def {func_name}({func_args_str}):\n    {docstring}\n    pass\n"
            function_definitions.append(function_definition)
        except Exception as e:
            print(f"Error {e}")
            print("=========================")
            print(json.dumps(func_spec))

    res = TOOL_SYSTEM_PROMPT_RUBRA.format( tool_text="\n".join(function_definitions))
    return res


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


def parse_function_call(call):
    func_name, args_str = call.split('(', 1)
    args_str = args_str.rstrip(')')
    args_list = args_str.split(',')
    args_dict = {}
    for arg in args_list:
        key, value = arg.split('=')
        key = key.strip()
        value = value.strip()
        try:
            # Use ast.literal_eval to safely parse the string to its Python type
            parsed_value = ast.literal_eval(value)
        except ValueError as e:
            # If parsing fails, keep the original string. 
            # This might happen if the value is a string that's not quoted as a Python literal.
            print(f"Error parsing value {value}: {e}")
            parsed_value = value
        args_dict[key] = parsed_value
    return {"name": func_name.strip(), "arguments": args_dict}


def rubra_fc_v1_tool_extractor(content: str) -> Union[str, Tuple[str, str]]:
    regex = re.compile(r"<<functions>>\[(.*?)\]", re.DOTALL)
    matches = re.findall(regex, content)

    print("content:", content)

    if not matches:
        return content

    try:
        result_dicts = []
        for match in matches:
            # Splitting each function call from the match. We add ')' back because it was used as a delimiter
            function_calls = [f"{func})" for func in match.split('),') if func]
            print(function_calls)
            for function_call in function_calls:
                # Removing the trailing ')' that was added for the last function call
                if function_call.endswith(')'):
                    function_call = function_call[:-1]
                result_dict = parse_function_call(function_call.strip())
                result_dicts.append(result_dict)
            print(result_dicts)
        return json.dumps(result_dicts, ensure_ascii=False)
    except Exception as e:
        print(f"Exception {e}")
        return content


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[Literal["default"]] = None

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
                raise RuntimeError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(
                        type(slot)
                    )
                )

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
            raise ValueError(
                "Name and arguments placeholders are required in the function formatter."
            )

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
                slot = slot.replace("{{name}}", name).replace(
                    "{{arguments}}", arguments
                )
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(
                    "Input must be string, set[str] or dict[str, str], got {}".format(
                        type(slot)
                    )
                )

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
            elif self.tool_format == "rubra-fc-v1":
                return [rubra_fc_v1_tool_formatter(tools)]
            else:
                raise NotImplementedError
        except Exception as e:
            print(e)
            return [""]

    def extract(self, content: str) -> Union[str, Tuple[str, str]]:
        # print("tool_format", self.tool_format)
        if self.tool_format == "default":
            return default_tool_extractor(content)
        elif self.tool_format == "rubra-fc-v1":
            return rubra_fc_v1_tool_extractor(content)
        else:
            raise NotImplementedError
