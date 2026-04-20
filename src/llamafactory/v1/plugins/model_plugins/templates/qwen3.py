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
import re

from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.types import Message, ModelInput, Processor, ToolCall
from ..rendering import RenderingPlugin


def _update_model_input(
    processor: Processor,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[int],
    temp_str: str,
    temp_weight: float,
) -> str:
    """Update model input with temporary string."""
    if not temp_str:
        return ""

    tokenizer = get_tokenizer(processor)
    temp_ids = tokenizer.encode(temp_str, add_special_tokens=False)
    input_ids.extend(temp_ids)
    loss_weights.extend([temp_weight] * len(temp_ids))
    if temp_weight > 1e-6:
        labels.extend(temp_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(temp_ids))

    return ""


def _concat_text_content(message: Message) -> str:
    """Concatenate text fields in a message."""
    message_text = ""
    for content in message["content"]:
        if content["type"] == "text":
            message_text += content["value"]
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return message_text


def _get_last_query_index(messages: list[Message]) -> int:
    """Find the last user query index, excluding wrapped tool responses."""
    last_query_index = len(messages) - 1
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message["role"] != "user":
            continue

        user_text = ""
        is_plain_text = True
        for content in message["content"]:
            if content["type"] != "text":
                is_plain_text = False
                break
            user_text += content["value"]

        if not is_plain_text:
            continue

        if not (user_text.startswith("<tool_response>") and user_text.endswith("</tool_response>")):
            last_query_index = idx
            break

    return last_query_index


def _split_assistant_content(message: Message) -> tuple[str, str, list[ToolCall]]:
    """Split assistant message into text, reasoning and tool calls."""
    text_content = ""
    reasoning_content = ""
    tool_calls: list[ToolCall] = []

    for content in message["content"]:
        if content["type"] == "text":
            text_content += content["value"]
        elif content["type"] == "reasoning":
            reasoning_content += content["value"]
        elif content["type"] == "tool_call":
            try:
                tool_call: ToolCall = json.loads(content["value"])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {content['value']}.")

            tool_calls.append(tool_call)
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return text_content, reasoning_content, tool_calls


@RenderingPlugin("qwen3").register("render_messages")
def render_qwen3_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Qwen3 template format.

    See https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-8B
    """
    input_ids, labels, loss_weights = [], [], []
    temp_str, temp_weight = "", 0.0
    if tools:
        temp_str += "<|im_start|>system\n"
        if messages[0]["role"] == "system":
            temp_str += _concat_text_content(messages[0]) + "\n\n"
            temp_weight = messages[0].get("loss_weight", 0.0)

        temp_str += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")

        if not isinstance(tools, list):
            tools = [tools]

        for tool in tools:
            temp_str += "\n" + json.dumps(tool, ensure_ascii=False)

        temp_str += (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
            '<function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
    elif messages[0]["role"] == "system":
        temp_str += "<|im_start|>system\n" + _concat_text_content(messages[0]) + "<|im_end|>\n"
        temp_weight = messages[0].get("loss_weight", 0.0)

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)
    last_query_index = _get_last_query_index(messages)

    for turn_idx, message in enumerate(messages):
        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|im_start|>" + message["role"] + "\n" + _concat_text_content(message) + "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            temp_str += "<|im_start|>" + message["role"] + "\n"

            text_content, reasoning_content, tool_calls = _split_assistant_content(message)
            if turn_idx > last_query_index and (turn_idx == len(messages) - 1 or reasoning_content):
                temp_str += "<think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + text_content.lstrip("\n")
            else:
                temp_str += text_content

            for tool_call_idx, tool_call in enumerate(tool_calls):
                if (tool_call_idx == 0 and text_content) or tool_call_idx > 0:
                    temp_str += "\n"

                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, ensure_ascii=False)

                temp_str += (
                    '<tool_call>\n{"name": "'
                    + tool_call["name"]
                    + '", "arguments": '
                    + arguments_str
                    + "}\n</tool_call>"
                )

            temp_str += "<|im_end|>\n"
            temp_weight = message.get("loss_weight", 1.0)
        elif message["role"] == "tool":
            if turn_idx == 0 or messages[turn_idx - 1]["role"] != "tool":
                temp_str += "<|im_start|>user"

            temp_str += "\n<tool_response>\n" + _concat_text_content(message) + "\n</tool_response>"
            if turn_idx == len(messages) - 1 or messages[turn_idx + 1]["role"] != "tool":
                temp_str += "<|im_end|>\n"

            temp_weight = message.get("loss_weight", 0.0)

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    if is_generate:
        temp_str += "<|im_start|>assistant\n"
        temp_weight = 0.0
        if enable_thinking is False:
            temp_str += "<think>\n\n</think>\n\n"

    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    attention_mask = [1] * len(input_ids)
    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("qwen3").register("parse_message")
def parse_qwen3_message(generated_text: str) -> Message:
    """Parse a message in the Qwen3 template format. Supports interleaved reasoning and tool calls.

    Args:
        generated_text (str): The generated text in the Qwen3 template format.

    Returns:
        Message: The parsed message.
    """
    pattern = re.compile(r"<(think|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
    content = []
    last_end = 0

    for match in pattern.finditer(generated_text):
        start, end = match.span()
        if start > last_end:
            text = generated_text[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tag_type = match.group(1)
        tag_value = match.group(2).strip()
        if tag_type == "think":
            content.append({"type": "reasoning", "value": tag_value.strip()})
        elif tag_type == "tool_call":
            try:
                json.loads(tag_value.strip())
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {tag_value.strip()}.")

            content.append({"type": "tool_call", "value": tag_value.strip()})

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)
