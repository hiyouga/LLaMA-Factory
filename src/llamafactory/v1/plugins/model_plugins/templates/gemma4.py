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

from ....data.tool_utils import FunctionCall, Gemma4ToolUtils
from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.types import Message, ModelInput, Processor, ToolCall
from ..rendering import RenderingPlugin

_GEMMA4_THOUGHT_START = "<|channel>thought\n"
_GEMMA4_THOUGHT_END = "<channel|>"
_GEMMA4_TOOL_START = "<|tool>"
_GEMMA4_TOOL_END = "<tool|>"


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


@RenderingPlugin("gemma4").register("render_messages")
def render_gemma4_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages in the Gemma4 template format.

    See https://huggingface.co/google/gemma-4-27b-it
    """
    input_ids, labels, loss_weights = [], [], []

    # Gemma4 requires a BOS token prefix.
    tokenizer = get_tokenizer(processor)
    if tokenizer.bos_token_id is not None:
        input_ids.append(tokenizer.bos_token_id)
        labels.append(IGNORE_INDEX)
        loss_weights.append(0.0)

    temp_str, temp_weight = "", 0.0

    # Collect system content and tool declarations for the system turn.
    system_content = ""
    system_weight = 0.0
    if messages[0]["role"] == "system":
        system_content = _concat_text_content(messages[0])
        system_weight = messages[0].get("loss_weight", 0.0)

    tool_declarations = ""
    if tools:
        try:
            tool_list = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")
        if not isinstance(tool_list, list):
            tool_list = [tool_list]
        tool_declarations = Gemma4ToolUtils.tool_formatter(tool_list)

    # Emit the system turn (always required for Gemma4 thinking via <|think|>).
    system_text = system_content or "You are a helpful assistant."
    temp_str += "<|turn>system\n<|think|>" + system_text
    if tool_declarations:
        temp_str += "\n" + tool_declarations
    temp_str += "<turn|>\n"
    temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, system_weight)

    last_query_index = _get_last_query_index(messages)

    for turn_idx, message in enumerate(messages):
        if turn_idx == 0 and message["role"] == "system":
            continue  # already handled above

        if message["role"] == "user" or (message["role"] == "system" and turn_idx != 0):
            temp_str += "<|turn>user\n" + _concat_text_content(message) + "<turn|>\n<|turn>model\n"
            temp_weight = message.get("loss_weight", 0.0)
        elif message["role"] == "assistant":
            text_content, reasoning_content, tool_calls = _split_assistant_content(message)

            if turn_idx > last_query_index and (turn_idx == len(messages) - 1 or reasoning_content):
                temp_str += (
                    _GEMMA4_THOUGHT_START + reasoning_content.strip("\n") + _GEMMA4_THOUGHT_END
                )

            temp_str += text_content.lstrip("\n")

            for tool_call_idx, tool_call in enumerate(tool_calls):
                if (tool_call_idx == 0 and text_content) or tool_call_idx > 0:
                    temp_str += "\n"

                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    arguments_str = arguments
                else:
                    arguments_str = json.dumps(arguments, ensure_ascii=False)

                fc = [FunctionCall(name=tool_call["name"], arguments=arguments_str)]
                temp_str += _GEMMA4_TOOL_START + Gemma4ToolUtils.function_formatter(fc) + _GEMMA4_TOOL_END

            temp_str += "<turn|>\n"
            temp_weight = message.get("loss_weight", 1.0)
        elif message["role"] == "tool":
            temp_str += "<|turn>tool\n" + _concat_text_content(message) + "<turn|>\n<|turn>model\n"
            temp_weight = message.get("loss_weight", 0.0)

        temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, temp_weight)

    if is_generate:
        # The user turn already ends with <|turn>model\n; inject an empty thinking
        # block to suppress reasoning when explicitly disabled.
        if enable_thinking is False:
            temp_str += _GEMMA4_THOUGHT_START + "\n" + _GEMMA4_THOUGHT_END
            temp_str = _update_model_input(processor, input_ids, labels, loss_weights, temp_str, 0.0)

    attention_mask = [1] * len(input_ids)
    return ModelInput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("gemma4").register("parse_message")
def parse_gemma4_message(generated_text: str) -> Message:
    """Parse a message in the Gemma4 template format.

    Handles interleaved reasoning blocks and tool calls:
      - Reasoning: <|channel>thought\\n...\\n<channel|>
      - Tool calls: <|tool><|tool_call>call:NAME{...}<tool_call|><tool|>
    """
    think_pattern = re.compile(r"<\|channel>thought\n(.*?)<channel\|>", re.DOTALL)
    tool_block_pattern = re.compile(r"<\|tool>(.*?)<tool\|>", re.DOTALL)

    content: list[dict] = []

    # Extract and remove the reasoning block (always at the start, if present).
    think_match = think_pattern.match(generated_text.lstrip())
    remaining = generated_text
    if think_match:
        reasoning = think_match.group(1).strip()
        if reasoning:
            content.append({"type": "reasoning", "value": reasoning})
        remaining = generated_text[think_match.end():]

    # Parse the remaining text for tool calls and plain text.
    last_end = 0
    for match in tool_block_pattern.finditer(remaining):
        start, end = match.span()
        if start > last_end:
            text = remaining[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tool_content = match.group(1)
        extracted = Gemma4ToolUtils.tool_extractor(tool_content)
        if isinstance(extracted, list):
            for fc in extracted:
                content.append(
                    {
                        "type": "tool_call",
                        "value": json.dumps({"name": fc.name, "arguments": fc.arguments}, ensure_ascii=False),
                    }
                )
        elif isinstance(extracted, str) and extracted.strip():
            content.append({"type": "text", "value": extracted.strip()})

        last_end = end

    if last_end < len(remaining):
        text = remaining[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)
