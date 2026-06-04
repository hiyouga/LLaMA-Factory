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

from ....utils.types import Message
from ..rendering import RenderingPlugin


@RenderingPlugin("qwen3").register("parse_message")
def parse_qwen3_message(generated_text: str) -> Message:
    """Parse a message in the Qwen3 template format. Supports interleaved reasoning and tool calls.

    Args:
        generated_text: The generated text in the Qwen3 template format.

    Returns:
        The parsed message.
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
            content.append({"type": "reasoning", "value": tag_value})
        elif tag_type == "tool_call":
            json.loads(tag_value)
            content.append({"type": "tool_call", "value": tag_value})

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)

