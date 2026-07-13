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

"""Message <-> HF-template plumbing for rendering.

Pure, stateless helpers: convert v1 ``Message`` to HF chat-template format. No tokenization policy
decisions live here -- only mechanical conversion used by ``rendering.py``.
"""

import json

from ...utils.types import Message


_FALLBACK_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def _to_hf_messages(messages: list[Message]) -> list[dict]:
    """Convert v1 Message format to HF format for apply_chat_template."""
    hf_messages = []
    for message in messages:
        tool_calls: list[dict] = []
        reasoning_content = ""

        text = ""
        for content in message["content"]:
            if content["type"] == "text":
                text += content["value"]
            elif content["type"] == "reasoning":
                reasoning_content += content["value"]
            elif content["type"] == "tool_call":
                try:
                    tc = json.loads(content["value"])
                except json.JSONDecodeError as e:
                    raise ValueError(f"tool_call value is not valid JSON: {content['value']!r}") from e
                if not isinstance(tc, dict) or "name" not in tc or "arguments" not in tc:
                    raise ValueError(
                        f"tool_call must be a JSON object with 'name' and 'arguments' keys, got {tc!r}"
                    )
                tool_calls.append(
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                )
        hf_msg = {"role": message["role"], "content": text}

        if tool_calls:
            hf_msg["tool_calls"] = tool_calls
        if reasoning_content:
            hf_msg["reasoning_content"] = reasoning_content

        hf_messages.append(hf_msg)
    return hf_messages

