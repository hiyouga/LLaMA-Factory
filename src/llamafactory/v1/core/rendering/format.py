# Copyright 2026 the LlamaFactory team.
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

Pure, stateless helpers: convert v1 ``Message`` to HF chat-template format, extract/count media,
search token-id subsequences, and probe the template to detect the assistant role markers (both
text and token-id forms). No tokenization policy decisions live here -- only mechanical
conversion and detection used by ``rendering.py``.
"""

import json

from ...utils.helper import get_tokenizer
from ...utils.types import Message


_FALLBACK_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def _to_hf_messages(messages: list[Message], is_multimodal: bool = False) -> list[dict]:
    """Convert v1 Message format to HF format for apply_chat_template."""
    hf_messages = []
    for message in messages:
        tool_calls: list[dict] = []
        reasoning_content = ""

        if is_multimodal:
            hf_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    hf_content.append({"type": "text", "text": content["value"]})
                elif content["type"] == "reasoning":
                    reasoning_content += content["value"]
                elif content["type"] == "tool_call":
                    tc = json.loads(content["value"])
                    tool_calls.append(
                        {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    )
                elif content["type"] == "image_url":
                    hf_content.append({"type": "image", "image": content["value"]})
                elif content["type"] == "video_url":
                    hf_content.append({"type": "video", "video": content["value"]})
                elif content["type"] == "audio_url":
                    hf_content.append({"type": "audio", "audio": content["value"]})
            hf_msg = {"role": message["role"], "content": hf_content}
        else:
            text = ""
            for content in message["content"]:
                if content["type"] == "text":
                    text += content["value"]
                elif content["type"] == "reasoning":
                    reasoning_content += content["value"]
                elif content["type"] == "tool_call":
                    tc = json.loads(content["value"])
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


def _extract_media_from_messages(messages: list[Message]) -> tuple[list, list]:
    """Extract image paths and video paths from messages in order."""
    images, videos = [], []
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images.append(content["value"])
            elif content["type"] == "video_url":
                videos.append(content["value"])
    return images, videos


def _count_media_in_messages(messages: list[Message]) -> tuple[int, int]:
    """Count total images and videos in messages."""
    n_images, n_videos = 0, 0
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                n_images += 1
            elif content["type"] == "video_url":
                n_videos += 1
    return n_images, n_videos


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """First index >= ``start`` where ``needle`` occurs as a contiguous subsequence, else -1."""
    if not needle:
        return -1
    first = needle[0]
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _rfind_subseq(haystack: list[int], needle: list[int], end: int | None = None) -> int:
    """Last index where ``needle`` occurs as a contiguous subsequence within ``haystack[:end]``."""
    if not needle:
        return -1
    hi = (len(haystack) if end is None else end) - len(needle)
    for i in range(hi, -1, -1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _detect_assistant_markers(template_caller) -> tuple[str, str]:
    r"""Detect the text markers that bracket assistant content in the rendered template.

    Returns (start_marker, end_marker), e.g. ('<|im_start|>assistant\n', '<|im_end|>').
    """
    CONTENT_A = "AABBCC_PROBE_CONTENT_1_XXYYZZ"
    CONTENT_B = "AABBCC_PROBE_CONTENT_2_XXYYZZ"
    test_msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": CONTENT_A},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": CONTENT_B},
    ]
    rendered = template_caller.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=False)

    pos_a = rendered.find(CONTENT_A)
    pos_b = rendered.find(CONTENT_B)
    if pos_a == -1 or pos_b == -1:
        raise ValueError(
            "Cannot detect assistant role markers: probe content not found in rendered template. "
            "The model's chat_template may not render assistant content literally."
        )

    end_text = rendered[pos_b + len(CONTENT_B) :]
    end_marker = end_text.split("\n")[0]
    if not end_marker:
        end_marker = end_text.rstrip()

    prefix = rendered[:pos_a]
    last_end = prefix.rfind(end_marker)
    if last_end != -1:
        start_marker = prefix[last_end + len(end_marker) :]
    else:
        start_marker = prefix

    start_marker = start_marker.lstrip("\n")
    if not start_marker or not end_marker:
        raise ValueError("Detected empty assistant start or end marker.")
    return start_marker, end_marker


def _detect_assistant_marker_ids(template_caller) -> tuple[list[int], list[int]]:
    r"""Token-id form of the assistant start/end markers, for scanning the expanded id stream.

    Derived from a probe so the ids match in-context tokenization (not standalone encoding).
    The start marker is taken from the FIRST probe assistant turn, because some templates inject
    a ``<think>`` wrapper only on the LAST turn; the marker proper is the role-opening run that
    begins at the first special token (e.g. ``<|im_start|>assistant\n``). The end marker is the
    run up to and including the first special token after the content (e.g. ``<|im_end|>``).
    """
    tokenizer = get_tokenizer(template_caller)
    content_a = "AABBCC_PROBE_CONTENT_1_XXYYZZ"
    content_b = "AABBCC_PROBE_CONTENT_2_XXYYZZ"
    test_msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": content_a},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": content_b},
    ]
    rendered = template_caller.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(content_a, add_special_tokens=False)["input_ids"]
    pa = _find_subseq(ids, a_ids)
    if pa == -1:
        raise ValueError("Cannot detect assistant marker ids: probe content not found.")

    special_ids = {tid for tid, t in tokenizer.added_tokens_decoder.items() if getattr(t, "special", False)}

    # End marker: tokens right after the assistant content up to and including the first special
    # (turn-terminator) token.
    end_ids: list[int] = []
    for tid in ids[pa + len(a_ids) :]:
        end_ids.append(tid)
        if tid in special_ids:
            break
    if not end_ids or end_ids[-1] not in special_ids:
        raise ValueError("Cannot detect assistant end marker id.")

    # Start marker: between the previous end-marker occurrence and the content, trimmed to begin
    # at the turn-opening special token (drops the previous turn's trailing newline).
    prev = _rfind_subseq(ids, end_ids, pa)
    region = ids[prev + len(end_ids) : pa] if prev != -1 else ids[:pa]
    while region and region[0] not in special_ids:
        region = region[1:]
    if not region:
        raise ValueError("Cannot detect assistant start marker id.")
    return region, end_ids
