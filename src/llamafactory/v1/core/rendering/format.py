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

Pure, stateless helpers: convert v1 ``Message`` to HF chat-template format, extract/count media,
and search token-id subsequences. No tokenization policy decisions live here -- only mechanical
conversion used by ``rendering.py``. Assistant role markers are declared per model in
``markers.py`` rather than probed.
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


def _extract_media_from_messages(messages: list[Message]) -> tuple[list, list, list]:
    """Extract image, video and audio paths/values from messages in order."""
    images, videos, audios = [], [], []
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                images.append(content["value"])
            elif content["type"] == "video_url":
                videos.append(content["value"])
            elif content["type"] == "audio_url":
                audios.append(content["value"])
    return images, videos, audios


def _count_media_in_messages(messages: list[Message]) -> tuple[int, int, int]:
    """Count total images, videos and audios in messages."""
    n_images, n_videos, n_audios = 0, 0, 0
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image_url":
                n_images += 1
            elif content["type"] == "video_url":
                n_videos += 1
            elif content["type"] == "audio_url":
                n_audios += 1
    return n_images, n_videos, n_audios


def _load_audios(values: list, sampling_rate: int) -> list:
    """Load audio inputs into mono waveforms resampled to ``sampling_rate``.

    Audio processors (e.g. Qwen2-Audio's Whisper feature extractor) expect raw waveforms at the
    model's sampling rate, not file paths -- so unlike images/videos we decode here, before the
    processor call. ``np.ndarray`` values are passed through untouched; str/path-like values are
    read and resampled. Mirrors v0 ``mm_plugin._regularize_audios``.
    """
    import numpy as np
    import torchaudio

    results = []
    for value in values:
        if isinstance(value, np.ndarray):
            results.append(value)
            continue

        waveform, sr = torchaudio.load(value)
        if waveform.shape[0] > 1:  # downmix to mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
        results.append(waveform.squeeze(0).numpy())
    return results


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """First index >= ``start`` where ``needle`` occurs as a contiguous subsequence, else -1."""
    if not needle:
        return -1
    first = needle[0]
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return -1
