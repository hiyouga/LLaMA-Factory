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

"""Rendering utils.

How to use:
renderer = Renderer(processor)
renderer.render_messages(messages: list[Message], tools: str | None) -> ModelInputs
renderer.parse_message(text: str) -> Message
renderer.process_samples(samples: list[Sample]) -> list[ModelInput]
"""

import json
import re

import numpy as np

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer, is_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample


_FALLBACK_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def _to_hf_messages(messages: list[Message], is_multimodal: bool = False) -> list[dict]:
    """Convert v1 Message format to HF format for apply_chat_template.

    Converts structured content types to their HF-native representations:
    - tool_call → message-level tool_calls field (HF function calling format)
    - reasoning → message-level reasoning_content field (HF reasoning format)
    - image/video/audio → multimodal content blocks
    """
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


def _detect_assistant_markers(template_caller) -> tuple[str, str]:
    r"""Detect the text markers that bracket assistant content in the rendered template.

    Returns (start_marker, end_marker) where:
    - start_marker: text immediately before assistant content (e.g., '<|im_start|>assistant\n')
    - end_marker: text immediately after assistant content (e.g., '<|im_end|>')
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
    return start_marker, end_marker


def _find_assistant_regions(text: str, start_marker: str, end_marker: str) -> list[tuple[int, int]]:
    """Find character ranges of assistant content (inclusive of end_marker) in rendered text.

    Returns list of (content_start_char, region_end_char) where:
    - content_start_char: first char of assistant content (after start_marker)
    - region_end_char: char after end_marker (exclusive bound)
    """
    regions = []
    pos = 0
    while True:
        start = text.find(start_marker, pos)
        if start == -1:
            break
        content_start = start + len(start_marker)
        end = text.find(end_marker, content_start)
        if end == -1:
            region_end = len(text)
        else:
            region_end = end + len(end_marker)
        regions.append((content_start, region_end))
        pos = region_end
    return regions


def _char_region_to_token_region(
    offsets: list[tuple[int, int]], content_start: int, region_end: int
) -> tuple[int, int]:
    """Map a character region to token indices using offset_mapping.

    Returns (token_start, token_end) as a half-open interval [start, end).
    """
    tok_start = None
    tok_end = None
    for i, (char_s, char_e) in enumerate(offsets):
        if char_s == char_e:
            continue
        if tok_start is None and char_e > content_start:
            tok_start = i
        if char_s < region_end:
            tok_end = i + 1
    return tok_start, tok_end


def _get_vision_token_ids(processor: Processor) -> set[int]:
    """Get known vision/media token IDs from the processor or its tokenizer."""
    vision_token_ids: set[int] = set()
    tokenizer = get_tokenizer(processor)
    for obj in (processor, tokenizer):
        for attr in ("image_token_id", "video_token_id", "audio_token_id"):
            tid = getattr(obj, attr, None)
            if tid is not None:
                vision_token_ids.add(tid)
    return vision_token_ids


def _build_text_to_expanded(
    text_ids: list[int], expanded_ids: list[int], vision_token_ids: set[int] | None = None
) -> list[int]:
    """Build mapping from text token index to expanded token index.

    Returns list of length len(text_ids)+1. mapping[i] = expanded position for text token i.
    mapping[len(text_ids)] = len(expanded_ids).

    When vision_token_ids is provided, uses them for precise expansion detection.
    Otherwise falls back to sequential scan heuristic.
    """
    mapping = [0] * (len(text_ids) + 1)
    e_ptr = 0

    for t_idx in range(len(text_ids)):
        mapping[t_idx] = e_ptr
        if vision_token_ids and text_ids[t_idx] in vision_token_ids:
            # Vision placeholder in text: skip all consecutive vision tokens in expanded
            while e_ptr < len(expanded_ids) and expanded_ids[e_ptr] in vision_token_ids:
                e_ptr += 1
        elif e_ptr < len(expanded_ids) and text_ids[t_idx] == expanded_ids[e_ptr]:
            e_ptr += 1
        else:
            # Fallback: scan expanded until we find the next text token
            if t_idx + 1 < len(text_ids):
                next_text_token = text_ids[t_idx + 1]
                while e_ptr < len(expanded_ids) and expanded_ids[e_ptr] != next_text_token:
                    e_ptr += 1
            else:
                e_ptr = len(expanded_ids)

    mapping[len(text_ids)] = e_ptr
    return mapping


def _render_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    assistant_start_marker: str | None = None,
    assistant_end_marker: str | None = None,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render messages using the model's own template with text-based boundary detection.

    Uses apply_chat_template to render the full conversation, then finds assistant
    content regions by searching for role markers in the rendered text. Character positions
    are mapped to token positions via offset_mapping.
    """
    tokenizer = get_tokenizer(processor)
    is_multimodal = not is_tokenizer(processor)
    has_media = is_multimodal and _count_media_in_messages(messages) != (0, 0)

    template_caller = processor if is_multimodal else tokenizer
    if not getattr(template_caller, "chat_template", None):
        template_caller.chat_template = _FALLBACK_CHATML_JINJA

    hf_messages = _to_hf_messages(messages, is_multimodal=is_multimodal)

    tools_parsed = None
    if tools:
        tools_parsed = json.loads(tools)
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]

    # 1. Render full text with model's own template
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    full_text = template_caller.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=is_generate, tools=tools_parsed, **template_kwargs
    )

    if has_media:
        # Multimodal path: call processor once for expansion
        images, videos = _extract_media_from_messages(messages)
        proc_kwargs = {"return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
        outputs = processor(text=full_text, **proc_kwargs)
        input_ids = outputs["input_ids"][0].tolist()

        # Get text-level tokenization for boundary detection
        text_encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        text_ids = text_encoding["input_ids"]
        text_offsets = text_encoding["offset_mapping"]
    else:
        # Text-only path
        encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        text_ids = input_ids
        text_offsets = encoding["offset_mapping"]
        outputs = None

    # 2. Find assistant content regions in rendered text
    if assistant_start_marker is None or assistant_end_marker is None:
        assistant_start_marker, assistant_end_marker = _detect_assistant_markers(template_caller)

    # Render without generation prompt for boundary detection (gen prompt is not assistant content)
    if is_generate:
        boundary_text = template_caller.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=False, tools=tools_parsed, **template_kwargs
        )
    else:
        boundary_text = full_text

    regions_char = _find_assistant_regions(boundary_text, assistant_start_marker, assistant_end_marker)

    # 3. Map char regions to text-token regions
    regions_text_tok = []
    for content_start, region_end in regions_char:
        tok_start, tok_end = _char_region_to_token_region(text_offsets, content_start, region_end)
        if tok_start is not None and tok_end is not None:
            regions_text_tok.append((tok_start, tok_end))

    # 4. Map text-token regions to expanded-token regions (multimodal expansion)
    if has_media and len(input_ids) != len(text_ids):
        vision_token_ids = _get_vision_token_ids(processor)
        exp_map = _build_text_to_expanded(text_ids, input_ids, vision_token_ids)
        regions_expanded = []
        for tok_start, tok_end in regions_text_tok:
            regions_expanded.append((exp_map[tok_start], exp_map[tok_end]))
    else:
        regions_expanded = regions_text_tok

    # 5. Build labels and loss_weights
    labels = [IGNORE_INDEX] * len(input_ids)
    loss_weights = [0.0] * len(input_ids)

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    for region_idx, (tok_start, tok_end) in enumerate(regions_expanded):
        if region_idx < len(assistant_messages):
            weight = assistant_messages[region_idx].get("loss_weight", 1.0)
        else:
            weight = 1.0

        for t in range(tok_start, min(tok_end, len(input_ids))):
            if weight > 1e-6:
                labels[t] = input_ids[t]
                loss_weights[t] = weight

    result = ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )

    if outputs is not None:
        if "pixel_values" in outputs:
            result["pixel_values"] = outputs["pixel_values"]
        if "image_grid_thw" in outputs:
            result["image_grid_thw"] = outputs["image_grid_thw"]
        if "pixel_values_videos" in outputs:
            result["pixel_values_videos"] = outputs["pixel_values_videos"]
        if "video_grid_thw" in outputs:
            result["video_grid_thw"] = outputs["video_grid_thw"]
        if "mm_token_type_ids" in outputs:
            result["mm_token_type_ids"] = outputs["mm_token_type_ids"][0].tolist()

    return result


def _parse_message(generated_text: str) -> Message:
    """Parse generated text to structured Message.

    Handles common patterns: <think>/<thinking> for reasoning, <tool_call> for tool calls.
    """
    pattern = re.compile(r"<(think|thinking|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
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
        if tag_type in ("think", "thinking"):
            content.append({"type": "reasoning", "value": tag_value})
        elif tag_type == "tool_call":
            json.loads(tag_value)
            content.append({"type": "tool_call", "value": tag_value})

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    if not content:
        content.append({"type": "text", "value": generated_text})

    return Message(role="assistant", content=content)


class Renderer:
    def __init__(self, processor: Processor):
        self.processor = processor
        self._assistant_start_marker = None
        self._assistant_end_marker = None

        template_caller = processor if not is_tokenizer(processor) else get_tokenizer(processor)
        if getattr(template_caller, "chat_template", None):
            try:
                self._assistant_start_marker, self._assistant_end_marker = _detect_assistant_markers(template_caller)
            except (ValueError, Exception):
                pass

    def render_messages(
        self,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        """Render messages to model input using apply_chat_template.

        Args:
            messages: The messages to render.
            tools: JSON string of tool definitions.
            is_generate: Whether to render for generation (adds generation prompt).
            enable_thinking: Whether to enable thinking mode (passed as template kwarg).

        Returns:
            ModelInput with input_ids, attention_mask, labels, and loss_weights.
        """
        return _render_messages(
            self.processor,
            messages,
            tools,
            is_generate,
            self._assistant_start_marker,
            self._assistant_end_marker,
            enable_thinking=enable_thinking,
        )

    def parse_message(self, generated_text: str) -> Message:
        """Parse generated text into a structured Message.

        Args:
            generated_text: The raw generated text from the model.

        Returns:
            Parsed Message with typed content blocks.
        """
        return _parse_message(generated_text)

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]:
        """Process samples to model input.

        Args:
            samples: The samples to process.

        Returns:
            List of processed model inputs.
        """
        model_inputs = []
        for sample in samples:
            if "messages" in sample:
                model_input = self.render_messages(sample["messages"], sample.get("tools"))
                if "position_ids" not in model_input:
                    model_input["position_ids"] = list(range(1, len(model_input["input_ids"]) + 1))
            elif "chosen_messages" in sample and "rejected_messages" in sample:
                chosen_input = self.render_messages(sample["chosen_messages"], sample.get("tools"))
                rejected_input = self.render_messages(sample["rejected_messages"], sample.get("tools"))
                chosen_input["token_type_ids"] = [1] * len(chosen_input["input_ids"])
                rejected_input["token_type_ids"] = [2] * len(rejected_input["input_ids"])
                model_input = ModelInput(
                    input_ids=chosen_input["input_ids"] + rejected_input["input_ids"],
                    attention_mask=chosen_input["attention_mask"] + rejected_input["attention_mask"],
                    labels=chosen_input["labels"] + rejected_input["labels"],
                    loss_weights=chosen_input["loss_weights"] + rejected_input["loss_weights"],
                    token_type_ids=chosen_input["token_type_ids"] + rejected_input["token_type_ids"],
                )
                if "position_ids" in chosen_input:
                    model_input["position_ids"] = np.concatenate(
                        [chosen_input["position_ids"], rejected_input["position_ids"]], axis=-1
                    )
            else:
                raise ValueError("No valid messages or chosen_messages/rejected_messages found in sample.")

            if "extra_info" in sample:
                model_input["extra_info"] = sample["extra_info"]

            if "_dataset_name" in sample:
                model_input["_dataset_name"] = sample["_dataset_name"]

            model_inputs.append(model_input)

        return model_inputs
