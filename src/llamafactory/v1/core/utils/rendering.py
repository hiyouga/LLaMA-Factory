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

Per-template steps can be customized by registering an override via ``RenderingPlugin``
(see ``plugins/model_plugins/rendering.py``) and constructing ``Renderer(processor, name=...)``.
"""

import json
import re

import numpy as np
import torch

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer, is_tokenizer
from ...utils.types import BatchInput, Message, ModelInput, Processor, Sample, Tensor


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


def _special_token_strings(tokenizer) -> list[str]:
    """Strings the tokenizer encodes to a reserved/special id.

    Such strings must be neutralized if they appear literally in user text. Derived from
    ``added_tokens_decoder`` so it covers every control token of the model (``<|im_start|>``,
    ``<|image_pad|>``, ``<tts_pad>`` ...), not only ``<|...|>``-shaped ones. Sorted longest-first
    so nested matches escape correctly.
    """
    specials = [str(t) for t in tokenizer.added_tokens_decoder.values() if getattr(t, "special", False)]
    return sorted((s for s in specials if len(s) >= 2), key=len, reverse=True)


def _escape_special(text: str, specials: list[str], special_ids: set[int], tokenizer) -> str:
    """Break any special-token string in user text by inserting U+200B after its first char.

    No-op (no tokenization cost) when the text contains no special-token string. When it does,
    self-validate that the result no longer encodes to a special id -- some normalizers strip
    zero-width chars and would resurrect the collision -- and raise if it does.
    """
    if not any(sp in text for sp in specials):
        return text
    out = text
    for sp in specials:
        if sp in out:
            # Insert a zero-width space (U+200B) after the first char to break the exact
            # special-token string match while keeping the text visually/semantically intact.
            out = out.replace(sp, sp[0] + "\u200b" + sp[1:])
    if special_ids.intersection(tokenizer(out, add_special_tokens=False)["input_ids"]):
        raise ValueError(
            "special-token escape failed: the tokenizer normalized away the break char; "
            "user text contains a literal control token that cannot be safely neutralized."
        )
    return out


def _escape_special_in_messages(
    messages: list[Message], specials: list[str], special_ids: set[int], tokenizer
) -> list[Message]:
    """Return messages with special-token strings neutralized in user-controlled literal text.

    Covers ``text``/``reasoning`` block values and string values inside ``tool_call`` arguments.
    """
    if not specials:
        return messages
    escaped: list[Message] = []
    for message in messages:
        new_content = []
        for content in message["content"]:
            if content["type"] in ("text", "reasoning"):
                new_content.append(
                    {**content, "value": _escape_special(content["value"], specials, special_ids, tokenizer)}
                )
            elif content["type"] == "tool_call":
                try:
                    tc = json.loads(content["value"])
                    args = tc.get("arguments")
                    if isinstance(args, dict):
                        tc["arguments"] = {
                            k: (_escape_special(v, specials, special_ids, tokenizer) if isinstance(v, str) else v)
                            for k, v in args.items()
                        }
                    new_content.append({**content, "value": json.dumps(tc)})
                except (json.JSONDecodeError, TypeError):
                    new_content.append(content)
            else:
                new_content.append(content)
        escaped.append({**message, "content": new_content})
    return escaped


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


def _check_placeholder_counts(processor: Processor, full_text: str, n_images: int, n_videos: int) -> None:
    """Guard: every media placeholder in the rendered text must originate from a media block.

    After escaping, literal placeholder strings in user text are broken, so any remaining
    placeholder must come from an image_url/video_url block. A mismatch means the data encodes
    media some other way (e.g. inline tags) -- raise rather than crash inside the processor.
    """
    tokenizer = get_tokenizer(processor)
    for attr, count, kind in (("image_token_id", n_images, "image"), ("video_token_id", n_videos, "video")):
        tid = getattr(processor, attr, None)
        if tid is None:
            tid = getattr(tokenizer, attr, None)
        if tid is None:
            continue
        placeholder = tokenizer.convert_ids_to_tokens(tid)
        seen = full_text.count(placeholder)
        if seen != count:
            raise ValueError(
                f"{kind} placeholder count ({seen}) != number of {kind} blocks ({count}); "
                "media must be provided via image_url/video_url content blocks."
            )


def _label_assistant_regions(
    input_ids: list[int], start_ids: list[int], end_ids: list[int], assistant_messages: list[Message]
) -> tuple[list[int], list[float], int]:
    """Label assistant content by scanning for the marker token-id subsequences.

    Each properly-closed ``start_ids ... end_ids`` span is one assistant region; the closing
    ``end_ids`` is included (parity with the previous char-based renderer). Regions map in order
    to assistant messages and take that message's ``loss_weight`` (weight 0 leaves the region
    unlabeled). An unterminated trailing start marker (the generation prompt under
    ``is_generate``) yields no region.
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    loss_weights = [0.0] * len(input_ids)
    regions: list[tuple[int, int]] = []
    n = len(input_ids)
    i = 0
    while i <= n - len(start_ids):
        if input_ids[i : i + len(start_ids)] == start_ids:
            content_start = i + len(start_ids)
            end = _find_subseq(input_ids, end_ids, content_start)
            if end == -1:
                break  # unterminated (generation prompt) -> not labeled
            region_end = end + len(end_ids)
            regions.append((content_start, region_end))
            i = region_end
        else:
            i += 1

    for idx, (start, end) in enumerate(regions):
        weight = assistant_messages[idx].get("loss_weight", 1.0) if idx < len(assistant_messages) else 1.0
        if weight > 1e-6:
            for t in range(start, min(end, n)):
                labels[t] = input_ids[t]
                loss_weights[t] = weight
    return labels, loss_weights, len(regions)


def _verify_render(regions_count: int, assistant_messages: list[Message]) -> None:
    """Cheap structural invariant: exactly one region per assistant message.

    A mismatch signals marker injection that slipped through, a tools-text false marker, or
    marker-detection failure -- fail loud rather than train on corrupted labels.
    """
    n_assistant = len(assistant_messages)
    if regions_count != n_assistant:
        raise ValueError(
            f"assistant region count ({regions_count}) != assistant messages ({n_assistant}); "
            "possible marker collision or detection failure."
        )


def _render_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    assistant_start_marker: str | None = None,
    assistant_end_marker: str | None = None,
    assistant_start_ids: list[int] | None = None,
    assistant_end_ids: list[int] | None = None,
    enable_thinking: bool = False,
) -> ModelInput:
    r"""Render messages using the model's own template, with provenance-preserving labeling.

    User-controlled literal text (``text``/``reasoning`` values, ``tool_call`` arg values, and
    ``tools`` definitions) is escaped first so any control token written literally by the user
    is neutralized -- this is a no-op for normal data. The escaped conversation is rendered and
    run through the processor/tokenizer, whose output (``input_ids``, ``mm_token_type_ids``,
    pixel features) is used VERBATIM (no splicing, so multimodal arrays stay aligned). Assistant
    regions are then located by scanning for the role-marker token-id subsequences directly in
    the expanded ``input_ids`` -- no character offsets and no text->expanded remap.

    Note: ``position_ids`` are not produced here; ``process_samples`` assigns a 1-based range and
    multimodal (mrope) position ids are expected to be recomputed by the model/trainer.
    """
    tokenizer = get_tokenizer(processor)
    is_multimodal = not is_tokenizer(processor)
    has_media = is_multimodal and _count_media_in_messages(messages) != (0, 0)

    template_caller = processor if is_multimodal else tokenizer
    if not getattr(template_caller, "chat_template", None):
        template_caller.chat_template = _FALLBACK_CHATML_JINJA

    # 0. Neutralize special-token strings in user-controlled text (no-op for normal data).
    specials = _special_token_strings(tokenizer)
    special_ids = {tid for tid, t in tokenizer.added_tokens_decoder.items() if getattr(t, "special", False)}
    messages = _escape_special_in_messages(messages, specials, special_ids, tokenizer)

    hf_messages = _to_hf_messages(messages, is_multimodal=is_multimodal)

    tools_parsed = None
    if tools:
        tools = _escape_special(tools, specials, special_ids, tokenizer)  # E3: tools text is user-controlled
        tools_parsed = json.loads(tools)
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]

    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    # 1. Render full text, then run the processor/tokenizer and use its output verbatim.
    full_text = template_caller.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=is_generate, tools=tools_parsed, **template_kwargs
    )

    if has_media:
        images, videos = _extract_media_from_messages(messages)
        # Every placeholder must come from a media block (escaping broke any literal ones).
        _check_placeholder_counts(processor, full_text, len(images), len(videos))
        proc_kwargs = {"return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        if videos:
            proc_kwargs["videos"] = videos
        outputs = processor(text=full_text, **proc_kwargs)
        input_ids = outputs["input_ids"][0].tolist()
    else:
        outputs = None
        input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # 2. Label assistant regions by scanning marker token-id subsequences in the expanded stream.
    if assistant_start_ids is None or assistant_end_ids is None:
        assistant_start_ids, assistant_end_ids = _detect_assistant_marker_ids(template_caller)

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    labels, loss_weights, regions_count = _label_assistant_regions(
        input_ids, assistant_start_ids, assistant_end_ids, assistant_messages
    )
    _verify_render(regions_count, assistant_messages)

    result = ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )

    if outputs is not None:
        for key in _MULTIMODAL_PASSTHROUGH_KEYS:
            if key in outputs:
                result[key] = outputs[key]
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
    def __init__(self, processor: Processor, name: str | None = None):
        self.processor = processor
        self.name = name
        self._assistant_start_marker = None
        self._assistant_end_marker = None
        self._assistant_start_ids = None
        self._assistant_end_ids = None

        template_caller = processor if not is_tokenizer(processor) else get_tokenizer(processor)
        if getattr(template_caller, "chat_template", None):
            try:
                self._assistant_start_marker, self._assistant_end_marker = _detect_assistant_markers(template_caller)
            except (ValueError, Exception):
                pass
            try:
                self._assistant_start_ids, self._assistant_end_ids = _detect_assistant_marker_ids(template_caller)
            except (ValueError, Exception):
                pass

    def _override(self, method_name: str):
        """Return a registered plugin override for ``method_name``, or ``None``.

        Imported lazily to avoid a core->plugins import cycle at module load.
        """
        if self.name is None:
            return None

        from ...plugins.model_plugins.rendering import RenderingPlugin

        return RenderingPlugin(self.name).get(method_name)

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
        override = self._override("render_messages")
        if override is not None:
            return override(
                self.processor,
                messages,
                tools=tools,
                is_generate=is_generate,
                enable_thinking=enable_thinking,
            )

        return _render_messages(
            self.processor,
            messages,
            tools,
            is_generate,
            self._assistant_start_marker,
            self._assistant_end_marker,
            self._assistant_start_ids,
            self._assistant_end_ids,
            enable_thinking=enable_thinking,
        )

    def parse_message(self, generated_text: str) -> Message:
        """Parse generated text into a structured Message.

        Args:
            generated_text: The raw generated text from the model.

        Returns:
            Parsed Message with typed content blocks.
        """
        override = self._override("parse_message")
        if override is not None:
            return override(generated_text)

        return _parse_message(generated_text)

    def get_dummy_media_fragment(self, modality: str) -> dict:
        """Build (and cache) a minimal valid media fragment for ``modality`` ("image"|"video").

        Renders one tiny synthetic image/video through the model's own processor and extracts
        the contiguous token span it emits for that media (the ``vision_start … vision_end``
        block, delimiters included) together with its pixel features. The collator appends this
        zero-loss fragment to a micro batch that lacks the modality so that every data-parallel
        rank invokes the vision tower the same number of times per step -- otherwise FSDP/DDP
        collectives over the (sharded) vision blocks desync and hang (NCCL timeout).

        The fragment is self-consistent by construction: the placeholder-token count matches the
        patch count, because both come from the same processor call.
        """
        if modality not in ("image", "video"):
            raise ValueError(f"Unsupported dummy media modality: {modality!r} (expected 'image' or 'video').")
        if is_tokenizer(self.processor):
            raise RuntimeError("Cannot build a dummy media fragment for a text-only processor.")

        if not hasattr(self, "_dummy_fragments"):
            self._dummy_fragments: dict[str, dict] = {}
        if modality in self._dummy_fragments:
            return self._dummy_fragments[modality]

        from PIL import Image as _PILImage

        if modality == "image":
            media_block = {"type": "image_url", "value": _PILImage.new("RGB", (64, 64))}
            target, pixel_key, grid_key = 1, "pixel_values", "image_grid_thw"
        else:
            # A minimal clip: the temporal patch size is typically 2, so provide two frames.
            media_block = {"type": "video_url", "value": np.zeros((2, 64, 64, 3), dtype=np.uint8)}
            target, pixel_key, grid_key = 2, "pixel_values_videos", "video_grid_thw"

        messages: list[Message] = [
            {"role": "user", "content": [media_block]},
            {"role": "assistant", "content": [{"type": "text", "value": "ok"}]},
        ]
        rendered = self.render_messages(messages)

        mm_type_ids = rendered.get("mm_token_type_ids")
        if not mm_type_ids or target not in mm_type_ids or pixel_key not in rendered:
            raise RuntimeError(f"Processor did not emit {modality} placeholder tokens for the dummy sample.")

        positions = [i for i, t in enumerate(mm_type_ids) if t == target]
        # Include the surrounding vision_start/vision_end delimiters so the fragment matches
        # exactly what the template emits around real media.
        lo = max(positions[0] - 1, 0)
        hi = min(positions[-1] + 2, len(rendered["input_ids"]))

        fragment: dict = {
            "input_ids": list(rendered["input_ids"][lo:hi]),
            "mm_token_type_ids": list(mm_type_ids[lo:hi]),
            pixel_key: rendered[pixel_key],
            grid_key: rendered[grid_key],
        }
        if modality == "video" and "second_per_grid_ts" in rendered:
            fragment["second_per_grid_ts"] = rendered["second_per_grid_ts"]

        self._dummy_fragments[modality] = fragment
        return fragment

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

                # Carry multimodal features. Chosen tokens precede rejected ones in the
                # concatenated sequence, so concatenate their pixel features in the same
                # order to keep the token<->pixel correspondence intact.
                for key in _MULTIMODAL_PASSTHROUGH_KEYS:
                    tensors = [inp[key] for inp in (chosen_input, rejected_input) if key in inp]
                    if tensors:
                        model_input[key] = torch.cat(tensors, dim=0)

                if "mm_token_type_ids" in chosen_input or "mm_token_type_ids" in rejected_input:
                    chosen_mm = chosen_input.get("mm_token_type_ids", [0] * len(chosen_input["input_ids"]))
                    rejected_mm = rejected_input.get("mm_token_type_ids", [0] * len(rejected_input["input_ids"]))
                    model_input["mm_token_type_ids"] = chosen_mm + rejected_mm
            else:
                raise ValueError("No valid messages or chosen_messages/rejected_messages found in sample.")

            if "extra_info" in sample:
                model_input["extra_info"] = sample["extra_info"]

            if "_dataset_name" in sample:
                model_input["_dataset_name"] = sample["_dataset_name"]

            model_inputs.append(model_input)

        return model_inputs


_MULTIMODAL_PASSTHROUGH_KEYS = frozenset(
    {
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
    }
)


def _pad_and_truncate(tensor: Tensor, max_seqlen: int, pad_value: int = 0) -> Tensor:
    if tensor.shape[-1] >= max_seqlen:
        return tensor[..., :max_seqlen]

    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seqlen - tensor.shape[-1]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=-1)


def _align_modality(
    sample: ModelInput,
    mm_type_ids: list[int],
    max_length: int,
    *,
    target: int,
    grid_key: str,
    pixel_key: str,
) -> list[int]:
    """Trim and zero one modality's orphaned tokens for a single sample.

    Layout-agnostic: a media item's placeholder tokens may be a single contiguous run or split
    into per-frame sub-runs; completeness is decided per token *position*, so both are handled
    identically.

    Returns the (possibly updated) ``mm_token_type_ids`` so chained calls see earlier zeroing.
    """
    if grid_key not in sample or pixel_key not in sample:
        return mm_type_ids

    grid = sample[grid_key]
    n_items = len(grid)
    if n_items == 0:
        return mm_type_ids

    positions = [i for i, t in enumerate(mm_type_ids) if t == target]
    patches_per_item = [int(grid[i].prod()) for i in range(n_items)]
    total_patches = sum(patches_per_item)
    total_tokens = len(positions)

    # merge_size**2 = pixel patches per placeholder token, derived from the data. Bail out
    # untouched if the sample is inconsistent.
    if total_tokens == 0 or total_patches % total_tokens != 0:
        return mm_type_ids
    merge_sq = total_patches // total_tokens
    tokens_per_item = [p // merge_sq for p in patches_per_item]
    if sum(tokens_per_item) != total_tokens:
        return mm_type_ids

    # Each item owns a contiguous slice of `positions`; it is complete iff its last
    # placeholder token lands inside the kept window [0, max_length).
    n_complete = 0
    cum = 0
    for n_i in tokens_per_item:
        if positions[cum + n_i - 1] < max_length:
            n_complete += 1
            cum += n_i
        else:
            break

    if n_complete >= n_items:
        return mm_type_ids

    # Trim pixel features and grid to the complete prefix.
    keep_patches = sum(patches_per_item[:n_complete])
    sample[pixel_key] = sample[pixel_key][:keep_patches]
    sample[grid_key] = grid[:n_complete]

    # Zero out orphaned placeholder tokens that fall inside the kept window; tokens
    # beyond max_length are removed by truncation anyway (positions are sorted).
    input_ids = list(sample["input_ids"])
    mm_type_ids = list(mm_type_ids)
    labels = list(sample["labels"]) if "labels" in sample else None
    loss_weights = list(sample["loss_weights"]) if "loss_weights" in sample else None

    for pos in positions[cum:]:
        if pos >= max_length:
            break
        input_ids[pos] = 0
        mm_type_ids[pos] = 0
        if labels is not None:
            labels[pos] = IGNORE_INDEX
        if loss_weights is not None:
            loss_weights[pos] = 0.0

    sample["input_ids"] = input_ids
    sample["mm_token_type_ids"] = mm_type_ids
    if labels is not None:
        sample["labels"] = labels
    if loss_weights is not None:
        sample["loss_weights"] = loss_weights
    return mm_type_ids


def _align_multimodal_on_truncation(sample: ModelInput, max_length: int) -> ModelInput:
    """Remove orphaned multimodal data when the sequence will be truncated.

    When cutoff_len truncates input_ids, media whose placeholder tokens are partially cut lose
    their token<->pixel correspondence. Trims pixel_values/grid_thw to the complete items and
    zeros out orphaned vision tokens so the model ignores them.
    """
    mm_type_ids = sample.get("mm_token_type_ids")
    if mm_type_ids is None:
        return sample

    sample = dict(sample)

    mm_type_ids = _align_modality(
        sample, mm_type_ids, max_length, target=1, grid_key="image_grid_thw", pixel_key="pixel_values"
    )
    mm_type_ids = _align_modality(
        sample, mm_type_ids, max_length, target=2, grid_key="video_grid_thw", pixel_key="pixel_values_videos"
    )

    # Remove empty multimodal fields entirely
    if "image_grid_thw" in sample and len(sample["image_grid_thw"]) == 0:
        del sample["pixel_values"]
        del sample["image_grid_thw"]
    if "video_grid_thw" in sample and len(sample["video_grid_thw"]) == 0:
        del sample["pixel_values_videos"]
        del sample["video_grid_thw"]

    return sample


def pad_and_truncate(samples: list[ModelInput], max_seqlen: int) -> list[BatchInput]:
    max_length = min(max(len(sample["input_ids"]) for sample in samples), max_seqlen)
    padded_samples = []
    for sample in samples:
        # Align multimodal fields before truncation: remove images/videos whose
        # placeholder tokens would be partially cut, preventing pixel<->token mismatch.
        if len(sample["input_ids"]) > max_length and any(k in sample for k in _MULTIMODAL_PASSTHROUGH_KEYS):
            sample = _align_multimodal_on_truncation(sample, max_length)

        padded_sample = {}
        for key, value in sample.items():
            if key in _MULTIMODAL_PASSTHROUGH_KEYS:
                padded_sample[key] = value
                continue

            if "label" in key:
                pad_value = IGNORE_INDEX
            else:
                pad_value = 0

            if not isinstance(value, str):
                padded_sample[key] = _pad_and_truncate(torch.tensor(value), max_length, pad_value)
            else:
                padded_sample[key] = value

        padded_samples.append(padded_sample)

    return padded_samples
