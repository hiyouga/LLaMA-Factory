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

"""Rendering: turn a v1 ``Sample`` into a tokenized ``ModelInput``.

This module is the orchestration + public API (``Renderer``). The mechanical pieces live in
sibling modules:
  - ``rendering_format``  -- v1<->HF message conversion, media extraction, subseq search
  - ``rendering_escape``  -- special-token escaping (prompt-injection hardening)
  - ``rendering_label``   -- assistant-region labeling + structural verification
  - ``markers``           -- per-model assistant role markers (explicit whitelist)
  - ``collation``         -- batch padding/truncation/MM alignment (consumed by the batch generators)

To support a new model, add its assistant-role markers to ``markers._ASSISTANT_MARKERS``; the
built-in ``_render_messages`` / ``_parse_message`` then handle it via the model's own chat template.
"""

import json
import re

import numpy as np
import torch

from ...utils.helper import get_tokenizer, is_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample
from ..utils.collation import _MULTIMODAL_PASSTHROUGH_KEYS
from .escape import _escape_special, _escape_special_in_messages, _special_token_strings
from .format import (
    _FALLBACK_CHATML_JINJA,
    _count_media_in_messages,
    _extract_media_from_messages,
    _to_hf_messages,
)
from .label import _check_placeholder_counts, _label_assistant_regions, _verify_render
from .markers import resolve_assistant_markers


def _render_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
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
        raise ValueError("assistant marker ids were not resolved; construct Renderer with a supported model.")

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
    def __init__(self, processor: Processor, config=None):
        self.processor = processor

        # Resolve the assistant role markers from the explicit per-model whitelist (no probing),
        # then encode them with this model's tokenizer to get the token-id forms used for labeling.
        # For supported models the marker strings tokenize identically standalone and in-context.
        model_type = getattr(config, "model_type", None)
        start_marker, end_marker = resolve_assistant_markers(model_type)
        tokenizer = get_tokenizer(processor)
        self._assistant_start_ids = tokenizer(start_marker, add_special_tokens=False)["input_ids"]
        self._assistant_end_ids = tokenizer(end_marker, add_special_tokens=False)["input_ids"]
        if not self._assistant_start_ids or not self._assistant_end_ids:
            raise ValueError(f"Empty assistant marker ids for model_type {model_type!r}.")

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
