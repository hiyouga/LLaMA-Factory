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
renderer = Renderer(template, processor)
renderer.render_messages(messages: list[Message], tools: str | None) -> ModelInputs
renderer.parse_message(text: str) -> Message
renderer.process_samples(samples: list[Sample]) -> list[ModelInput]
"""

import json

import numpy as np

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer, is_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample


DEFAULT_CHATML_JINJA = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)

DEFAULT_TRAINING_JINJA = (
    "{% for message in messages %}"
    "{% if message['content'] is string %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% else %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'text' %}{{item['text']}}"
    "{% elif item['type'] == 'image' %}{{'<|vision_start|><|image_pad|><|vision_end|>'}}"
    "{% elif item['type'] == 'video' %}{{'<|vision_start|><|video_pad|><|vision_end|>'}}"
    "{% endif %}"
    "{% endfor %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


def _is_prefix_stable(template_caller, tokenizer) -> bool:
    """Check if the chat template is prefix-stable (render(prefix) is a prefix of render(full)).

    Tests multiple message patterns to catch templates that strip/add content
    based on message position (e.g., Qwen3.5 strips <think> from non-last assistant).
    """
    test_cases = [
        [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "more"},
        ],
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "second reply"},
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ],
    ]
    for msgs in test_cases:
        for i in range(1, len(msgs)):
            try:
                text_short = template_caller.apply_chat_template(
                    msgs[:i], tokenize=False, add_generation_prompt=False
                )
                text_long = template_caller.apply_chat_template(
                    msgs[: i + 1], tokenize=False, add_generation_prompt=False
                )
            except Exception:
                return False
            if not text_long.startswith(text_short):
                return False
    return True


def _to_hf_messages(messages: list[Message], is_multimodal: bool = False) -> list[dict]:
    """Convert v1 Message format to HF format for apply_chat_template."""
    hf_messages = []
    for message in messages:
        if is_multimodal:
            hf_content = []
            for content in message["content"]:
                if content["type"] == "text":
                    hf_content.append({"type": "text", "text": content["value"]})
                elif content["type"] == "reasoning":
                    hf_content.append({"type": "text", "text": "<think>\n" + content["value"] + "\n</think>\n"})
                elif content["type"] == "tool_call":
                    hf_content.append({"type": "text", "text": content["value"]})
                elif content["type"] == "image_url":
                    hf_content.append({"type": "image", "image": content["value"]})
                elif content["type"] == "video_url":
                    hf_content.append({"type": "video", "video": content["value"]})
                elif content["type"] == "audio_url":
                    hf_content.append({"type": "audio", "audio": content["value"]})
            hf_messages.append({"role": message["role"], "content": hf_content})
        else:
            text = ""
            for content in message["content"]:
                if content["type"] == "text":
                    text += content["value"]
                elif content["type"] == "reasoning":
                    text += "<think>\n" + content["value"] + "\n</think>\n"
                elif content["type"] == "tool_call":
                    text += content["value"]
            hf_messages.append({"role": message["role"], "content": text})
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


def _build_text_to_expanded_mapping(
    text_ids: list[int],
    expanded_ids: list[int],
    video_grid_thw,
    image_pad_id: int,
    vision_start_id: int,
    vision_end_id: int,
) -> list[int]:
    """Build a cumulative position mapping from text_ids positions to expanded_ids positions.

    Returns a list of length len(text_ids)+1 where mapping[i] gives the corresponding
    position in expanded_ids. Turn boundaries (which fall at non-media positions) can be
    looked up directly.
    """
    mapping = [0] * (len(text_ids) + 1)
    t_cursor = 0
    e_cursor = 0
    video_idx = 0

    while t_cursor < len(text_ids):
        if (
            t_cursor + 2 < len(text_ids)
            and text_ids[t_cursor] == vision_start_id
            and text_ids[t_cursor + 2] == vision_end_id
        ):
            media_type = text_ids[t_cursor + 1]
            mapping[t_cursor] = e_cursor

            if media_type == image_pad_id:
                assert expanded_ids[e_cursor] == vision_start_id
                e_cursor += 1
                while e_cursor < len(expanded_ids) and expanded_ids[e_cursor] == image_pad_id:
                    e_cursor += 1
                assert expanded_ids[e_cursor] == vision_end_id
                e_cursor += 1
            else:
                num_frames = int(video_grid_thw[video_idx][0])
                frames_found = 0
                while frames_found < num_frames:
                    if expanded_ids[e_cursor] == vision_end_id:
                        frames_found += 1
                    e_cursor += 1
                video_idx += 1

            mapping[t_cursor + 1] = e_cursor
            mapping[t_cursor + 2] = e_cursor
            t_cursor += 3
        else:
            mapping[t_cursor] = e_cursor
            t_cursor += 1
            e_cursor += 1

    mapping[len(text_ids)] = e_cursor
    return mapping


def _render_auto_messages_multimodal(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
) -> ModelInput:
    """Render messages for multimodal models using processor to expand image/video placeholders.

    Optimization: calls processor only once for the full sequence, then uses a position
    mapping (text_ids → expanded_ids) to compute per-turn boundaries without repeated
    processor calls.
    """
    tokenizer = get_tokenizer(processor)

    if not getattr(processor, "chat_template", None):
        processor.chat_template = DEFAULT_TRAINING_JINJA

    hf_messages = _to_hf_messages(messages, is_multimodal=True)
    images, videos = _extract_media_from_messages(messages)

    tools_parsed = None
    if tools:
        tools_parsed = json.loads(tools)
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]

    first_user_idx = 0
    if tools_parsed:
        for idx, msg in enumerate(messages):
            if msg["role"] == "user":
                first_user_idx = idx
                break

    full_text = processor.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=is_generate, tools=tools_parsed
    )

    proc_kwargs = {"return_tensors": "pt"}
    if images:
        proc_kwargs["images"] = images
    if videos:
        proc_kwargs["videos"] = videos

    outputs = processor(text=full_text, **proc_kwargs)
    expanded_input_ids = outputs["input_ids"][0].tolist()

    full_text_no_gen = processor.apply_chat_template(
        hf_messages, tokenize=False, add_generation_prompt=False, tools=tools_parsed
    )
    text_ids_full = tokenizer.encode(full_text_no_gen, add_special_tokens=False)

    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    mapping = _build_text_to_expanded_mapping(
        text_ids_full,
        expanded_input_ids,
        outputs.get("video_grid_thw"),
        image_pad_id,
        vision_start_id,
        vision_end_id,
    )

    input_ids, labels, loss_weights = [], [], []
    prev_expanded_pos = 0

    for i, message in enumerate(messages):
        if tools_parsed and i < first_user_idx:
            continue

        curr_text = processor.apply_chat_template(
            hf_messages[: i + 1], tokenize=False, add_generation_prompt=False, tools=tools_parsed
        )
        curr_text_len = len(tokenizer.encode(curr_text, add_special_tokens=False))
        curr_expanded_pos = mapping[curr_text_len]

        turn_ids = expanded_input_ids[prev_expanded_pos:curr_expanded_pos]
        turn_len = curr_expanded_pos - prev_expanded_pos

        if tools_parsed and i == first_user_idx and first_user_idx > 0:
            turn_weight = 0.0
        else:
            turn_weight = message.get("loss_weight", 1.0 if message["role"] == "assistant" else 0.0)

        input_ids.extend(turn_ids)
        loss_weights.extend([turn_weight] * turn_len)
        if turn_weight > 1e-6:
            labels.extend(turn_ids)
        else:
            labels.extend([IGNORE_INDEX] * turn_len)

        prev_expanded_pos = curr_expanded_pos

    if is_generate:
        gen_suffix = expanded_input_ids[prev_expanded_pos:]
        input_ids.extend(gen_suffix)
        loss_weights.extend([0.0] * len(gen_suffix))
        labels.extend([IGNORE_INDEX] * len(gen_suffix))
    else:
        assert prev_expanded_pos == len(expanded_input_ids), (
            f"Position mapping mismatch: computed {prev_expanded_pos}, "
            f"actual {len(expanded_input_ids)}. Template may not be prefix-stable."
        )

    result = ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )

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


def _render_auto_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
) -> ModelInput:
    """Render messages using apply_chat_template with per-turn loss masking."""
    tokenizer = get_tokenizer(processor)
    is_multimodal = not is_tokenizer(processor)

    if is_multimodal and _count_media_in_messages(messages) != (0, 0):
        return _render_auto_messages_multimodal(processor, messages, tools, is_generate)

    template_caller = processor if is_multimodal else tokenizer
    if not getattr(template_caller, "chat_template", None):
        template_caller.chat_template = DEFAULT_TRAINING_JINJA

    hf_messages = _to_hf_messages(messages, is_multimodal=is_multimodal)

    tools_parsed = None
    if tools:
        tools_parsed = json.loads(tools)
        if not isinstance(tools_parsed, list):
            tools_parsed = [tools_parsed]

    input_ids, labels, loss_weights = [], [], []
    prev_ids = []

    # When tools are present, some templates require a user message in the slice.
    # Find the first user message index to batch system+user together.
    first_user_idx = 0
    if tools_parsed:
        for idx, msg in enumerate(messages):
            if msg["role"] == "user":
                first_user_idx = idx
                break

    for i, message in enumerate(messages):
        if tools_parsed and i < first_user_idx:
            continue

        curr_text = template_caller.apply_chat_template(
            hf_messages[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools_parsed,
        )
        curr_ids = tokenizer.encode(curr_text, add_special_tokens=False)

        turn_ids = curr_ids[len(prev_ids):]
        if tools_parsed and i == first_user_idx and first_user_idx > 0:
            turn_weight = 0.0
        else:
            turn_weight = message.get("loss_weight", 1.0 if message["role"] == "assistant" else 0.0)

        input_ids.extend(turn_ids)
        loss_weights.extend([turn_weight] * len(turn_ids))
        if turn_weight > 1e-6:
            labels.extend(turn_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(turn_ids))

        prev_ids = curr_ids

    if is_generate:
        gen_text = template_caller.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_parsed,
        )
        gen_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        gen_suffix = gen_ids[len(prev_ids):]
        input_ids.extend(gen_suffix)
        loss_weights.extend([0.0] * len(gen_suffix))
        labels.extend([IGNORE_INDEX] * len(gen_suffix))

    return ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )


def _parse_auto_message(generated_text: str) -> Message:
    """Parse generated text to Message (generic)."""
    return Message(role="assistant", content=[{"type": "text", "value": generated_text}])


def render_chatml_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
) -> ModelInput:
    """Apply chatml template to messages and convert them to model input.

    See https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen2-7B-Instruct
    """
    tokenizer = get_tokenizer(processor)
    input_ids, labels, loss_weights = [], [], []

    for message in messages:
        temp_str = "<|im_start|>" + message["role"] + "\n"
        for content in message["content"]:
            if content["type"] == "text":
                temp_str += content["value"]
            else:
                raise ValueError(f"Unsupported content type: {content['type']}")

        temp_str += "<|im_end|>\n"
        temp_weight = message.get("loss_weight", 1.0 if message["role"] == "assistant" else 0.0)
        temp_ids = tokenizer.encode(temp_str, add_special_tokens=False)
        input_ids.extend(temp_ids)
        loss_weights.extend([temp_weight] * len(temp_ids))
        if temp_weight > 1e-6:
            labels.extend(temp_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(temp_ids))

    if is_generate:
        temp_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        input_ids.extend(temp_ids)
        loss_weights.extend([0.0] * len(temp_ids))
        labels.extend([IGNORE_INDEX] * len(temp_ids))

    return ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )


def parse_chatml_message(generated_text: str) -> Message:
    """Parse a message in ChatML format."""
    return Message(role="assistant", content=[{"type": "text", "value": generated_text}])


class Renderer:
    def __init__(self, template: str, processor: Processor):
        self.template = template
        self.processor = processor

    def render_messages(
        self,
        messages: list[Message],
        tools: str | None = None,
        is_generate: bool = False,
        enable_thinking: bool = False,
    ) -> ModelInput:
        """Apply template to messages and convert them to model input.

        Args:
            messages (list[Message]): The messages to render.
            tools (str | None, optional): The tools to use. Defaults to None.
            is_generate (bool, optional): Whether to render for generation. Defaults to False.
            enable_thinking (bool, optional): Whether to enable thinking mode for generation. Defaults to False.

        Returns:
            ModelInput: The rendered model input.
        """
        if self.template == "auto":
            return _render_auto_messages(self.processor, messages, tools, is_generate)
        elif self.template == "chatml":
            return render_chatml_messages(self.processor, messages, tools, is_generate)
        else:
            from ...plugins.model_plugins.rendering import RenderingPlugin

            return RenderingPlugin(self.template).render_messages(
                self.processor, messages, tools, is_generate, enable_thinking
            )

    def parse_message(self, generated_text: str) -> Message:
        """Parse a message in the template format.

        Args:
            generated_text (str): The generated text in the template format.

        Returns:
            Message: The parsed message.
        """
        if self.template == "auto":
            return _parse_auto_message(generated_text)
        elif self.template == "chatml":
            return parse_chatml_message(generated_text)
        else:
            from ...plugins.model_plugins.rendering import RenderingPlugin

            return RenderingPlugin(self.template).parse_message(generated_text)

    def process_samples(self, samples: list[Sample]) -> list[ModelInput]:
        """Process samples to model input.

        Args:
            samples (list[Sample]): The samples to process.

        Returns:
            list[ModelInput]: The processed model inputs.
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
