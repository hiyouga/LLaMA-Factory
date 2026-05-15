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


def _render_auto_messages_multimodal(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
) -> ModelInput:
    """Render messages for multimodal models using processor to expand image/video placeholders."""
    if not getattr(processor, "chat_template", None):
        processor.chat_template = DEFAULT_CHATML_JINJA

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

    input_ids, labels, loss_weights = [], [], []
    prev_len = 0
    images_so_far, videos_so_far = [], []

    for i, message in enumerate(messages):
        if tools_parsed and i < first_user_idx:
            continue

        for content in message["content"]:
            if content["type"] == "image_url":
                images_so_far.append(content["value"])
            elif content["type"] == "video_url":
                videos_so_far.append(content["value"])

        curr_text = processor.apply_chat_template(
            hf_messages[: i + 1], tokenize=False, add_generation_prompt=False, tools=tools_parsed
        )

        curr_kwargs = {"return_tensors": "pt"}
        if images_so_far:
            curr_kwargs["images"] = images_so_far
        if videos_so_far:
            curr_kwargs["videos"] = videos_so_far
        curr_outputs = processor(text=curr_text, **curr_kwargs)
        curr_len = len(curr_outputs["input_ids"][0])

        turn_len = curr_len - prev_len
        if tools_parsed and i == first_user_idx and first_user_idx > 0:
            turn_weight = 0.0
        else:
            turn_weight = message.get("loss_weight", 1.0 if message["role"] == "assistant" else 0.0)

        turn_ids = expanded_input_ids[prev_len:curr_len]
        input_ids.extend(turn_ids)
        loss_weights.extend([turn_weight] * turn_len)
        if turn_weight > 1e-6:
            labels.extend(turn_ids)
        else:
            labels.extend([IGNORE_INDEX] * turn_len)

        prev_len = curr_len

    if is_generate:
        gen_suffix = expanded_input_ids[prev_len:]
        input_ids.extend(gen_suffix)
        loss_weights.extend([0.0] * len(gen_suffix))
        labels.extend([IGNORE_INDEX] * len(gen_suffix))

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
        template_caller.chat_template = DEFAULT_CHATML_JINJA

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
