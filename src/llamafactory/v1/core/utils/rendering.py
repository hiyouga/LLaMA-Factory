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

import numpy as np

from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer
from ...utils.types import Message, ModelInput, Processor, Sample


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
    """Parse a message in ChatML format.

    Args:
        generated_text (str): The generated text in ChatML format.

    Returns:
        Message: The parsed message.
    """
    return Message(role="assistant", content=[{"type": "text", "value": generated_text}])


class Renderer:
    def __init__(self, template: str, processor: Processor):
        self.template = template
        self.processor = processor

    def render_messages(
        self, messages: list[Message], tools: str | None = None, is_generate: bool = False
    ) -> ModelInput:
        """Apply template to messages and convert them to model input.

        Args:
            messages (list[Message]): The messages to render.
            tools (str | None, optional): The tools to use. Defaults to None.
            is_generate (bool, optional): Whether to render for generation. Defaults to False.

        Returns:
            ModelInput: The rendered model input.
        """
        if self.template == "chatml":
            return render_chatml_messages(self.processor, messages, tools, is_generate)
        else:
            from ...plugins.model_plugins.rendering import RenderingPlugin

            return RenderingPlugin(self.template).render_messages(self.processor, messages, tools, is_generate)

    def parse_message(self, generated_text: str) -> Message:
        """Parse a message in the template format.

        Args:
            generated_text (str): The generated text in the template format.

        Returns:
            Message: The parsed message.
        """
        if self.template == "chatml":
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
