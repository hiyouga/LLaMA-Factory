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


from ...utils.constants import IGNORE_INDEX
from ...utils.helper import get_tokenizer
from ...utils.types import Message, ModelInput, Processor


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
    """Parse a message in ChatML format. Supports interleaved reasoning and tool calls.

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
        if self.template == "chatml":
            return render_chatml_messages(self.processor, messages, tools, is_generate)
        else:
            from ...plugins.model_plugins.rendering import RenderingPlugin

            return RenderingPlugin(self.template).render_messages(self.processor, messages, tools, is_generate)

    def parse_message(self, generated_text: str) -> Message:
        if self.template == "chatml":
            return parse_chatml_message(generated_text)
        else:
            from ...plugins.model_plugins.rendering import RenderingPlugin

            return RenderingPlugin(self.template).parse_message(generated_text)
