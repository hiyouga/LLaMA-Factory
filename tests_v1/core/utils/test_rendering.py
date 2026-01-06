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

from transformers import AutoTokenizer

from llamafactory.v1.core.utils.rendering import Renderer
from llamafactory.v1.utils.types import Processor


HF_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is LLM?"},
    {"role": "assistant", "content": "LLM stands for Large Language Model."},
]
V1_MESSAGES = [
    {"role": "system", "content": [{"type": "text", "value": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "value": "What is LLM?"}]},
    {"role": "assistant", "content": [{"type": "text", "value": "LLM stands for Large Language Model."}]},
]


def test_chatml_rendering():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)

    hf_inputs = tokenizer.apply_chat_template(HF_MESSAGES[:-1], add_generation_prompt=True)
    v1_inputs = renderer.render_messages(V1_MESSAGES[:-1], is_generate=True)
    assert v1_inputs["input_ids"] == hf_inputs
    assert v1_inputs["attention_mask"] == [1] * len(hf_inputs)
    assert v1_inputs["labels"] == [-100] * len(hf_inputs)
    assert v1_inputs["loss_weights"] == [0.0] * len(hf_inputs)

    hf_inputs_part = tokenizer.apply_chat_template(HF_MESSAGES[:-1], add_generation_prompt=False)
    hf_inputs_full = tokenizer.apply_chat_template(HF_MESSAGES, add_generation_prompt=False)
    v1_inputs_full = renderer.render_messages(V1_MESSAGES, is_generate=False)
    assert v1_inputs_full["input_ids"] == hf_inputs_full
    assert v1_inputs_full["attention_mask"] == [1] * len(hf_inputs_full)
    assert v1_inputs_full["labels"] == [-100] * len(hf_inputs_part) + hf_inputs_full[len(hf_inputs_part) :]
    assert v1_inputs_full["loss_weights"] == [0.0] * len(hf_inputs_part) + [1.0] * (
        len(hf_inputs_full) - len(hf_inputs_part)
    )


def test_chatml_parse():
    tokenizer: Processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    renderer = Renderer(template="chatml", processor=tokenizer)
    generated_text = "LLM stands for Large Language Model."
    parsed_message = renderer.parse_message(generated_text)
    assert parsed_message == V1_MESSAGES[-1]


if __name__ == "__main__":
    test_chatml_rendering()
    test_chatml_parse()
