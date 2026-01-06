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


def test_chatml_rendering():
    tokenizer: Processor = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    renderer = Renderer(template="chatml", processor=tokenizer)
    hf_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is LLM?",
        },
    ]
    lmf_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "value": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "value": "What is LLM?"}],
        },
    ]
    hf_inputs = tokenizer.apply_chat_template(hf_messages, add_generation_prompt=True)
    lmf_inputs = renderer.render_messages(lmf_messages, is_generate=True)
    assert lmf_inputs["input_ids"] == hf_inputs
    assert lmf_inputs["attention_mask"] == [1.0] * len(hf_inputs)
    assert lmf_inputs["labels"] == [-100] * len(hf_inputs)
    assert lmf_inputs["loss_weights"] == [0.0] * len(hf_inputs)


def test_chatml_parse():
    tokenizer: Processor = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    renderer = Renderer(template="chatml", processor=tokenizer)
    generated_text = "LLM stands for Large Language Model."
    parsed_message = renderer.parse_message(generated_text)
    assert parsed_message == {
        "role": "assistant",
        "content": [{"type": "text", "value": "LLM stands for Large Language Model."}],
    }


if __name__ == "__main__":
    test_chatml_rendering()
    test_chatml_parse()
