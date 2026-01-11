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

from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    config = Qwen3Config(
        hidden_size=1408,
        image_size=336,
        intermediate_size=5632,
        num_attention_heads=16,
        num_hidden_layers=4,
        vision_output_dim=4096,
    )
    model = Qwen3ForCausalLM.from_config(config)
    model.save_pretrained("tiny-qwen3")
    tokenizer.save_pretrained("tiny-qwen3")
    model.push_to_hub("llamafactory/tiny-random-qwen3")
    tokenizer.push_to_hub("llamafactory/tiny-random-qwen3")
