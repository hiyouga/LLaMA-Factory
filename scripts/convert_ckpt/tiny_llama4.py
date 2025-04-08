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

from transformers import Llama4Config, Llama4ForConditionalGeneration, Llama4TextConfig, Llama4VisionConfig


if __name__ == "__main__":
    vision_config = Llama4VisionConfig(
        hidden_size=1408,
        image_size=336,
        intermediate_size=5632,
        num_attention_heads=16,
        num_hidden_layers=4,
        vision_output_dim=4096,
    )
    text_config = Llama4TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        intermediate_size_mlp=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=512 // 8,
        num_local_experts=2,
    )
    config = Llama4Config(vision_config=vision_config, text_config=text_config)
    model = Llama4ForConditionalGeneration._from_config(config)
    model.save_pretrained("tiny-llama4")
