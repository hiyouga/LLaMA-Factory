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


from dataclasses import dataclass, field
from typing import Optional

from .arg_utils import ModelClass, PluginConfig, get_plugin_config


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to the model or model identifier from Hugging Face."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code from Hugging Face."},
    )
    use_fast_processor: bool = field(
        default=True,
        metadata={"help": "Use fast processor from Hugging Face."},
    )
    model_class: ModelClass = field(
        default=ModelClass.LLM,
        metadata={"help": "Model class from Hugging Face."},
    )
    peft_config: Optional[PluginConfig] = field(
        default=None,
        metadata={"help": "PEFT configuration for the model."},
    )
    kernel_config: Optional[PluginConfig] = field(
        default=None,
        metadata={"help": "Kernel configuration for the model."},
    )
    quant_config: Optional[PluginConfig] = field(
        default=None,
        metadata={"help": "Quantization configuration for the model."},
    )

    def __post_init__(self) -> None:
        self.peft_config = get_plugin_config(self.peft_config)
        self.kernel_config = get_plugin_config(self.kernel_config)
        self.quant_config = get_plugin_config(self.quant_config)
