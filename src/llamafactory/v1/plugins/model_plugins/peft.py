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

from typing import Literal, TypedDict

from peft import LoraConfig, PeftModel, get_peft_model

from ...utils.plugin import BasePlugin
from ...utils.types import HFModel


class LoraConfigDict(TypedDict, total=False):
    name: Literal["lora"]
    """Plugin name."""
    r: int
    """Lora rank."""
    lora_alpha: int
    """Lora alpha."""
    target_modules: list[str]
    """Target modules."""


class PeftPlugin(BasePlugin):
    pass


@PeftPlugin("lora").register
def get_lora_model(model: HFModel, config: LoraConfigDict) -> PeftModel:
    peft_config = LoraConfig(**config)
    model = get_peft_model(model, peft_config)
    return model
