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

import torch


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help": "Path to the model or model identifier from Hugging Face."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code from Hugging Face."},
    )


@dataclass
class LoraArguments:
    """LoRA configuration arguments"""

    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank (dimension)"},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA scaling factor, defaults to rank*2"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout rate"},
    )
    lora_target: str = field(
        default="all",
        metadata={"help": "Target modules: 'all' or comma-separated module names"},
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Additional trainable modules"},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use RSLora"},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether to use DoRA"},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether to use PiSSA initialization"},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "Number of FSVD iterations for PiSSA"},
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA+ learning rate ratio (lr_B / lr_A)"},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA+ learning rate for embedding layers"},
    )

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_rank * 2