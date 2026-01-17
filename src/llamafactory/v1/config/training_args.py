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

import os
from dataclasses import dataclass, field
from uuid import uuid4

from .arg_utils import BatchingStrategy, PluginConfig, get_plugin_config


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default=os.path.join("outputs", str(uuid4().hex)),
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int | None = field(
        default=None,
        metadata={"help": "Global batch size for training, default to DP size * micro batch size."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs."},
    )
    max_steps: int | None = field(
        default=None,
        metadata={"help": "Maximum number of training steps. If set, overrides num_train_epochs."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for training."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16 for training."},
    )
    batching_strategy: BatchingStrategy = field(
        default=BatchingStrategy.NORMAL,
        metadata={"help": "Batching strategy for training."},
    )
    batching_workers: int = field(
        default=16,
        metadata={"help": "Number of workers for batching."},
    )
    enable_activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable activation checkpointing for training."},
    )
    dist_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Distribution configuration for training."},
    )
    optim_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Optimizer configuration for training."},
    )
    lr_scheduler_config: PluginConfig | None = field(
        default=None,
        metadata={"help": "Learning rate scheduler configuration for training."},
    )

    def __post_init__(self) -> None:
        self.dist_config = get_plugin_config(self.dist_config)
        self.optim_config = get_plugin_config(self.optim_config)
        self.lr_scheduler_config = get_plugin_config(self.lr_scheduler_config)
