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


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="",
        metadata={"help": "Path to the output directory."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size for training."},
    )
    global_batch_size: int = field(
        default=1,
        metadata={"help": "Global batch size for training."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for training."},
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16 for training."},
    )
