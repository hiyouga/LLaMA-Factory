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
class DataArguments:
    dataset: str | None = field(
        default=None,
        metadata={"help": "Path to the dataset."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "Cutoff length for the dataset."},
    )
