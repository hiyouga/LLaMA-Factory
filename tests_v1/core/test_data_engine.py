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

import random

import pytest
from datasets import load_dataset

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine


@pytest.mark.parametrize("num_samples", [16])
def test_map_dataset(num_samples: int):
    data_args = DataArguments(dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args)
    original_data = load_dataset("llamafactory/v1-sft-demo", split="train")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        assert data_engine[index] == {"_dataset_name": "default", **original_data[index]}


if __name__ == "__main__":
    test_map_dataset(1)
