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


from typing import TYPE_CHECKING, Optional, Union

from ..config.data_args import DataArguments


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin

    Processor = Union["PreTrainedTokenizer", "ProcessorMixin"]


class DataCollator:
    def __init__(self, processor: "Processor") -> None:
        self.processor = processor


class DataEngine:
    def __init__(self, data_args: DataArguments) -> None:
        self.args = data_args
        self.datasets = []

    def get_dataset(self) -> "dict[str, Dataset]":
        pass

    def load_single_dataset(self, converter: Optional[str] = None) -> "Dataset":
        pass

    def mix_datasets(self, datasets: "dict[str, Dataset]") -> "Dataset":
        pass

    def get_data_collator(self, processor: "Processor") -> "DataCollator":
        pass
