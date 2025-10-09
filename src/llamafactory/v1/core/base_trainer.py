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

from ..config.training_args import TrainingArguments
from ..extras.types import DataCollator, Model, Processor, TorchDataset


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: Model,
        processor: Processor,
        dataset: TorchDataset,
        data_collator: DataCollator,
    ) -> None:
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.data_collator = data_collator
        self.optimizer = None
        self.lr_scheduler = None

    def create_dataloader(self) -> None:
        pass

    def fit(self) -> None:
        pass
