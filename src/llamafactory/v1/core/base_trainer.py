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

"""The definition of trainer.

Init Phase:

1. Init dataloader.
2. Init optimizer (deepspeed).
3. Shard model.
4. Init optimizer (fsdp).
5. Init scheduler.

Train Phase:
1. Train Loop

"""

from ..config.training_args import TrainingArguments
from ..utils.types import HFModel, Processor, TorchDataset
from .trainer_utils.data_collator import DataCollator


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        model: HFModel,
        processor: Processor,
        dataset: TorchDataset,
    ) -> None:
        self.args = args
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.data_collator = DataCollator()
        self.optimizer = None
        self.lr_scheduler = None

    def init_model_and_optimizer(self) -> None:
        pass

    def create_dataloader(self) -> None:
        pass

    def fit(self) -> None:
        pass
