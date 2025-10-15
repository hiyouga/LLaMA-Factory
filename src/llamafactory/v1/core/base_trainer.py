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

from typing import Any

from ..config.training_args import TrainingArguments
from ..extras.types import Processor, Tensor, TorchDataset
from .model_worker import ModelWorker


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        """Collate features into a batch."""
        for feature in features:
            pass

        # sft: messages
        # dpo: chosen_messages, rejected_messages


class BaseTrainer:
    def __init__(
        self,
        args: TrainingArguments,
        dataset: TorchDataset,
        data_collator: DataCollator,
        model_worker: ModelWorker,
    ) -> None:
        self.args = args
        self.dataset = dataset
        self.data_collator = data_collator
        self.model_worker = model_worker
        self.optimizer = None
        self.lr_scheduler = None

    def init_device_mesh(self) -> None:
        pass

    def init_model_and_optimizer(self) -> None:
        self.model_config = self.model_worker.get_model_config()
        with self.dist_plugin.get_model_init_context():
            self.model = self.model_worker.get_model(self.model_config)

    def create_dataloader(self) -> None:
        pass

    def fit(self) -> None:
        pass
