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

from ...utils.types import Processor, Tensor, TorchDataset


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


class DataLoader:
    """Default DataLoader."""

    def __init__(self, dataset: TorchDataset) -> None:
        self.dataset = dataset
        # 1. Init stateful dataloader (tokenize)
        # 2. Add to buffer (2 * max seq len per device)
        # 3. Yield batch indexes (micro batch * grad acc)
        #    a ) non pack + non dynamic
        #    b ) non pack + dynamic
        #    c ) pack + non dynamic
        #    d ) pack + dynamic
