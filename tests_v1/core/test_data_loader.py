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

"""Integration tests for DataLoader with different combinations of packing and dynamic batching.

Tests the 4 scenarios:
a) non pack + non dynamic.
b) non pack + dynamic.
c) pack + non dynamic.
d) pack + dynamic.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.core.trainer_utils.data_collator import (
    DefaultCollator,
)
from llamafactory.v1.core.trainer_utils.data_loader import DataLoader
from llamafactory.v1.plugins.data_plugins.template import QwenTemplate
from llamafactory.v1.utils.batching_queue import TextBatchingQueue


class TensorDataset(Dataset):
    """Wrapper dataset that converts DataEngine samples to tensor format."""

    def __init__(self, data_engine: DataEngine, processor, template, max_samples: int = None):
        self.data_engine = data_engine
        self.processor = processor
        self.template = template
        self.max_samples = max_samples or len(data_engine)
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def __len__(self):
        return min(self.max_samples, len(self.data_engine))

    def __getitem__(self, idx):
        # Get sample from DataEngine
        sample = self.data_engine[idx]

        # Extract messages from sample
        # DataEngine returns samples with format like {"messages": [...], ...}
        # For llamafactory/v1-sft-demo, the format should have "messages" field
        messages = None
        if "messages" in sample:
            messages = sample["messages"]
        elif "conversations" in sample:
            messages = sample["conversations"]
        elif "conversation" in sample:
            messages = sample["conversation"]
        else:
            # Try to find message-like fields (skip _dataset_name)
            for key, value in sample.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, list) and len(value) > 0:
                    # Check if it looks like a message list
                    if isinstance(value[0], dict) and "role" in value[0]:
                        messages = value
                        break

        if messages is None:
            raise ValueError(f"Could not find messages in sample: {list(sample.keys())}")

        # Encode messages using template
        encoded = self.template.encode_messages(self.tokenizer, messages)

        # Convert to tensors
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
        }


def create_real_dataset(max_samples: int = 20, batch_size: int = 4):
    """Create a real dataset using DataEngine."""
    data_args = DataArguments(dataset="llamafactory/v1-sft-demo")
    data_engine = DataEngine(data_args)

    # Create processor and template
    processor = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen2.5")
    template = QwenTemplate()

    # Create tensor dataset
    raw_data_dataset = TensorDataset(data_engine, processor, template, max_samples=max_samples)

    # Create torch DataLoader
    torch_dataloader = TorchDataLoader(
        raw_data_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    return torch_dataloader, processor, template


class TestDataLoaderNonPackNonDynamic:
    """Test case a) non pack + non dynamic."""

    def test_basic_functionality(self):
        """Test DataLoader without packing and without dynamic batching."""
        # Create real dataset
        torch_dataloader, processor, template = create_real_dataset(max_samples=80, batch_size=8)

        # Create collator (non-packing)
        collator = DefaultCollator(processor=processor, template=template)

        # Create DataLoader without batching_queue (non-dynamic)
        data_loader = DataLoader(
            dataloader=torch_dataloader,
            collate_fn=collator,
            num_micro_batch=1,
            batching_queue=None,
        )

        # Iterate and check results
        batches = list(iter(data_loader))
        assert len(batches) > 0

        # Check first batch
        one_batch = batches[0]
        micro_batches = one_batch[0]
        assert "input_ids" in micro_batches
        assert "attention_mask" in micro_batches
        assert "labels" in micro_batches
        assert micro_batches["input_ids"].shape[0] == 1  # batch_size=1
        assert micro_batches["input_ids"].ndim == 2  # [batch_size, seq_len]


class TestDataLoaderNonPackDynamic:
    """Test case b) non pack + dynamic."""

    def test_basic_functionality(self):
        """Test DataLoader without packing but with dynamic batching."""
        # Create real dataset
        torch_dataloader, processor, template = create_real_dataset(max_samples=80, batch_size=8)
        collator = DefaultCollator(processor=processor, template=template)

        # Create batching queue for dynamic batching
        batching_queue = TextBatchingQueue(
            token_micro_bsz=120,
            buffer_size=8,
        )

        data_loader = DataLoader(
            dataloader=torch_dataloader,
            collate_fn=collator,
            num_micro_batch=4,
            batching_queue=batching_queue,
        )

        # Iterate and check
        batches = list(iter(data_loader))
        micro_batch_tokens_first = [micro_batch["attention_mask"].sum() for micro_batch in batches[0]]
        assert all(num_tokens <= 120 for num_tokens in micro_batch_tokens_first)
        assert len(batches) > 0
