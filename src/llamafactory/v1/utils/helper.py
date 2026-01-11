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


import torch
from transformers import PreTrainedTokenizer

from ..accelerator.interface import DistributedInterface
from .constants import IGNORE_INDEX
from .types import BatchInput, ModelInput, Processor, Tensor


def is_tokenizer(processor: Processor) -> bool:
    """Check if processor is tokenizer.

    Args:
        processor: Processor.

    Returns:
        Whether processor is tokenizer.
    """
    return not hasattr(processor, "tokenizer")


def get_tokenizer(processor: Processor) -> PreTrainedTokenizer:
    """Get tokenizer from processor.

    Args:
        processor: Processor.

    Returns:
        Tokenizer.
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def _pad_and_truncate(tensor: Tensor, max_seqlen: int, pad_value: int = 0) -> Tensor:
    if tensor.shape[-1] >= max_seqlen:
        return tensor[..., :max_seqlen]

    pad_shape = list(tensor.shape)
    pad_shape[-1] = max_seqlen - tensor.shape[-1]
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=-1)


def pad_and_truncate(samples: list[ModelInput], max_seqlen: int) -> list[BatchInput]:
    max_length = min(max(len(sample["input_ids"]) for sample in samples), max_seqlen)
    padded_samples = []
    for sample in samples:
        padded_sample = {}
        for key, value in sample.items():
            if "label" in key:
                pad_value = IGNORE_INDEX
            else:
                pad_value = 0

            if not isinstance(value, str):
                padded_sample[key] = _pad_and_truncate(torch.tensor(value), max_length, pad_value)
            else:
                padded_sample[key] = value

        padded_samples.append(padded_sample)

    return padded_samples


def compute_valid_tokens(batches: list[BatchInput]) -> int:
    """Compute valid tokens in batches.

    Args:
        batches: Batches.

    Returns:
        Number of valid tokens.
    """
    device = DistributedInterface().current_device
    return sum(
        (batch["labels"].to(device, non_blocking=True) != IGNORE_INDEX).sum().item()
        for batch in batches
        if "labels" in batch
    )
