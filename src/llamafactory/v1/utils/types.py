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

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, Union


if TYPE_CHECKING:
    import datasets
    import numpy as np
    import torch
    import torch.utils.data
    import transformers
    from torch.distributed import ProcessGroup
    from torch.distributed.fsdp import FullyShardedDataParallel

    Tensor = torch.Tensor
    TensorLike = Union[int, float, list[int], list[float], np.ndarray, Tensor]
    TorchDataset = Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]
    HFDataset = Union[datasets.Dataset, datasets.IterableDataset]
    DataCollator = transformers.DataCollator
    DataLoader = torch.utils.data.DataLoader
    HFConfig = transformers.PretrainedConfig
    HFModel = transformers.PreTrainedModel
    DistModel = Union[torch.nn.parallel.DistributedDataParallel, FullyShardedDataParallel]
    Processor = Union[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]
    Optimizer = torch.optim.Optimizer
    Scheduler = torch.optim.lr_scheduler.LRScheduler
    ProcessGroup = ProcessGroup
else:
    Tensor = None
    TensorLike = None
    TorchDataset = None
    HFDataset = None
    DataCollator = None
    DataLoader = None
    HFConfig = None
    HFModel = None
    DistModel = None
    Processor = None
    Optimizer = None
    Scheduler = None
    ProcessGroup = None


class DatasetInfo(TypedDict, total=False):
    path: str
    """Local file path."""
    source: NotRequired[Literal["hf_hub", "ms_hub", "local"]]
    """Dataset source, default to "hf_hub"."""
    split: NotRequired[str]
    """Dataset split, default to "train"."""
    converter: NotRequired[str]
    """Dataset converter, default to None."""
    size: NotRequired[int]
    """Number of samples, default to all samples."""
    weight: NotRequired[float]
    """Dataset weight, default to 1.0."""
    streaming: NotRequired[bool]
    """Is streaming dataset, default to False."""


class DistributedConfig(TypedDict, total=False):
    mp_replicate_size: NotRequired[int]
    """Model parallel replicate size, default to 1."""
    mp_shard_size: NotRequired[int]
    """Model parallel shard size, default to world_size // mp_replicate_size."""
    dp_size: NotRequired[int]
    """Data parallel size, default to world_size // cp_size."""
    cp_size: NotRequired[int]
    """Context parallel size, default to 1."""
    timeout: NotRequired[int]
    """Timeout for distributed communication, default to 600."""


class Content(TypedDict):
    type: Literal["text", "reasoning", "tools", "tool_calls", "image_url"]
    value: str


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[Content]
    loss_weight: float


class SFTSample(TypedDict):
    messages: list[Message]
    extra_info: NotRequired[str]
    _dataset_name: NotRequired[str]


class DPOSample(TypedDict):
    chosen_messages: list[Message]
    rejected_messages: list[Message]
    extra_info: NotRequired[str]
    _dataset_name: NotRequired[str]


Sample = Union[SFTSample, DPOSample]
