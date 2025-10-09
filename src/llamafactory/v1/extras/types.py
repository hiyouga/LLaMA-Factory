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

from typing import TYPE_CHECKING, NotRequired, TypedDict, Union


if TYPE_CHECKING:
    import datasets
    import torch
    import torch.utils.data
    import transformers

    Tensor = torch.Tensor
    TorchDataset = Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]
    HFDataset = Union[datasets.Dataset, datasets.IterableDataset]
    DataCollator = transformers.DataCollator
    DataLoader = torch.utils.data.DataLoader
    Model = transformers.PreTrainedModel
    Processor = Union[transformers.PreTrainedTokenizer, transformers.ProcessorMixin]
else:
    Tensor = None
    TorchDataset = None
    HFDataset = None
    DataCollator = None
    DataLoader = None
    Model = None
    Processor = None


class DatasetInfo(TypedDict, total=False):
    hf_hub_url: NotRequired[str]
    """HF hub dataset uri."""
    file_name: NotRequired[str]
    """Local file path."""
    dataset_dir: NotRequired[str]
    """Dataset directory, default to args.dataset_dir."""
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
