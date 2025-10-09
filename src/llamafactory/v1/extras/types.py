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
    from datasets import Dataset as HFArrowDataset
    from datasets import IterableDataset as HFIterableDataset
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchArrowDataset
    from torch.utils.data import IterableDataset as TorchIterableDataset
    from transformers import DataCollator as HFDataCollator
    from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    TorchDataset = Union[TorchArrowDataset, TorchIterableDataset]
    HFDataset = Union[HFArrowDataset, HFIterableDataset]
    DataCollator = HFDataCollator
    DataLoader = TorchDataLoader
    Model = PreTrainedModel
    Processor = Union[PreTrainedTokenizer, ProcessorMixin]
else:
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
    """Dataset directory."""
    split: NotRequired[str]
    """Dataset split."""
    converter: NotRequired[str]
    """Dataset converter."""
    num_samples: NotRequired[int]
    """Number of samples."""
    weight: NotRequired[float]
    """Dataset weight."""
    streaming: NotRequired[bool]
    """Is streaming dataset."""
