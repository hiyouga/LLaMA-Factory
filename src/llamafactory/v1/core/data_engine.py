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

import os
from collections.abc import AsyncIterable, Iterable
from typing import Any, Union

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from ..config.data_args import DataArguments
from ..extras.types import DatasetInfo, HFDataset, Processor, Tensor
from ..plugins.data_plugins.loader import DataGetItemPlugin, DataIndexPlugin, DataLoaderPlugin


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor, dataset_info: dict[str, DatasetInfo]) -> None:
        self.processor = processor
        self.dataset_info = dataset_info

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        pass


class DataEngine(Dataset):
    """Data engine."""

    def __init__(self, data_args: DataArguments) -> None:
        self.args = data_args
        """Data arguments."""
        self.datasets: dict[str, HFDataset] = {}
        """Dict of (dataset_name, dataset)"""
        self.dataset_info: dict[str, DatasetInfo] = {}
        """Dict of (dataset_name, dataset_info)"""
        self.streaming: bool = False
        """Whether dataset is streaming."""
        self.data_index: list[tuple[str, int]] = []
        """List of (dataset_name, sample_index)"""
        self.data_loader_plugin = DataLoaderPlugin(args=self.args)
        """Data loader plugin."""
        self.data_index_plugin = DataIndexPlugin()
        """Data index plugin."""
        self.data_getitem_plugin = DataGetItemPlugin(datasets=self.datasets, data_index=self.data_index)
        """Data getitem plugin."""
        self.get_dataset_info()
        self.load_dataset()
        self.build_data_index()

    def get_dataset_info(self) -> None:
        """Get dataset info."""
        if self.args.dataset.endswith(".yaml") and os.path.isfile(
            os.path.join(self.args.dataset_dir, self.args.dataset)
        ):  # local file
            self.dataset_info = OmegaConf.load(os.path.join(self.args.dataset_dir, self.args.dataset))
        elif self.args.dataset.endswith(".yaml"):  # hf hub uri, e.g. llamafactory/v1-sft-demo/dataset_info.yaml
            repo_id, filename = os.path.split(self.args.dataset)
            filepath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            self.dataset_info = OmegaConf.load(filepath)
        elif os.path.exists(os.path.join(self.args.dataset_dir, self.args.dataset)):  # local file(s)
            self.dataset_info = {"default": {"file_name": self.args.dataset}}
        else:  # hf hub dataset, e.g. llamafactory/v1-sft-demo
            self.dataset_info = {"default": {"hf_hub_url": self.args.dataset}}

    def load_dataset(self) -> None:
        """Load dataset from dataset info."""
        for key, value in self.dataset_info.items():
            split = value.get("split", "train")
            streaming = value.get("streaming", False)
            self.streaming |= streaming
            if "hf_hub_url" in value:
                self.datasets[key] = load_dataset(value["hf_hub_url"], split=split, streaming=streaming)
            else:  # data loader plugin
                self.datasets[key] = self.data_loader_plugin.auto_load_data(value)

    def build_data_index(self) -> None:
        """Build dataset index."""
        for dataset_name, dataset in self.datasets.items():
            size = self.dataset_info[dataset_name].get("size")
            weight = self.dataset_info[dataset_name].get("weight")
            if self.streaming:
                data_index = [(dataset_name, -1) for _ in range(1000)]
            else:
                data_index = [(dataset_name, sample_index) for sample_index in range(len(dataset))]

            if size or weight:  # data index plugin
                data_index = self.data_index_plugin.adjust_data_index(data_index, size, weight)

            self.data_index.extend(data_index)

    def get_data_collator(self, processor: Processor) -> DataCollator:
        """Get data collator.

        Args:
            processor (Processor): Processor.

        Returns:
            DataCollator: Data collator.
        """
        return DataCollator(processor=processor, dataset_info=self.dataset_info)

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: Dataset length.
        """
        if self.streaming:
            return -1
        else:
            return len(self.data_index)

    def __getitem__(self, index: Union[int, slice, list[int]]) -> Union[dict, list[dict]]:
        """Get dataset item.

        Args:
            index (int): Dataset index.

        Returns:
            dict: Dataset item.
        """
        if self.streaming:
            raise ValueError("Streaming dataset does not support index access.")

        if isinstance(index, int):
            dataset_name, sample_index = self.data_index[index]
            return {"_dataset_name": dataset_name, **self.datasets[dataset_name][sample_index]}
        else:
            return self.data_getitem_plugin.get_data(index)

    def __iter__(self) -> Iterable:
        """Get dataset iterator.

        Returns:
            Iterable: Dataset iterator.
        """
        if self.streaming:
            pass
        else:
            # TODO: add shuffle here
            pass

        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterable:
        """Get dataset async iterator.

        Returns:
            AsyncIterable: Dataset async iterator.
        """
        if self.streaming:
            pass
        else:
            # TODO: add shuffle here
            pass

        raise NotImplementedError()
