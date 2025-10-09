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
from collections.abc import AsyncIterator, Iterator
from typing import Literal, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from ..config.data_args import DataArguments
from ..extras.types import DatasetInfo, HFDataset, Processor


class DataCollator:
    """Default Data collator."""

    def __init__(self, processor: Processor) -> None:
        self.processor = processor


class DatasetPathMixin:
    """Path utilities."""

    args: DataArguments
    """Data arguments."""

    def _abspath(self, path: str, dataset_dir: Optional[str] = None) -> str:
        """Get absolute path of dataset.

        Args:
            path (str): Dataset path.
            dataset_dir (Optional[str], optional): Dataset directory. Defaults to None.

        Returns:
            str: Absolute path of dataset.
        """
        dataset_dir = dataset_dir or self.args.dataset_dir
        return os.path.abspath(os.path.expanduser(os.path.join(dataset_dir, path)))

    def _exists(self, path: str, dataset_dir: Optional[str] = None) -> bool:
        """Check if dataset exists.

        Args:
            path (str): Dataset path.
            dataset_dir (Optional[str], optional): Dataset directory. Defaults to None.

        Returns:
            bool: Whether dataset exists.
        """
        return os.path.exists(self._abspath(path, dataset_dir))

    def _isfile(self, path: str, dataset_dir: Optional[str] = None) -> bool:
        """Check if dataset is a file.

        Args:
            path (str): Dataset path.
            dataset_dir (Optional[str], optional): Dataset directory. Defaults to None.

        Returns:
            bool: Whether dataset is a file.
        """
        return os.path.isfile(self._abspath(path, dataset_dir))

    def _isdir(self, path: str, dataset_dir: Optional[str] = None) -> bool:
        """Check if dataset is a directory.

        Args:
            path (str): Dataset path.
            dataset_dir (Optional[str], optional): Dataset directory. Defaults to None.

        Returns:
            bool: Whether dataset is a directory.
        """
        return os.path.isdir(self._abspath(path, dataset_dir))

    def _get_builder_name(self, path: str) -> Literal["arrow", "csv", "json", "parquet", "text"]:
        """Get dataset builder name.

        Args:
            path (str): Dataset path.

        Returns:
            Literal["arrow", "csv", "json", "parquet", "text"]: Dataset builder name.
        """
        return os.path.splitext(path)[-1][1:].replace("jsonl", "json").replace("txt", "text")


class DataEngine(Dataset, DatasetPathMixin):
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
        self.get_dataset_info()
        self.load_dataset()
        self.build_data_index()

    def get_dataset_info(self) -> None:
        """Get dataset info."""
        if self.args.dataset.endswith(".yaml") and self._isfile(self.args.dataset):  # local file
            self.dataset_info = OmegaConf.load(self._abspath(self.args.dataset))
        elif self.args.dataset.endswith(".yaml"):  # hf hub uri, e.g. llamafactory/v1-sft-demo/dataset_info.yaml
            repo_id, filename = os.path.split(self.args.dataset)
            filepath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            self.dataset_info = OmegaConf.load(filepath)
        elif self._exists(self.args.dataset):  # local file(s)
            self.dataset_info = {"default": {"file_name": self.args.dataset}}
        else:  # hf hub dataset, e.g. llamafactory/v1-sft-demo
            self.dataset_info = {"default": {"hf_hub_url": self.args.dataset}}

    def load_dataset(self) -> None:
        """Load dataset from dataset info."""
        for key, value in self.dataset_info.items():
            dataset_dir = value.get("dataset_dir", self.args.dataset_dir)
            split = value.get("split", "train")
            streaming = value.get("streaming", False)
            self.streaming |= streaming
            if "hf_hub_url" in value:
                self.datasets[key] = load_dataset(value["hf_hub_url"], split=split, streaming=streaming)
            elif "file_name" in value:
                filepath = self._abspath(value["file_name"], dataset_dir)
                if os.path.isdir(filepath):
                    filetype = self._get_builder_name(os.listdir(filepath)[0])
                    self.datasets[key] = load_dataset(filetype, data_dir=filepath, split=split)
                elif os.path.isfile(filepath):
                    filetype = self._get_builder_name(filepath)
                    self.datasets[key] = load_dataset(filetype, data_files=filepath, split=split)
                else:
                    raise ValueError(f"Can not load dataset {key} from {filepath}.")

                if streaming:
                    self.datasets[key] = self.datasets[key].to_iterable_dataset()
            else:
                # TODO: support dataset loader plugins
                raise ValueError(f"Dataset {key} is not supported.")

    def build_data_index(self) -> None:
        """Build dataset index."""
        for dataset_name, dataset in self.datasets.items():
            if self.streaming:
                self.data_index.append((dataset_name, -1))
            else:
                # TODO: add sample_num, weight
                self.data_index.extend([(dataset_name, sample_index) for sample_index in range(len(dataset))])

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: Dataset length.
        """
        if self.streaming:
            return -1
        else:
            return len(self.data_index)

    def __getitem__(self, index: int) -> dict:
        """Get dataset item.

        Args:
            index (int): Dataset index.

        Returns:
            dict: Dataset item.
        """
        dataset_name, sample_index = self.data_index[index]
        return self.datasets[dataset_name][sample_index]

    def __iter__(self) -> Iterator:
        """Get dataset iterator.

        Returns:
            Iterator: Dataset iterator.
        """
        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterator:
        """Get dataset async iterator.

        Returns:
            AsyncIterator: Dataset async iterator.
        """
        raise NotImplementedError()
