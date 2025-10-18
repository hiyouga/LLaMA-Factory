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
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from datasets import load_dataset

from ...config.data_args import DataArguments
from ...extras.types import DatasetInfo, HFDataset


@dataclass
class DataLoaderPlugin:
    """Plugin for loading dataset."""

    args: DataArguments
    """Data arguments."""

    def _get_builder_name(self, path: str) -> Literal["arrow", "csv", "json", "parquet", "text"]:
        """Get dataset builder name.

        Args:
            path (str): Dataset path.

        Returns:
            Literal["arrow", "csv", "json", "parquet", "text"]: Dataset builder name.
        """
        return os.path.splitext(path)[-1][1:].replace("jsonl", "json").replace("txt", "text")

    def auto_load_data(self, dataset_info: DatasetInfo) -> HFDataset:
        dataset_dir = dataset_info.get("dataset_dir", self.args.dataset_dir)
        split = dataset_info.get("split", "train")
        streaming = dataset_info.get("streaming", False)
        if "file_name" in dataset_info:
            filepath = os.path.join(dataset_dir, dataset_info["file_name"])
            return self.load_data_from_file(filepath, split, streaming)
        else:
            raise NotImplementedError()

    def load_data_from_file(self, filepath: str, split: str, streaming: bool) -> HFDataset:
        if os.path.isdir(filepath):
            filetype = self._get_builder_name(os.listdir(filepath)[0])
            dataset = load_dataset(filetype, data_dir=filepath, split=split)
        elif os.path.isfile(filepath):
            filetype = self._get_builder_name(filepath)
            dataset = load_dataset(filetype, data_files=filepath, split=split)
        else:
            raise ValueError(f"Can not load dataset from {filepath}.")

        if streaming:
            dataset = dataset.to_iterable_dataset()

        return dataset


@dataclass
class DataIndexPlugin:
    """Plugin for adjusting dataset index."""

    def adjust_data_index(
        self, data_index: list[tuple[str, int]], size: Optional[int], weight: Optional[float]
    ) -> list[tuple[str, int]]:
        """Adjust dataset index by size and weight.

        Args:
            data_index (list[tuple[str, int]]): List of (dataset_name, sample_index).
            size (Optional[int]): Desired dataset size.
            weight (Optional[float]): Desired dataset weight.

        Returns:
            list[tuple[str, int]]: Adjusted dataset index.
        """
        if size is not None:
            data_index = self.adjust_by_size(data_index, size)

        if weight is not None:
            data_index = self.adjust_by_weight(data_index, weight)

        return data_index

    def adjust_by_size(self, data_index: list[tuple[str, int]], size: int) -> list[tuple[str, int]]:
        raise NotImplementedError()

    def adjust_by_weight(self, data_index: list[tuple[str, int]], weight: float) -> list[tuple[str, int]]:
        raise NotImplementedError()


@dataclass
class DataSelectorPlugin:
    """Plugin for selecting dataset samples."""

    data_index: list[tuple[str, int]]
    """List of (dataset_name, sample_index)"""

    def select(self, index: Union[slice, list[int], Any]) -> Union[tuple[str, int], list[tuple[str, int]]]:
        """Select dataset samples.

        Args:
            index (Union[slice, list[int], Any]): Index of dataset samples.

        Returns:
            Union[tuple[str, int], list[tuple[str, int]]]: Selected dataset samples.
        """
        if isinstance(index, slice):
            return [self.data_index[i] for i in range(*index.indices(len(self.data_index)))]
        elif isinstance(index, list):
            return [self.data_index[i] for i in index]
        else:
            raise ValueError(f"Invalid index type {type(index)}.")
