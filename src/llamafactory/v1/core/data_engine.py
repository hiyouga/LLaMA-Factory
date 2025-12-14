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

"""The definition of data engine.

Init Data engine:
1. Parse dataset info from arguments.
2. Load datasets according to dataset info.
3. Build data index (and reweight samples if necessary).

Get Data Sample:
1. Get sample from data index.
2. Convert sample to standard format.
3. Return sample.
"""

import os
from collections.abc import Iterable
from typing import Any, Union

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from ..config.data_args import DataArguments
from ..utils.types import DatasetInfo, HFDataset, Sample


class DataEngine(Dataset):
    """Data engine.

    Args:
        data_args: Data arguments.
    """

    def __init__(self, data_args: DataArguments) -> None:
        self.args = data_args
        """Data arguments."""
        self.datasets: dict[str, HFDataset] = {}
        """Dict of (dataset_name, dataset)"""
        self.dataset_infos: dict[str, DatasetInfo] = {}
        """Dict of (dataset_name, dataset_info)"""
        self.data_index: list[tuple[str, int]] = []
        """List of (dataset_name, sample_index)"""
        self.streaming: bool = False
        """Whether dataset is streaming."""
        self._get_dataset_info()
        self._load_dataset()
        self._build_data_index()

    def _get_dataset_info(self) -> None:
        """Get dataset info from data arguments."""
        if self.args.dataset.endswith(".yaml") and os.path.isfile(self.args.dataset):  # local file
            self.dataset_infos = OmegaConf.load(self.args.dataset)
        elif self.args.dataset.endswith(".yaml"):  # hf hub uri, e.g. llamafactory/v1-sft-demo/dataset_info.yaml
            repo_id, filename = os.path.split(self.args.dataset)
            filepath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            self.dataset_infos = OmegaConf.load(filepath)
        elif os.path.exists(self.args.dataset):  # local file(s)
            self.dataset_infos = {"default": {"path": self.args.dataset, "source": "local"}}
        else:  # hf hub dataset, e.g. llamafactory/v1-sft-demo
            self.dataset_infos = {"default": {"path": self.args.dataset}}

    def _load_dataset(self) -> None:
        """Load datasets according to dataset info."""
        for dataset_name, dataset_info in self.dataset_infos.items():
            split = dataset_info.get("split", "train")
            streaming = dataset_info.get("streaming", False)
            self.streaming |= streaming
            if dataset_info.get("source", "hf_hub") == "hf_hub":
                from datasets import load_dataset

                self.datasets[dataset_name] = load_dataset(dataset_info["path"], split=split, streaming=streaming)
            else:  # data loader plugin
                from ..plugins.data_plugins.loader import DataLoaderPlugin

                self.datasets[dataset_name] = DataLoaderPlugin(dataset_info["source"]).load(dataset_info)

    def _build_data_index(self) -> None:
        """Build dataset index."""
        for dataset_name, dataset in self.datasets.items():
            streaming = self.dataset_infos[dataset_name].get("streaming", False)
            if streaming:
                data_index = [(dataset_name, -1) for _ in range(1000)]
            else:
                data_index = [(dataset_name, sample_index) for sample_index in range(len(dataset))]

            size = self.dataset_infos[dataset_name].get("size")
            weight = self.dataset_infos[dataset_name].get("weight")
            if size or weight:  # data index plugin
                from ..plugins.data_plugins.loader import DataIndexPlugin

                data_index = DataIndexPlugin().adjust_data_index(data_index, size, weight)

            self.data_index.extend(data_index)

    def _convert_data_sample(self, raw_sample: dict[str, Any], dataset_name: str) -> Sample:
        """Convert dataset sample.

        Args:
            raw_sample (dict[str, Any]): Raw dataset sample.
            dataset_name (str): Dataset name.

        Returns:
            Sample: Dataset sample.
        """
        converter = self.dataset_infos[dataset_name].get("converter")
        if converter is not None:
            from ..plugins.data_plugins.converter import DataConverterPlugin

            return {"_dataset_name": dataset_name, **DataConverterPlugin(converter)(raw_sample)}
        else:
            return {"_dataset_name": dataset_name, **raw_sample}

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: Dataset length.
        """
        if self.streaming:
            return -1
        else:
            return len(self.data_index)

    def __getitem__(self, index: Union[int, Any]) -> Union[Sample, list[Sample]]:
        """Get dataset item.

        Args:
            index (int): Dataset index.

        Returns:
            Sample: Dataset item.
        """
        if self.streaming:
            raise ValueError("Streaming dataset does not support index access.")

        if isinstance(index, int):
            dataset_name, sample_index = self.data_index[index]
            return self._convert_data_sample(self.datasets[dataset_name][sample_index], dataset_name)
        else:  # data selector plugin
            from ..plugins.data_plugins.loader import DataSelectorPlugin

            selected_index = DataSelectorPlugin().select(self.data_index, index)
            if isinstance(selected_index, list):
                return [
                    self._convert_data_sample(self.datasets[dataset_name][sample_index], dataset_name)
                    for dataset_name, sample_index in selected_index
                ]
            else:
                dataset_name, sample_index = selected_index
                return self._convert_data_sample(self.datasets[dataset_name][sample_index], dataset_name)

    def __iter__(self) -> Iterable[Sample]:
        """Get dataset iterator.

        Returns:
            Iterable[Sample]: Dataset iterator.
        """
        # NOTE: hf iterable dataset uses worker ids while map dataset does not
        # NOTE: add worker id and shuffle to the map dataset
        # https://github.com/huggingface/datasets/blob/4.0.0/src/datasets/iterable_dataset.py#L2214

        raise NotImplementedError()


if __name__ == "__main__":
    """
    python -m llamafactory.v1.core.data_engine --model none --dataset data/v1_sft_demo.yaml
    python -m llamafactory.v1.core.data_engine --model none --dataset data/v1_dpo_demo.yaml
    """
    from ..config.arg_parser import get_args

    data_args, *_ = get_args()
    data_engine = DataEngine(data_args=data_args)
    print(data_engine[0])
