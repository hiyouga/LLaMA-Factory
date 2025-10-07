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

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from ..config.data_args import DataArguments
from ..extras.types import DataLoader, Dataset, Processor


class DataCollator:
    def __init__(self, processor: Processor) -> None:
        self.processor = processor


class DatasetPathMixin:
    args: DataArguments

    def _abspath(self, path: str) -> str:
        return os.path.abspath(os.path.join(self.args.dataset_dir, path))

    def _exists(self, path: str) -> bool:
        return os.path.exists(self._abspath(path))

    def _isfile(self, path: str) -> bool:
        return os.path.isfile(self._abspath(path))


class DataEngine(DatasetPathMixin):
    def __init__(self, data_args: DataArguments) -> None:
        self.args = data_args
        self.datasets: dict[str, Dataset] = {}
        dataset_info = self.get_dataset_info()
        self.load_dataset(dataset_info)


    def get_dataset_info(self) -> dict:
        """Get dataset info from dataset path.

        Returns:
            dict: Dataset info.
        """
        if self.args.dataset.endswith(".yaml") and self._isfile(self.args.dataset):  # local file
            return OmegaConf.load(self._abspath(self.args.dataset))
        elif self.args.dataset.endswith(".yaml"):  # hf hub uri
            repo_id, filename = os.path.split(self.args.dataset)
            filepath = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            return OmegaConf.load(filepath)
        elif self._exists(self.args.dataset):  # local file(s)
            return {"default": {"file_name": self.args.dataset}}
        else:  # hf hub dataset
            return {"default": {"hf_hub_url": self.args.dataset}}

    def load_dataset(self, dataset_info: dict) -> None:
        for key, value in dataset_info.items():
            if "hf_hub_url" in value:
                dataset_info[key] = load_dataset(value["hf_hub_url"])
            elif "file_name" in value:
                dataset_info[key] = load_dataset(value["file_name"])

    def get_data_loader(self, processor: Processor) -> DataLoader:
        pass
