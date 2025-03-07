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

from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", Dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", Dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    dataset_dict = {}
    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                dataset_dict["validation"] = dataset.take(int(data_args.val_size))
                dataset_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset_dict = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset = dataset.train_test_split(test_size=val_size, seed=seed)
                dataset_dict = {"train": dataset["train"], "validation": dataset["test"]}
        else:
            dataset_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            dataset_dict.update({f"validation_{name}": data for name, data in eval_dataset.items()})
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            dataset_dict["validation"] = eval_dataset

    return DatasetDict(dataset_dict)


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""
    Converts dataset or dataset dict to dataset module.
    """
    dataset_module: "DatasetModule" = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module
