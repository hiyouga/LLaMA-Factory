import os
import hashlib
from typing import TYPE_CHECKING, List, Optional

from datasets import Value, concatenate_datasets, interleave_datasets, load_dataset

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset
    from llmtuner.hparams import ModelArguments, DataArguments


logger = get_logger(__name__)


EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "txt": "text"
}


def checksum(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")
        return

    if len(data_files) != 1:
        logger.warning("Checksum failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning("Checksum failed: mismatched SHA-1 hash value at {}.".format(data_files[0]))


def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> "Dataset":
    max_samples = data_args.max_samples
    all_datasets: List["Dataset"] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:
        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # directory
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))
                    if data_path is None:
                        data_path = EXT2TYPE.get(file_name.split(".")[-1], None)
                    else:
                        assert data_path == EXT2TYPE.get(file_name.split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)): # single file
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = EXT2TYPE.get(dataset_attr.dataset_name.split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."
            checksum(data_files, dataset_attr.dataset_sha1)
        else:
            raise NotImplementedError

        dataset = load_dataset(
            data_path,
            data_files=data_files,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
            streaming=data_args.streaming,
            use_auth_token=True if model_args.use_auth_token else None
        )

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        for column_name in ["prompt", "query", "response", "history"]: # align datasets
            if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
                dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)

        if dataset_attr.source_prefix: # add prefix
            features = None
            if data_args.streaming:
                features = dataset.features
                features["prefix"] = Value(dtype="string", id=None)
            dataset = dataset.map(lambda _: {"prefix": dataset_attr.source_prefix}, features=features)

        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        stopping_strategy = "first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        return interleave_datasets(all_datasets, stopping_strategy=stopping_strategy)
    else:
        raise ValueError("Unknown mixing strategy.")
