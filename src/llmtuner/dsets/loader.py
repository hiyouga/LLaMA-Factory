import os
import hashlib
from typing import List

from datasets import Dataset, concatenate_datasets, load_dataset

from llmtuner.extras.logging import get_logger
from llmtuner.hparams import ModelArguments, DataArguments


logger = get_logger(__name__)


def get_dataset(
    model_args: ModelArguments,
    data_args: DataArguments
) -> Dataset:

    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    ext2type = {
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "txt": "text"
    }

    max_samples = data_args.max_samples
    all_datasets: List[Dataset] = [] # support multiple datasets

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

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))

                    if data_path is None:
                        data_path = ext2type.get(data_files[0].split(".")[-1], None)
                    else:
                        assert data_path == ext2type.get(data_files[-1].split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = ext2type.get(data_files[0].split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."

            if len(data_files) == 1 and dataset_attr.dataset_sha1 is not None:
                checksum(data_files[0], dataset_attr.dataset_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json or too many files.")
        else:
            raise NotImplementedError

        raw_datasets = load_dataset(
            data_path,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None
        )
        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        prefix_data = [dataset_attr.source_prefix] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_attr, column_name) != target_name:
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)
        dataset = dataset.add_column("prefix", prefix_data)
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)

    return all_datasets
