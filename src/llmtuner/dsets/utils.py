import hashlib
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from llmtuner.extras.logging import get_logger

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import TrainingArguments
    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


EXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
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


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "TrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6: # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}
