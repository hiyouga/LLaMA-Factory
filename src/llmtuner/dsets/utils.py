from typing import Dict
from datasets import Dataset


def split_dataset(
    dataset: Dataset, dev_ratio: float, do_train: bool
) -> Dict[str, Dataset]:
    # Split the dataset
    if do_train:
        if dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=dev_ratio)
            return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}
