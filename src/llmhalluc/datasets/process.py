from datasets import Dataset, DatasetDict
from typing import Callable, Any
from pathlib import Path


def process_dataset(
    dataset: Dataset | DatasetDict,
    processor: Callable,
    data_dir: str = "./data",
    split: str | list[str] = "train",
    dataset_kwargs: dict[str, Any] = {"batched": False},
):
    if isinstance(split, list):
        for split in split:
            process_dataset(dataset, processor, data_dir, split)
    else:
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        processed_dataset = dataset.map(processor, **dataset_kwargs)
        save_path = Path(data_dir) / dataset.name / f"{split}.json"
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        processed_dataset.to_json(str(save_path), orient="records")
