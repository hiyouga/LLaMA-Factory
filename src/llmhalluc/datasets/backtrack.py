from typing import Any
from datasets import load_dataset
from pathlib import Path


def backtrack_processor(example: dict[str, list[Any]]) -> dict[str, list[Any]]:
    prompt = example["prompt"]
    query = example["query"]
    response = example["response"]
