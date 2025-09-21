from datasets import load_dataset
from typing import Any
from pathlib import Path


def squad_v2_preprocess(
    splits: str | list[str] = ["train", "validation"],
    cache_dir: str = "./.cache",
    data_dir: str = "./data",
    redownload: bool = False,
):
    if isinstance(splits, list):
        for split in splits:
            squad_v2_preprocess(split, cache_dir=cache_dir, data_dir=data_dir, redownload=redownload)
    else:
        data = load_dataset(
            "rajpurkar/squad_v2",
            cache_dir=cache_dir,
            split=splits,
            download_mode="force_redownload" if redownload else "reuse_dataset_if_exists",
        )

        def helper(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return {
                "prompt": examples["question"],
                "query": examples["context"],
                "response": [answer["text"] for answer in examples["answers"]],
            }

        dataset = data.map(helper, batched=True, remove_columns=data.column_names)

        save_path = Path(data_dir) / "squad_v2" / f"{splits}.json"

        with save_path.open("w") as f:
            dataset.to_json(f, orient="records")


squad_v2_preprocess()
