from datasets import load_dataset
from typing import Any
from pathlib import Path

from llmhalluc.prompts.QAPrompt import QA_INSTRUCTION


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
        prompt = QA_INSTRUCTION

        def helper(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
            return {
                "prompt": [prompt for _ in examples["context"]],
                "query": [
                    f"Context: {context}\nQuestion: {question}"
                    for context, question in zip(examples["context"], examples["question"])
                ],
                "response": [
                    answer["text"][0] if len(answer["text"]) > 0 else "IDK" for answer in examples["answers"]
                ],
            }

        dataset = data.map(helper, batched=True, remove_columns=data.column_names)

        save_path = Path(data_dir) / "squad_v2" / f"{splits}.json"

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.to_json(str(save_path), orient="records")


squad_v2_preprocess()
