import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "Human preference data about helpfulness and harmlessness for ChatGLM."
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/Anthropic/hh-rlhf"
_LICENSE = "mit"
_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
_URLS = {
    "train": [
        _URL + "harmless-base/train.jsonl.gz",
        _URL + "helpful-base/train.jsonl.gz",
        _URL + "helpful-online/train.jsonl.gz",
        _URL + "helpful-rejection-sampled/train.jsonl.gz"
    ],
    "test": [
        _URL + "harmless-base/test.jsonl.gz",
        _URL + "helpful-base/test.jsonl.gz",
        _URL + "helpful-online/test.jsonl.gz",
        _URL + "helpful-rejection-sampled/test.jsonl.gz"
    ]
}


class HhRlhfEn(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Sequence(datasets.Value("string")),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": file_path["train"]
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": file_path["test"]
                }
            )
        ]

    def _generate_examples(self, filepaths: List[str]) -> Dict[int, Dict[str, Any]]: # generate multi-turn chat for ChatGLM
        key = 0
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    assist_idx = rejected.rfind("\n\nAssistant: ")
                    r_reject = rejected[assist_idx+13:].strip()
                    assist_idx = chosen.rfind("\n\nAssistant: ")
                    r_accept = chosen[assist_idx+13:].strip()

                    human_idx = chosen.rfind("\n\nHuman: ")
                    query = chosen[human_idx+9:assist_idx].strip()
                    prompt = chosen[:human_idx]
                    history = []

                    while prompt.rfind("\n\nAssistant: ") != -1:
                        assist_idx = prompt.rfind("\n\nAssistant: ")
                        human_idx = prompt.rfind("\n\nHuman: ")
                        if human_idx != -1:
                            old_query = prompt[human_idx+9:assist_idx].strip()
                            old_resp = prompt[assist_idx+13:].strip()
                            history.insert(0, (old_query, old_resp))
                        else:
                            break
                        prompt = prompt[:human_idx]

                    yield key, {
                        "instruction": query,
                        "output": [r_accept, r_reject],
                        "history": history
                    }
                    key += 1
