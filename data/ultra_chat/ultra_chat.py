import json
import os
from typing import List

import datasets


_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")

_DESCRIPTION = "UltraChat: Large-scale, Informative, and Diverse Multi-round Dialogue Data."

_CITATION = """\
@misc{UltraChat,
  author = {Ding, Ning and Chen, Yulin and Xu, Bokai and Hu, Shengding and Qin, Yujia and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  title = {UltraChat: A Large-scale Auto-generated Multi-round Dialogue Data},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/thunlp/ultrachat}},
}
"""

_HOMEPAGE = "{}/datasets/stingning/ultrachat".format(_HF_ENDPOINT)
_LICENSE = "cc-by-nc-4.0"
_BASE_DATA_URL = "{}/datasets/stingning/ultrachat/resolve/main/train_{{idx}}.jsonl".format(_HF_ENDPOINT)


class UltraChat(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}]}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_paths = [dl_manager.download(_BASE_DATA_URL.format(idx=idx)) for idx in range(10)]  # multiple shards
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_paths})]

    def _generate_examples(self, filepaths: List[str]):
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    try:
                        data = json.loads(row)
                    except Exception:
                        continue
                    key: int = data["id"]
                    content: List[str] = data["data"]
                    if len(content) % 2 == 1:
                        content.pop(-1)
                    if len(content) < 2:
                        continue
                    conversations = [
                        {"from": "human" if i % 2 == 0 else "gpt", "value": content[i]} for i in range(len(content))
                    ]
                    yield key, {"conversations": conversations}
