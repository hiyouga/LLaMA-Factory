import json
import datasets
from typing import Any, Dict, List


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

_HOMEPAGE = "https://huggingface.co/datasets/stingning/ultrachat"
_LICENSE = "cc-by-nc-4.0"
_BASE_DATA_URL = "https://huggingface.co/datasets/stingning/ultrachat/resolve/main/train_{idx}.jsonl"


class BelleMultiturn(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Value("string"),
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
        file_paths = [dl_manager.download(_BASE_DATA_URL.format(idx=idx)) for idx in range(9)] # multiple shards
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepaths": file_paths
                }
            )
        ]

    def _generate_examples(self, filepaths: List[str]) -> Dict[int, Dict[str, Any]]: # generate multi-turn chat for ChatGLM
        for filepath in filepaths:
            with open(filepath, "r", encoding="utf-8") as f:
                for row in f:
                    try:
                        data = json.loads(row)
                    except:
                        continue
                    key = data["id"]
                    content = data["data"]
                    if len(content) % 2 == 1:
                        content.pop(-1)
                    if len(content) < 2:
                        continue

                    query = content[-2]
                    response = content[-1]
                    history = [[content[2*i], content[2*i+1]] for i in range(len(content) // 2 - 1)]

                    yield key, {
                        "instruction": query,
                        "output": response,
                        "history": history
                    }
