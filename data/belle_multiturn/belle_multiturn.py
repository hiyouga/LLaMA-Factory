import json
import datasets
from typing import Any, Dict, List


_DESCRIPTION = "BELLE multiturn chat dataset."

_CITATION = """\
@article{belle2023exploring,
  title={Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases},
  author={Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, Xiangang Li},
  journal={arXiv preprint arXiv:2303.14742},
  year={2023}
}
"""

_HOMEPAGE = "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M"
_LICENSE = "gpl-3.0"
_URL = "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/multiturn_chat_0.8M.json"


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
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]: # generate multi-turn chat with history
        with open(filepath, "r", encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                prompt = data["instruction"].strip()
                response = data["output"].strip()

                assist_idx = prompt.rfind("Assistant:")
                human_idx = prompt.rfind("Human:")
                query = prompt[human_idx+6:assist_idx].strip()
                prompt = prompt[:human_idx].strip()
                history = []

                while prompt.rfind("Assistant:") != -1:
                    assist_idx = prompt.rfind("Assistant:")
                    human_idx = prompt.rfind("Human:")
                    if human_idx != -1:
                        old_query = prompt[human_idx+6:assist_idx].strip()
                        old_resp = prompt[assist_idx+10:].strip()
                        history.insert(0, (old_query, old_resp))
                    else:
                        break
                    prompt = prompt[:human_idx].strip()

                yield key, {
                    "instruction": query,
                    "output": response,
                    "history": history
                }
