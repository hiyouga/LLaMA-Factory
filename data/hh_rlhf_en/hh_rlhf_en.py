# Copyright 2025 the LlamaFactory team.
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import datasets


_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://huggingface.co")
_DESCRIPTION = "Human preference data about helpfulness and harmlessness."
_CITATION = ""
_HOMEPAGE = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf"
_LICENSE = "mit"
_URL = f"{_HF_ENDPOINT}/datasets/Anthropic/hh-rlhf/resolve/main/"
_URLS = {
    "train": [
        _URL + "harmless-base/train.jsonl.gz",
        _URL + "helpful-base/train.jsonl.gz",
        _URL + "helpful-online/train.jsonl.gz",
        _URL + "helpful-rejection-sampled/train.jsonl.gz",
    ],
    "test": [
        _URL + "harmless-base/test.jsonl.gz",
        _URL + "helpful-base/test.jsonl.gz",
        _URL + "helpful-online/test.jsonl.gz",
        _URL + "helpful-rejection-sampled/test.jsonl.gz",
    ],
}


class HhRlhfEn(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "instruction": datasets.Value("string"),
                "chosen": datasets.Value("string"),
                "rejected": datasets.Value("string"),
                "history": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        file_path = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": file_path["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": file_path["test"]}),
        ]

    def _generate_examples(self, filepaths: list[str]):
        key = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                for row in f:
                    data = json.loads(row)
                    chosen = data["chosen"]
                    rejected = data["rejected"]

                    assist_idx = rejected.rfind("\n\nAssistant: ")
                    r_reject = rejected[assist_idx + 13 :].strip()
                    assist_idx = chosen.rfind("\n\nAssistant: ")
                    r_accept = chosen[assist_idx + 13 :].strip()

                    human_idx = chosen.rfind("\n\nHuman: ")
                    query = chosen[human_idx + 9 : assist_idx].strip()
                    prompt = chosen[:human_idx]
                    history = []

                    while prompt.rfind("\n\nAssistant: ") != -1:
                        assist_idx = prompt.rfind("\n\nAssistant: ")
                        human_idx = prompt.rfind("\n\nHuman: ")
                        if human_idx != -1:
                            old_query = prompt[human_idx + 9 : assist_idx].strip()
                            old_resp = prompt[assist_idx + 13 :].strip()
                            history.insert(0, (old_query, old_resp))
                        else:
                            break
                        prompt = prompt[:human_idx]

                    yield key, {"instruction": query, "chosen": r_accept, "rejected": r_reject, "history": history}
                    key += 1
