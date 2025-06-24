# Copyright 2025 the PEFTFactory team.
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


import numpy as np
import pandas as pd
from datasets import ClassLabel, load_dataset


datasets = {
    "glue": ["mnli", "qqp", "qnli", "sst2", "stsb", "mrpc", "rte", "cola"],
    "super_glue": ["record", "multirc", "boolq", "wic", "wsc.fixed", "cb", "copa"],
    "nlu_reason": ["cais/mmlu", "ybisk/piqa", "allenai/social_i_qa", "hellaswag", "winogrande", "allenai/openbookqa"],
    "math": ["allenai/math_qa", "openai/gsm8k", "ChilleD/SVAMP"],
    "code": ["neulab/conala", "Abzu/CodeAlpacaPython", "codeparrot/apps"],
}

valid_mapping = {
    "mnli": "validation_matched",
    "openai/gsm8k": "test",
    "ChilleD/SVAMP": "test",
    "neulab/conala": "test",
    "Abzu/CodeAlpacaPython": "test",
    "codeparrot/apps": "test",
}

train_mapping = {"cais/mmlu": "auxiliary_train"}

label_mapping = {
    "record": "answers",
    "cais/mmlu": "answer",
    "winogrande": "answer",
    "allenai/openbookqa": "answerKey",
    "allenai/math_qa": "correct",
    "openai/gsm8k": "answer",
    "ChilleD/SVAMP": "Answer",
    "neulab/conala": "snippet",
    "Abzu/CodeAlpacaPython": "response",
    "codeparrot/apps": "solutions",
}


dataset_info = {}

for b in datasets:
    for d in datasets[b]:
        if b == "glue" or b == "super_glue":
            loaded_dataset = load_dataset(b, d, trust_remote_code=True)
        elif d == "cais/mmlu":
            loaded_dataset = load_dataset(d, "all", trust_remote_code=True)
        elif d == "winogrande":
            loaded_dataset = load_dataset(d, "winogrande_xl", trust_remote_code=True)
        elif d == "openai/gsm8k":
            loaded_dataset = load_dataset(d, "main", trust_remote_code=True)
        elif d == "codeparrot/apps":
            loaded_dataset = load_dataset(d, "all", trust_remote_code=True)
        else:
            loaded_dataset = load_dataset(d, trust_remote_code=True)

        dataset_info[d] = {}
        print()
        print(d)
        print(loaded_dataset)

        train_size = len(loaded_dataset[train_mapping.get(d, "train")])
        valid_size = len(loaded_dataset[valid_mapping.get(d, "validation")])
        print(train_size, valid_size)

        if label_mapping.get(d, "label") in loaded_dataset[train_mapping.get(d, "train")].features and isinstance(
            loaded_dataset[train_mapping.get(d, "train")].features[label_mapping.get(d, "label")], ClassLabel
        ):
            print(loaded_dataset[train_mapping.get(d, "train")].features[label_mapping.get(d, "label")])

            labels = set(loaded_dataset[train_mapping.get(d, "train")].features[label_mapping.get(d, "label")].names)
            dataset_info[d]["labels"] = labels
            dataset_info[d]["n_labels"] = len(labels)
        else:
            labels = loaded_dataset[train_mapping.get(d, "train")][label_mapping.get(d, "label")]

            if isinstance(labels[0], list):
                print("N/A")
                dataset_info[d]["labels"] = np.nan
                dataset_info[d]["n_labels"] = np.nan
            else:
                labels = set(labels)
                if len(labels) < 100:
                    print(labels)
                    dataset_info[d]["labels"] = labels
                    dataset_info[d]["n_labels"] = len(labels)
                else:
                    print("N/A")
                    dataset_info[d]["labels"] = np.nan
                    dataset_info[d]["n_labels"] = np.nan

        dataset_info[d]["train_size"] = train_size
        dataset_info[d]["valid_size"] = valid_size


df = pd.DataFrame(dataset_info).T
print(df)
print(df.to_latex())
