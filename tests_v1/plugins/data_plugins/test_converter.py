# Copyright 2025 the LlamaFactory team.
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

import random

import pytest
from datasets import load_dataset

from llamafactory.v1.config.data_args import DataArguments
from llamafactory.v1.core.data_engine import DataEngine
from llamafactory.v1.plugins.data_plugins.converter import DataConverterPlugin


@pytest.mark.parametrize("num_samples", [16])
def test_alpaca_converter(num_samples: int):
    data_args = DataArguments(dataset="llamafactory/v1-dataset-info/tiny-supervised-dataset.yaml")
    data_engine = DataEngine(data_args)
    original_data = load_dataset("llamafactory/tiny-supervised-dataset", split="train")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        expected_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": original_data[index]["instruction"] + original_data[index]["input"]}
                    ],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["output"]}],
                    "loss_weight": 1.0,
                },
            ]
        }
        assert data_engine[index] == {"_dataset_name": "tiny_dataset", **expected_data}


def test_sharegpt_converter():
    example = {
        "conversations": [
            {"from": "system", "value": "System"},
            {"from": "human", "value": "User"},
            {"from": "function_call", "value": "Tool"},
            {"from": "observation", "value": "Observation"},
            {"from": "gpt", "value": "Assistant"},
        ]
    }
    expected_data = {
        "messages": [
            {"content": [{"type": "text", "value": "System"}], "loss_weight": 0.0, "role": "system"},
            {"content": [{"type": "text", "value": "User"}], "loss_weight": 0.0, "role": "user"},
            {"content": [{"type": "tool_calls", "value": "Tool"}], "loss_weight": 1.0, "role": "assistant"},
            {"content": [{"type": "text", "value": "Observation"}], "loss_weight": 0.0, "role": "tool"},
            {"content": [{"type": "text", "value": "Assistant"}], "loss_weight": 1.0, "role": "assistant"},
        ]
    }
    assert DataConverterPlugin("sharegpt")(example) == expected_data


@pytest.mark.parametrize("num_samples", [16])
def test_pair_converter(num_samples: int):
    data_args = DataArguments(dataset="llamafactory/v1-dataset-info/orca-dpo-pairs.yaml")
    data_engine = DataEngine(data_args)
    original_data = load_dataset("HuggingFaceH4/orca_dpo_pairs", split="train_prefs")
    indexes = random.choices(range(len(data_engine)), k=num_samples)
    for index in indexes:
        print(data_engine[index])
        print(original_data[index])
        expected_data = {
            "chosen_messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][0]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][1]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["chosen"][2]["content"]}],
                    "loss_weight": 1.0,
                },
            ],
            "rejected_messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][0]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][1]["content"]}],
                    "loss_weight": 0.0,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": original_data[index]["rejected"][2]["content"]}],
                    "loss_weight": 1.0,
                },
            ],
        }
        assert data_engine[index] == {"_dataset_name": "tiny_dataset", **expected_data}


if __name__ == "__main__":
    test_alpaca_converter(1)
    test_sharegpt_converter()
    test_pair_converter(1)
