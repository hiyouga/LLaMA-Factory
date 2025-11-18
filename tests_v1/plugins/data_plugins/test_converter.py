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
from llamafactory.v1.plugins.data_plugins.converter import get_converter


@pytest.mark.parametrize("num_samples", [16])
def test_alpaca_converter(num_samples: int):
    data_args = DataArguments(dataset="llamafactory/v1-sft-demo/dataset_info.yaml")
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


def test_sharegpt_converter_invalid():
    example = {
        "conversations": [
            {
                "from": "system",
                "value": "Processes historical market data to generate trading signals "
                "based on specified technical indicators.",
            },
            {
                "from": "human",
                "value": "I possess a detailed dataset, 'Historical_Market_Data.csv'. "
                "Could you proceed with these function calls to assist me with the task?",
            },
            {
                "from": "gpt",
                "value": "```tool_call\n{'arguments': '{\"data_file\": \"Historical_Market_Data.csv\"]}', "
                "'name': 'backtest_trading_signals'}```\n",
            },
            {
                "from": "tool",
                "value": '<tool id="D2">\n{"analysis": {"RSI_signals": [{"date": "2025-01-10", '
                '"symbol": "AAPL", "signal": "Buy"}]}}}\n</tool>\n',
            },
        ]
    }
    dataset_converter = get_converter("sharegpt")
    assert dataset_converter(example) == {"messages": []}


def test_sharegpt_converter_valid():
    example = {
        "conversations": [
            {
                "from": "system",
                "value": "Processes historical market data to generate trading signals based on "
                "specified technical indicators.",
            },
            {
                "from": "human",
                "value": "I possess a detailed dataset, 'Historical_Market_Data.csv'. "
                "Could you proceed with these function calls to assist me with the task?",
            },
            {
                "from": "gpt",
                "value": "```tool_call\n{'arguments': '{\"data_file\": \"Historical_Market_Data.csv\"]}', "
                "'name': 'backtest_trading_signals'}```\n",
            },
        ]
    }
    dataset_converter = get_converter("sharegpt")
    expected_data = {
        "messages": [
            {
                "content": [
                    {
                        "type": "text",
                        "value": "Processes historical market data to generate trading signals based on "
                        "specified technical indicators.",
                    }
                ],
                "loss_weight": 0.0,
                "role": "system",
            },
            {
                "content": [
                    {
                        "type": "text",
                        "value": "I possess a detailed dataset, 'Historical_Market_Data.csv'. "
                        "Could you proceed with these function calls to assist me with the task?",
                    }
                ],
                "loss_weight": 0.0,
                "role": "user",
            },
            {
                "content": [
                    {
                        "type": "text",
                        "value": "```tool_call\n{'arguments': '{\"data_file\": \"Historical_Market_Data.csv\"]}', "
                        "'name': 'backtest_trading_signals'}```\n",
                    }
                ],
                "loss_weight": 1.0,
                "role": "assistant",
            },
        ]
    }
    assert dataset_converter(example) == expected_data


@pytest.mark.parametrize("num_samples", [16])
def test_pair_converter(num_samples: int):
    data_args = DataArguments(dataset="frozenleaves/tiny-dpo/dataset_info.yaml")
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
        assert data_engine[index] == {"_dataset_name": "dpo_zh_demo", **expected_data}


if __name__ == "__main__":
    test_alpaca_converter(1)
    test_sharegpt_converter_invalid()
    test_sharegpt_converter_valid()
    test_pair_converter(1)
