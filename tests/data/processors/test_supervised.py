# Copyright 2024 the LlamaFactory team.
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

import os
import random

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.test_utils import load_train_dataset


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA = os.getenv("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TINY_DATA = os.getenv("TINY_DATA", "llamafactory/tiny-supervised-dataset")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "template": "llama3",
    "cutoff_len": 8192,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("num_samples", [16])
def test_supervised_single_turn(num_samples: int):
    train_dataset = load_train_dataset(dataset_dir="ONLINE", dataset=TINY_DATA, **TRAIN_ARGS)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    original_data = load_dataset(TINY_DATA, split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        prompt = original_data["instruction"][index]
        if original_data["input"][index]:
            prompt += "\n" + original_data["input"][index]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": original_data["output"][index]},
        ]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        assert train_dataset["input_ids"][index] == ref_input_ids


@pytest.mark.parametrize("num_samples", [8])
def test_supervised_multi_turn(num_samples: int):
    train_dataset = load_train_dataset(dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", **TRAIN_ARGS)
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_input_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        assert train_dataset["input_ids"][index] == ref_input_ids


@pytest.mark.parametrize("num_samples", [4])
def test_supervised_train_on_prompt(num_samples: int):
    train_dataset = load_train_dataset(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", train_on_prompt=True, **TRAIN_ARGS
    )
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        ref_ids = ref_tokenizer.apply_chat_template(original_data["messages"][index])
        assert train_dataset["input_ids"][index] == ref_ids
        assert train_dataset["labels"][index] == ref_ids


@pytest.mark.parametrize("num_samples", [4])
def test_supervised_mask_history(num_samples: int):
    train_dataset = load_train_dataset(
        dataset_dir="REMOTE:" + DEMO_DATA, dataset="system_chat", mask_history=True, **TRAIN_ARGS
    )
    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)
    original_data = load_dataset(DEMO_DATA, name="system_chat", split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        messages = original_data["messages"][index]
        ref_input_ids = ref_tokenizer.apply_chat_template(messages)
        prompt_len = len(ref_tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True))
        ref_label_ids = [IGNORE_INDEX] * prompt_len + ref_input_ids[prompt_len:]
        assert train_dataset["input_ids"][index] == ref_input_ids
        assert train_dataset["labels"][index] == ref_label_ids
