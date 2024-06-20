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

from llamafactory.data import get_dataset
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 8192,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("num_samples", [16])
def test_supervised(num_samples: int):
    model_args, data_args, training_args, _, _ = get_train_args(TRAIN_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenized_data = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)

    ref_tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA)

    original_data = load_dataset(TRAIN_ARGS["dataset"], split="train")
    indexes = random.choices(range(len(original_data)), k=num_samples)
    for index in indexes:
        prompt = original_data[index]["instruction"]
        if original_data[index]["input"]:
            prompt += "\n" + original_data[index]["input"]

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": original_data[index]["output"]},
        ]
        templated_result = ref_tokenizer.apply_chat_template(messages, tokenize=False)
        decoded_result = tokenizer.decode(tokenized_data["input_ids"][index])
        assert templated_result == decoded_result
