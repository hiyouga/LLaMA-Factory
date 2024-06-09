import os

import pytest
from datasets import load_dataset

from llamafactory.data import get_dataset
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


TINY_LLAMA = os.environ.get("TINY_LLAMA", "llamafactory/tiny-random-LlamaForCausalLM")

TRAINING_ARGS = {
    "model_name_or_path": TINY_LLAMA,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "dataset": "llamafactory/tiny_dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_cache": True,
    "output_dir": "dummy_dir",
    "overwrite_output_dir": True,
    "fp16": True,
}


@pytest.mark.parametrize("test_num", [5])
def test_supervised(test_num: int):
    model_args, data_args, training_args, _, _ = get_train_args(TRAINING_ARGS)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    tokenized_data = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)

    original_data = load_dataset(TRAINING_ARGS["dataset"], split="train")
    for test_idx in range(test_num):
        decode_result = tokenizer.decode(tokenized_data["input_ids"][test_idx])
        messages = [
            {"role": "user", "content": original_data[test_idx]["instruction"]},
            {"role": "assistant", "content": original_data[test_idx]["output"]},
        ]
        templated_result = tokenizer.apply_chat_template(messages, tokenize=False)
        assert decode_result == templated_result
