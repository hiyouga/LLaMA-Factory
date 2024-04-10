# coding=utf-8
# Calculates the distribution of the input lengths in the dataset.
# Usage: python length_cdf.py --model_name_or_path path_to_model --dataset alpaca_en --template default

from collections import defaultdict
from typing import Optional

import fire
from tqdm import tqdm

from llmtuner.data import get_dataset
from llmtuner.hparams import get_train_args
from llmtuner.model import load_tokenizer


def length_cdf(
    model_name_or_path: str,
    dataset: Optional[str] = "alpaca_en",
    dataset_dir: Optional[str] = "data",
    template: Optional[str] = "default",
    interval: Optional[int] = 1000,
):
    model_args, data_args, training_args, _, _ = get_train_args(
        dict(
            stage="sft",
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=1_000_000,
            output_dir="dummy_dir",
            overwrite_cache=True,
        )
    )
    tokenizer = load_tokenizer(model_args)
    trainset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
    total_num = len(trainset)
    length_dict = defaultdict(int)
    for sample in tqdm(trainset["input_ids"]):
        length_dict[len(sample) // interval * interval] += 1

    length_tuples = list(length_dict.items())
    length_tuples.sort()
    count_accu, prob_accu = 0, 0
    for length, count in length_tuples:
        count_accu += count
        prob_accu += count / total_num * 100
        print("{:d} ({:.2f}%) samples have length < {}.".format(count_accu, prob_accu, length + interval))


if __name__ == "__main__":
    fire.Fire(length_cdf)
