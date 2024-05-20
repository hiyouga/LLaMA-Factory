# coding=utf-8
# Calculates the ppl on the dataset of the pre-trained models.
# Usage: python cal_ppl.py --model_name_or_path path_to_model --save_name ppl.json

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from llamafactory.data import get_dataset
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    train_on_prompt: bool = False

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        chosen_features = []
        for feature in features:
            prompt_len, answer_len = len(feature["prompt_ids"]), len(feature["chosen_ids"])
            input_ids = feature["prompt_ids"] + feature["chosen_ids"]
            attention_mask = [1] * (prompt_len + answer_len)
            labels = input_ids if self.train_on_prompt else [IGNORE_INDEX] * prompt_len + feature["chosen_ids"]
            chosen_features.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

        return super().__call__(chosen_features)


def cal_ppl(
    model_name_or_path: str,
    save_name: str,
    batch_size: int = 4,
    stage: Literal["pt", "sft", "rm"] = "sft",
    dataset: str = "alpaca_en",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 1024,
    max_samples: Optional[int] = None,
    train_on_prompt: bool = False,
):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        dict(
            stage=stage,
            model_name_or_path=model_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            train_on_prompt=train_on_prompt,
            output_dir="dummy_dir",
            overwrite_cache=True,
        )
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    trainset = get_dataset(model_args, data_args, training_args, stage, **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    if stage == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif stage == "sft":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX)
    elif stage == "rm":
        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer, label_pad_token_id=IGNORE_INDEX, train_on_prompt=train_on_prompt
        )
    else:
        raise NotImplementedError

    dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_ppl = 0
    perplexities = []
    batch: Dict[str, "torch.Tensor"]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)
            outputs = model(**batch)
            shift_logits: "torch.Tensor" = outputs["logits"][..., :-1, :]
            shift_labels: "torch.Tensor" = batch["labels"][..., 1:]
            loss_mask = shift_labels != IGNORE_INDEX
            flatten_logits = shift_logits.contiguous().view(shift_labels.size(0) * shift_labels.size(1), -1)
            flatten_labels = shift_labels.contiguous().view(-1)
            token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels)
            token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)
            sentence_logps = (token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            total_ppl += sentence_logps.exp().sum().item()
            perplexities.extend(sentence_logps.exp().tolist())

    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(perplexities, f, indent=2)

    print("Average perplexity is {:.2f}".format(total_ppl / len(perplexities)))
    print("Perplexities have been saved at {}.".format(save_name))


if __name__ == "__main__":
    fire.Fire(cal_ppl)
