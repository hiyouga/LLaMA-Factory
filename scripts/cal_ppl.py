# coding=utf-8
# Calculates the ppl of pre-trained models.
# Usage: python cal_flops.py --model_name_or_path path_to_model --batch_size 1 --seq_length 512

import json
from typing import Dict

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq

from llmtuner.data import get_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.hparams import get_train_args
from llmtuner.model import load_model, load_tokenizer


def cal_ppl(
    model_name_or_path: str,
    batch_size: int = 4,
    stage: str = "sft",
    dataset: str = "alpaca_en",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 1024,
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
    else:
        raise NotImplementedError

    dataloader = DataLoader(trainset, batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
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
            perplexities.extend(sentence_logps.exp().tolist())

    with open("ppl.json", "w", encoding="utf-8") as f:
        json.dump(perplexities, f, indent=2)

    print("Perplexities have been saved at ppl.json.")


if __name__ == "__main__":
    fire.Fire(cal_ppl)
