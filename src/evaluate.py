# coding=utf-8
# Evaluates the performance of pre-trained models.
# Usage: python evaluate.py --model_name_or_path path_to_model --checkpoint_dir path_to_ckpt --template vanilla
#                           --task ceval --split validation --lang zh --n_shot 5 --batch_size 4 --save_name result
# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import os
import fire
import json
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm, trange
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

from llmtuner import ChatModel

if TYPE_CHECKING:
    from datasets import Dataset


choices = ["A", "B", "C", "D"]


@dataclass
class EvalTemplate:

    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(
        self,
        example: Dict[str, str]
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in choices if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        subject_name: str,
        use_history: bool
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example(target_data)
        history = [self.parse_example(support_set[k]) for k in range(len(support_set))]

        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.system.format(subject=subject_name) + temp[0], temp[1]))
        else:
            query = self.system.format(subject=subject_name) + query

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates = {
    "en": EvalTemplate(
        system="The following are multiple choice questions (with answers) about {subject}.\n\n",
        choice="\n{choice}. {content}",
        answer="\nAnswer: ",
        prefix=" "
    ),
    "zh": EvalTemplate(
        system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
        choice="\n{choice}. {content}",
        answer="\n答案：",
        prefix="\n"
    )
}


@torch.inference_mode()
def batch_inference(
    chat_model: ChatModel,
    batch_input: Dict[str, torch.Tensor],
    prefix_char: str
) -> List[str]:
    logits = chat_model.model(**batch_input).logits
    lengths = torch.sum(batch_input["attention_mask"], dim=-1)
    nextword_logits = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
    probs = torch.nn.functional.softmax(
        torch.stack(
            [
                nextword_logits[:, chat_model.tokenizer.encode(prefix_char + choice, add_special_tokens=False)[-1]]
                for choice in choices
            ],
            dim=-1
        ),
        dim=-1
    ).detach()
    return [chr(ord("A") + offset.item()) for offset in torch.argmax(probs, dim=-1)]


def evaluate(
    model_name_or_path: str,
    finetuning_type: Optional[str] = "lora",
    checkpoint_dir: Optional[str] = None,
    template: Optional[str] = "vanilla",
    task: Optional[str] = "ceval",
    dataset_dir: Optional[str] = "evaluation",
    split: Optional[Literal["validation", "test"]] = "validation",
    lang: Optional[Literal["zh", "en"]] = "zh",
    n_shot: Optional[int] = 5,
    n_avg: Optional[int] = 1,
    batch_size: Optional[int] = 4,
    save_name: Optional[str] = None
):
    with open(os.path.join(dataset_dir, task, "mapping.json"), "r", encoding="utf-8") as f:
        categorys: Dict[str, Dict[str, str]] = json.load(f)

    chat_model = ChatModel(dict(
        model_name_or_path=model_name_or_path,
        finetuning_type=finetuning_type,
        checkpoint_dir=checkpoint_dir,
        template=template
    ))
    chat_model.tokenizer.padding_side = "right" # avoid overflow issue in batched inference for llama2
    eval_template = eval_templates[lang]

    category_corrects: Dict[str, np.ndarray] = {
        subj: np.array([], dtype="bool") for subj in ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
    }
    pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
    results = {}
    for subject in pbar:
        dataset = load_dataset(os.path.join(dataset_dir, task), subject)
        labels, answers, all_outputs = [], [], []
        for epoch in range(n_avg):
            pbar.set_postfix_str("{} Trial: {}".format(categorys[subject]["name"], epoch))
            inputs, outputs = [], []
            for i in trange(len(dataset[split]), desc="Formatting batches", position=1, leave=False):
                support_set = dataset["train"].shuffle().select(range(min(n_shot, len(dataset["train"]))))
                query, resp, history = eval_template.format_example(
                    target_data=dataset[split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                    use_history=chat_model.template.use_history
                )
                input_ids, _ = chat_model.template.encode_oneturn(
                    tokenizer=chat_model.tokenizer, query=query, resp=resp, history=history
                )
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                if epoch == 0:
                    labels.append(resp)

            for i in trange(0, len(inputs), batch_size, desc="Predicting batches", position=1, leave=False):
                batch_input = chat_model.tokenizer.pad(
                    inputs[i : i + batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(chat_model.model.device)
                preds = batch_inference(chat_model, batch_input, eval_template.prefix)
                outputs += preds
            all_outputs.append(outputs)

        for i in range(len(all_outputs[0])):
            count = Counter([all_outputs[epoch][i] for epoch in range(n_avg)])
            answers.append(count.most_common(1)[0][0])

        corrects = (np.array(answers) == np.array(labels))
        category_name = categorys[subject]["category"]
        category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
        category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
        results[subject] = {str(i): answers[i] for i in range(len(answers))}

    score_info = "\n".join([
        "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
        for category_name, category_correct in category_corrects.items() if len(category_correct)
    ])

    print(score_info)
    if save_name is not None:
        with open(save_name + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2)

        with open(save_name + ".log", "w", encoding="utf-8", newline="\n") as f:
            f.write(score_info)


if __name__ == "__main__":
    fire.Fire(evaluate)
