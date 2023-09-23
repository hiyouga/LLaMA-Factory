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
from tqdm import tqdm, trange
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from datasets import load_dataset
from dataclasses import dataclass

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
    probs = torch.nn.functional.softmax(
        torch.stack(
            [
                logits[:, -1, chat_model.tokenizer.encode(prefix_char + "A")[-1]],
                logits[:, -1, chat_model.tokenizer.encode(prefix_char + "B")[-1]],
                logits[:, -1, chat_model.tokenizer.encode(prefix_char + "C")[-1]],
                logits[:, -1, chat_model.tokenizer.encode(prefix_char + "D")[-1]]
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
    batch_size: Optional[int] = 4,
    save_name: Optional[str] = None
):
    with open(os.path.join(dataset_dir, task, "mapping.json"), "r", encoding="utf-8") as f:
        categorys = json.load(f)

    chat_model = ChatModel(dict(
        model_name_or_path=model_name_or_path,
        finetuning_type=finetuning_type,
        checkpoint_dir=checkpoint_dir,
        template=template
    ))
    eval_template = eval_templates[lang]
    assert chat_model.tokenizer.padding_side == "left", "only left-padded tensor can be accepted."

    category_corrects: Dict[str, np.ndarray] = {
        subj: np.array([], dtype="bool") for subj in ["STEM", "Social Sciences", "Humanities", "Other"]
    }
    overall_corrects = np.array([], dtype="bool")

    pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
    results = {}
    for subject in pbar:
        pbar.set_postfix_str(categorys[subject]["name"])
        inputs, labels = [], []
        dataset = load_dataset(os.path.join(dataset_dir, task), subject)
        for i in range(len(dataset[split])):
            support_set = dataset["train"].shuffle().select(range(min(n_shot, len(dataset["train"]))))
            query, resp, history = eval_template.format_example(
                target_data=dataset[split][i],
                support_set=support_set,
                subject_name=categorys[subject]["name"],
                use_history=chat_model.template.use_history
            )
            input_ids, _ = chat_model.template.encode_oneturn(
                tokenizer=chat_model.tokenizer,
                query=query,
                resp=resp,
                history=history
            )
            inputs.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids)
            })
            labels.append(resp)

        outputs = []
        for i in trange(0, len(inputs), batch_size, desc="Processing batches", position=1, leave=False):
            batch_input = chat_model.tokenizer.pad(
                inputs[i : i + batch_size],
                return_attention_mask=True,
                return_tensors="pt"
            ).to(chat_model.model.device)
            preds = batch_inference(chat_model, batch_input, eval_template.prefix)
            outputs += preds

        corrects = (np.array(outputs) == np.array(labels))
        category_name = categorys[subject]["category"]
        category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
        overall_corrects = np.concatenate([overall_corrects, corrects], axis=0)
        results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

    score_info = "Average accuracy: {:.2f}".format(100 * np.mean(overall_corrects))
    for category_name, category_correct in category_corrects.items():
        if len(category_correct):
            score_info += "\n{:>16}: {:.2f}".format(category_name, 100 * np.mean(category_correct))

    print(score_info)
    if save_name is not None:
        with open(save_name + ".json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2)

        with open(save_name + ".log", "w", encoding="utf-8", newline="\n") as f:
            f.write(score_info)


if __name__ == "__main__":
    fire.Fire(evaluate)
