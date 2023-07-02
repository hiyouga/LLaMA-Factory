# coding=utf-8
# Evaluates fine-tuned models automatically.
# Usage: python evaluate_zh.py --evalset ceval/ceval-exam:law --split dev --output_file result.json
#                              --api_base http://localhost:8000/v1 --task_type choice --n_samples 100
# dataset format: question (string), A (string), B (string), C (string), D (string), answer (Literal["A", "B", "C", "D"])


import os
import fire
import json
import openai
from tqdm import tqdm
from typing import Literal, Optional
from datasets import load_dataset


def format_example_choice(examples):
    model_inputs = {"query": [], "label": []}
    task_template = "请从ABCD四个选项中选出正确的选项，仅输出选项序号。\n{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案："
    for i in range(len(examples["id"])):
        query = task_template.format(
            question=examples["question"][i],
            A=examples["A"][i],
            B=examples["B"][i],
            C=examples["C"][i],
            D=examples["D"][i]
        )
        label = examples["answer"][i]
        model_inputs["query"].append(query)
        model_inputs["label"].append(label)
    return model_inputs


def format_example_cloze(examples):
    model_inputs = {"query": [], "label": []}
    task_template = "请选择正确的答案填空，仅输出正确的选项。\n{question}\n选项：{A}\n{B}\n{C}\n{D}\n答案："
    for i in range(len(examples["id"])):
        query = task_template.format(
            question=examples["question"][i],
            A=examples["A"][i],
            B=examples["B"][i],
            C=examples["C"][i],
            D=examples["D"][i]
        )
        label = examples[examples["answer"][i]][i]
        model_inputs["query"].append(query)
        model_inputs["label"].append(label)
    return model_inputs


def format_example_openqa(examples):
    model_inputs = {"query": [], "label": []}
    task_template = "回答以下问题：{question}\n答案："
    for i in range(len(examples["id"])):
        query = task_template.format(question=examples["question"][i])
        label = examples[examples["answer"][i]][i]
        model_inputs["query"].append(query)
        model_inputs["label"].append(label)
    return model_inputs


TASK_DICT = {
    "choice": format_example_choice,
    "cloze": format_example_cloze,
    "openqa": format_example_openqa
}


EXT2TYPE = {
    "csv": "csv",
    "json": "json",
    "jsonl": "json"
}


def evaluate(
        evalset: str,
        api_base: str,
        output_file: str,
        split: Optional[str] = "val",
        task_type: Optional[Literal["choice", "cloze", "openqa"]] = "choice",
        n_samples: Optional[int] = 20
):

    openai.api_base = api_base
    openai.api_key = "none"

    if os.path.isfile(evalset):
        dataset = load_dataset(EXT2TYPE[evalset.split(".")[-1]], data_files=evalset)["train"]
    elif ":" in evalset:
        evalset, subset = evalset.split(":")
        dataset = load_dataset(evalset, subset, split=split)
    else:
        dataset = load_dataset(evalset, split=split)

    n_samples = min(len(dataset), n_samples)

    dataset = dataset.map(TASK_DICT[task_type], batched=True)
    dataset = dataset.select(range(n_samples))

    n_correct = 0
    predictions = []
    for example in tqdm(dataset):
        query, label = example["query"], example["label"]
        predict = openai.ChatCompletion.create(
            model="default",
            messages=[{"role": "user", "content": query}],
            temperature=0.01,
            top_p=0.01,
            max_new_tokens=20
        ).choices[0].message.content

        if task_type == "choice" and predict[0].lower() == label[0].lower():
            n_correct += 1
        if task_type == "cloze" and label in [predict[:len(label)], predict[-len(label):]]:
            n_correct += 1
        if task_type == "openqa" and label in predict:
            n_correct += 1

        predictions.append({
            "query": query,
            "label": label,
            "predict": predict
        })

    print("Result: {}/{}\nAccuracy: {:.2f}%".format(n_correct, n_samples, n_correct / n_samples * 100))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(evaluate)
